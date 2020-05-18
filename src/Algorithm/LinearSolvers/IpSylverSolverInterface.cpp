// Copyright (C) 2020, Los Alamos National Laboratory
// Copyright (C) 2012, The Science and Technology Facilities Council.
// Copyright (C) 2009, Jonathan Hogg <jhogg41.at.gmail.com>.
// Copyright (C) 2004, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSylverSolverInterface.cpp 2020-03-21 00:00:00Z tasseff $
//
// Authors: Byron Tasseff                    LANL   2020-05-17
//          Jonathan Hogg                    STFC   2012-12-21
//          Jonathan Hogg                           2009-07-29
//          Carl Laird, Andreas Waechter     IBM    2004-03-17

#include "IpoptConfig.h"
#include "IpSylverSolverInterface.hpp"
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <cmath>

using namespace std;

namespace Ipopt
{

SylverSolverInterface::~SylverSolverInterface()
{
   delete[] val_;

   if ( scaling_ )
   {
      delete[] scaling_;
   }

   sylver_ssids_free(&akeep_, &fkeep_);
}

void SylverSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{
   roptions->AddLowerBoundedIntegerOption(
      "sylver_cpu_block_size", "CPU Parallelization Block Size", 1, 256,
      "Block size to use for parallelization of large nodes on CPU resources.");

   roptions->AddLowerBoundedNumberOption(
      "sylver_gpu_perf_coeff", "GPU Performance Coefficient", 0.0, true, 1.0,
      "How many times faster a GPU is than CPU at factoring a subtree.");

   roptions->AddStringOption2(
      "sylver_ignore_numa", "NUMA Region Setting.",
      "yes", "no", "Do not treat CPUs and GPUs as belonging to a single NUMA region.",
      "yes", "Treat CPUs and GPUs as belonging to a single NUMA region.", "");

   roptions->AddLowerBoundedNumberOption(
      "sylver_max_load_inbalance",
      "Maximum Permissible Load", 1.0, true, 1.2,
      "Maximum permissible load inbalance for leaf subtree allocations.");

   roptions->AddLowerBoundedNumberOption(
      "sylver_min_gpu_work", "Minimum GPU Work", 0.0, false, 5.0e9,
      "Minimum number of flops in subtree before scheduling on GPU.");

   roptions->AddLowerBoundedIntegerOption(
      "sylver_nemin", "Node Amalgamation Parameter", 1, 32,
      "Two nodes in elimination tree are merged if result has fewer than "
      "sylver_nemin variables.");

   roptions->AddStringOption2(
      "sylver_order",
      "Controls type of ordering used by SYLVER", "matching",
      "metis", "Use METIS with default settings.",
      "matching", "Use matching-based elimination ordering.", "");

   roptions->AddStringOption3(
      "sylver_pivot_method",
      "Specifies strategy for scaling in SYLVER linear solver.", "block",
      "aggressive", "Aggressive a posteori pivoting.",
      "block", "Block a posteori pivoting.",
      "threshold", "Threshold partial pivoting (not parallel).", "");

   roptions->AddIntegerOption(
      "sylver_print_level", "Print level for the linear solver SYLVER", -1, ""
      /*
       "<0 No printing.\n"
       "0  Error and warning messages only.\n"
       "=1 Limited diagnostic printing.\n"
       ">1 Additional diagnostic printing."*/);

   roptions->AddStringOption5(
      "sylver_scaling",
      "Specifies strategy for scaling in SYLVER linear solver.", "matching",
      "none", "Do not scale the linear system matrix.",
      "mc64", "Scale using weighted bipartite matching (MC64).",
      "auction", "Scale using the auction algorithm.",
      "matching", "Scale using the matching-based ordering.",
      "ruiz", "Scale using the norm-equilibration algorithm of Ruiz (MC77).",
      "");

   roptions->AddLowerBoundedNumberOption(
      "sylver_small", "Zero Pivot Threshold", 0.0, true, 1.0e-20,
      "Any pivot less than sylver_small is treated as zero.");

   roptions->AddLowerBoundedNumberOption(
      "sylver_small_subtree_threshold", "Small Subtree Threshold", 0.0, true, 4.0e6,
      "Maximum number of flops in a subtree treated as a single task.");

   roptions->AddBoundedNumberOption(
      "sylver_u",
      "Pivoting Threshold", 0.0, true, 0.5, false, 1.0e-8,
      "Relative pivot threshold used in symmetric indefinite case.");

   roptions->AddBoundedNumberOption(
      "sylver_umax", "Maximum Pivoting Threshold", 0.0, true, 0.5, false, 1.0e-4,
      "See SYLVER documentation.");

   roptions->AddStringOption2(
      "sylver_use_gpu", "GPU Setting",
      "yes", "no", "Do not use NVIDIA GPUs.",
      "yes", "Use NVIDIA GPUs if present.", "");
}

int SylverSolverInterface::PivotMethodNameToNum(
   const std::string& name
)
{
   if ( name == "aggressive" )
   {
      return 0;
   }
   else if ( name == "block" )
   {
      return 1;
   }
   else if ( name == "threshold" )
   {
      return 2;
   }
   else
   {
      assert(0);
      return -1;
   }
}

int SylverSolverInterface::ScaleNameToNum(
   const std::string& name
)
{
   if ( name == "none" )
   {
      return 0;
   }
   else if ( name == "mc64" )
   {
      return 1;
   }
   else if ( name == "auction" )
   {
      return 2;
   }
   else if ( name == "matching" )
   {
      return 3;
   }
   else if ( name == "ruiz" )
   {
      return 4;
   }
   else
   {
      assert(0);
      return -1;
   }
}

bool SylverSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   sylver_ssids_default_options(&control_);
   control_.array_base = 0; // Use Fortran numbering (documentation incorrect).
   control_.action = true; // Continue factorization on discovery of a zero pivot.
   /* Note: we can't set control_.action = false as we need to know the
    * inertia. (Otherwise we just enter the restoration phase and fail.) */

   options.GetBoolValue("sylver_ignore_numa", control_.ignore_numa, prefix);
   options.GetBoolValue("sylver_use_gpu", control_.use_gpu, prefix);
   options.GetIntegerValue("sylver_cpu_block_size", control_.cpu_block_size, prefix);
   options.GetIntegerValue("sylver_nemin", control_.nemin, prefix);
   options.GetIntegerValue("sylver_print_level", control_.print_level, prefix);
   options.GetNumericValue("sylver_small", control_.small, prefix);
   options.GetNumericValue("sylver_u", control_.u, prefix);
   options.GetNumericValue("sylver_umax", umax_, prefix);

   // Set gpu_perf_coeff.
   double gpu_perf_coeff_tmp = 1.0;
   options.GetNumericValue("sylver_gpu_perf_coeff", gpu_perf_coeff_tmp, prefix);
   control_.gpu_perf_coeff = (float)gpu_perf_coeff_tmp;

   // Set max_load_inbalance.
   double max_load_inbalance_tmp = 1.2;
   options.GetNumericValue("sylver_max_load_inbalance", max_load_inbalance_tmp, prefix);
   control_.max_load_inbalance = (float)max_load_inbalance_tmp;

   // Set min_gpu_work.
   double min_gpu_work_tmp = 5.0e9;
   options.GetNumericValue("sylver_min_gpu_work", min_gpu_work_tmp, prefix);
   control_.min_gpu_work = (long)min_gpu_work_tmp;

   // Set the pivot method.
   std::string pivot_method;
   options.GetStringValue("sylver_pivot_method", pivot_method, prefix);
   control_.pivot_method = PivotMethodNameToNum(pivot_method);

   // Set small_subtree_threshold.
   double small_subtree_threshold_tmp = 4.0e6;
   options.GetNumericValue("sylver_small_subtree_threshold", small_subtree_threshold_tmp, prefix);
   control_.small_subtree_threshold = (long)small_subtree_threshold_tmp;

   // Reset all private data.
   pivtol_changed_ = false;

   std::string order_method;
   options.GetStringValue("sylver_order", order_method, prefix);

   if (order_method == "metis")
   {
      control_.ordering = 1;
   }
   else if (order_method == "matching")
   {
      control_.ordering = 2;
   }

   std::string scaling_method;
   options.GetStringValue("sylver_scaling", scaling_method, prefix);
   current_level_ = 0;
   scaling_type_ = ScaleNameToNum(scaling_method);

   // Set scaling and ordering.
   control_.scaling = scaling_type_;
   control_.ordering = scaling_type_ != 3 ? 1 : 2;

   return true; // All is well.
}

ESymSolverStatus SylverSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   struct sylver_ssids_inform info;

   // Store size for later use
   ndim_ = dim;

   // Setup memory for values
   if ( val_ != NULL )
   {
      delete[] val_;
   }

   val_ = new double[nonzeros];

   // Correct scaling and ordering if necessary.
   if ( control_.ordering == 2 && control_.scaling != 3 )
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface, "
                     "matching-based ordering was used, but matching-based scaling was "
                     "not. Setting scaling using the matching-based ordering.\n");
      control_.scaling = scaling_type_ = 3;
   }

   if ( control_.ordering != 2 && control_.scaling == 3 )
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface, "
                     "matching-based scaling was used, but matching-based ordering was "
                     "not. Setting ordering using the matching-based algorithm.\n");
      control_.ordering = 2;
   }

   // Perform analyse.
   if ( !( control_.ordering == 2 && control_.scaling == 3 ) )
   {
      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
      }

      sylver_ssids_analyse_ptr32(false, ndim_, NULL, ia, ja, NULL, &akeep_, &control_, &info);

      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "nfactor = %d, nflops = %d:\n",
                     info.num_factor, info.num_flops);

      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemSymbolicFactorization().End();
      }

      if ( info.flag >= 0 )
      {
         return SYMSOLVER_SUCCESS;
      }
      else
      {
         return SYMSOLVER_FATAL_ERROR;
      }
   }
   else
   {
      return SYMSOLVER_SUCCESS;
   }
}

ESymSolverStatus SylverSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* ia,
   const Index* ja,
   Index        nrhs,
   double*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   struct sylver_ssids_inform info;
   Number t1 = 0, t2;

   if ( new_matrix || pivtol_changed_ )
   {
      // Set scaling option
      if ( rescale_ )
      {
         control_.scaling = scaling_type_;
         control_.ordering = scaling_type_ != 3 ? 1 : 2;

         if ( scaling_type_ != 0 && scaling_ == NULL )
         {
            scaling_ = new double[ndim_];
         }
      }
      else
      {
         control_.scaling = 0; // None or user (depends if scaling_ is allocated).
      }

      if ( control_.ordering == 2 && control_.scaling == 3 && rescale_ )
      {
         if ( HaveIpData() )
         {
            IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
         }

         sylver_ssids_analyse_ptr32(false, ndim_, NULL, ia, ja, val_, &akeep_, &control_, &info);

         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "nfactor = %d, nflops = %d:\n",
                        info.num_factor, info.num_flops);

         if ( HaveIpData() )
         {
            IpData().TimingStats().LinearSystemSymbolicFactorization().End();
         }

         if ( info.flag == 6 || info.flag == -5 )
         {
            Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface::Factorization: "
                           "Singular system, estimated rank %d of %d\n", info.matrix_rank, ndim_);
            return SYMSOLVER_SINGULAR;
         }
         else if ( info.flag < 0 )
         {
            return SYMSOLVER_FATAL_ERROR;
         }
      }

      if ( HaveIpData() )
      {
         t1 = IpData().TimingStats().LinearSystemFactorization().TotalWallclockTime();
         IpData().TimingStats().LinearSystemFactorization().Start();
      }

      sylver_ssids_factor_ptr32(false, ia, ja, val_, scaling_, akeep_, &fkeep_, &control_, &info);

      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "SYLVER: delays %d, nfactor %d, "
                     "nflops %ld, maxfront %d\n", info.num_delay, info.num_factor, info.num_flops,
                     info.maxfront);

      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
         t2 = IpData().TimingStats().LinearSystemFactorization().TotalWallclockTime();
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "SylverSolverInterface::Factorization: "
                        "sylver_factor_solve took %10.3f\n", t2 - t1);
      }

      if ( info.flag == 7 || info.flag == 6 || info.flag == -5 )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface::Factorization: "
                        "Singular system, estimated rank %d of %d\n", info.matrix_rank, ndim_);
         return SYMSOLVER_SINGULAR;
      }

      if ( info.flag < 0 )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface::Factorization: "
                        "Unhandled error. info.flag = %d\n", info.flag);
         return SYMSOLVER_FATAL_ERROR;
      }

      if ( check_NegEVals && info.num_neg != numberOfNegEVals )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "In SylverSolverInterface::Factorization: "
                        "info.num_neg = %d, but numberOfNegEVals = %d\n", info.num_neg, numberOfNegEVals);
         return SYMSOLVER_WRONG_INERTIA;
      }

      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemBackSolve().Start();
      }

      sylver_ssids_solve(0, nrhs, rhs_vals, ndim_, akeep_, fkeep_, &control_, &info);

      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemBackSolve().End();
      }

      numneg_ = info.num_neg;

      pivtol_changed_ = false;
   }
   else
   {
      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemBackSolve().Start();
      }

      sylver_ssids_solve(0, nrhs, rhs_vals, ndim_, akeep_, fkeep_, &control_, &info);

      if ( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemBackSolve().End();
      }
   }

   return SYMSOLVER_SUCCESS;
}

bool SylverSolverInterface::IncreaseQuality()
{
   if ( control_.u >= umax_ )
   {
      return false;
   }

   pivtol_changed_ = true;
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "Increasing pivot tolerance "
                  "for SYLVER from %7.2e ", control_.u);
   control_.u = Min(umax_, pow(control_.u, 0.75));
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA, "to %7.2e.\n", control_.u);
   return true;
}

} // namespace Ipopt
