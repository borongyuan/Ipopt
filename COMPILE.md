This is a guide detailing the compilation of Ipopt with SPRAL as a linear solver.
It was developed assuming a standard installation of Ubuntu 18.04 LTS.
To begin, first, compile the [LANL ANSI version of SPRAL](https://github.com/lanl-ansi/spral) using the [compilation suggestions described therein](https://github.com/lanl-ansi/spral/blob/master/COMPILE.md).

## Cloning the Repository
First, create a directory where Ipopt will be compiled from source (not via `coinbrew`), e.g.,
```bash
mkdir -p ${HOME}/Software
```
The remainder of this guide assumes such a directory has been created.
Then, clone the Ipopt repository via
```bash
cd ${HOME}/Software
git clone https://github.com/lanl-ansi/Ipopt.git
```

## Rebuilding Configuration Files (optional)
To rebuild configuration files for Ipopt, if needed (e.g., during development), execute
```bash
cd ${HOME}/Software/Ipopt
git clone https://github.com/coin-or-tools/BuildTools.git
export COIN_AUTOTOOLS_DIR="${HOME}/local2"
./BuildTools/install_autotools.sh
./BuildTools/run_autotools
```
If you are not modifying the directory or source structure of Ipopt, this step is not required.

## Compilation with SPRAL
### Multicore CPUs Only
To compile Ipopt with SPRAL (CPU support only), specify `${SPRALDIR}` as the directory containing `lib/libspral.a`, then execute
```bash
cd ${HOME}/Software/Ipopt
mkdir build
cd build
../configure --prefix=${PWD} --with-spral="-L${SPRALDIR}/lib -L${METISDIR}/lib \
    -lspral -lgfortran -lhwloc -lm -lcoinmetis -lopenblas -lstdc++ -fopenmp" \
    --with-lapack-lflags="-llapack -lopenblas"
make && make install
```

### Multicore CPUs and NVIDIA GPUs
To compile with GPU support, execute
```bash
cd ${HOME}/Software/Ipopt
mkdir build
cd build
../configure --prefix=${PWD} --with-spral="-L${SPRALDIR}/lib -L${METISDIR}/lib \
    -lspral -lgfortran -lhwloc -lm -lcoinmetis -lopenblas -lstdc++ -fopenmp \
    -lcudadevrt -lcudart -lcuda -lcublas" --with-lapack-lflags="-llapack -lopenblas"
make && make install
```

## Usage
Ensure the following environment variables are set when using the SPRAL library:
```bash
export OMP_CANCELLATION=TRUE
export OMP_NESTED=TRUE
export OMP_PROC_BIND=TRUE
```

## Testing
Within the `build` directory created above, the `examples/ScalableProblems` directory contains a set of scalable test problems.
After compilation of Ipopt, these examples can be compiled via
```bash
cd ${HOME}/Software/Ipopt/build
cd examples/ScalableProblems && make
```
As an example, if Ipopt was compiled with SPRAL support, try creating a file named `ipopt.opt` in this directory with the contents
```
linear_solver spral
spral_use_gpu no
```
Then, solve a test problem, e.g.,
```bash
time ./solve_problem MBndryCntrl1 768
```
If SPRAL was compiled with GPU support, next try modifying the `ipopt.opt` file to contain
```
linear_solver spral
spral_use_gpu yes
```
Then, solve the same test problem, e.g.,
```bash
time ./solve_problem MBndryCntrl1 768
```
The `real` time required by the solver should typically decrease on very large, dense problems, compared with a solve using `spral_use_gpu no`.
If this is not the case, ensure your GPU is actually being recognized by SPRAL.
This issue is briefly discussed near the end of the associated  [SPRAL compilation guide](https://github.com/lanl-ansi/spral/blob/master/COMPILE.md#multicore-cpus-and-nvidia-gpus-optional).
