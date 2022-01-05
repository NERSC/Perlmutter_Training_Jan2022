# OpenMP / OpenACC C++
This directory contains examples for building and running OpenMP/OpenACC C++ codes on Perlmutter using the `nvhpc` compiler. 
The exercises also cover linking with MPI libraries as well as performing rank to device mappings on a simple code example.

## Exercise 1: Simple OpenMP/OpenACC C++ program
Both OpenMP and OpenACC are directive based programming frameworks. `nvhpc` (Nvidia) compiler supports both the frameworks. 
Hence we use `PrgEnv-nvidia` which is the default programming environment on Perlmutter.

The examples in this section mirror the CUDA examples but in OpenMP and OpenACC programming frameworks.
Each example has a Makefile which builds the sequential version by default. 
To build the OpenMP/OpenACC versions use the following build command: 
``` console
make clean OPENMP=y 
make OPENMP=y 
or
make clean OPENACC=y
make OPENACC=y
```
