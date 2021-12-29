# CUDA C++
This directory contains examples for building and running basic CUDA C++ codes on Perlmutter using different compilers. The exercises also cover linking with MPI libraries as well as performing rank to device mappings on a simple code example.

## Exercise 1: Simple CUDA C++ program
Most of the times CUDA kernels and code containing CUDA API calls are placed in a file with extension `.cu `. The `nvcc` compiler provided with CUDAtoolkit can by default recognize `.cu` files as containing CUDA code and can link in the required libraries while also compiling the host side code with a host compiler.
This example contains a simple vector addition kernel in `vecAdd.cu` that is called from a main function contained within the same file. This can be built simply with `nvcc` compiler. The build steps have been placed in a `Makefile` placed within the same directory. 

It must be noted that to build this example `nvcc` is being passed a `-arch=sm_80` flag, this is to make sure that code is built for devices with `Compute Capability 8.0` i.e. NVIDIA A100 devices that are available on Perlmutter. To build this example make sure that module `cudatoolkit` has been loaded and then follow the steps below:

```bash
cd Ex-1
make
```
To run:

```bash
./vec_add
```
The output should look like:
```bash
final result: 1.000000
```

## Exercise 2: Simple CUDA C++ program, separate compilation.
