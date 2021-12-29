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
 The case presented in Exercise 1 is a rare occurence when working with a large code base, most of the time device (CUDA) code is kept completely separate. Moreover, application may require the use of a compiler different than nvcc. This requires the build phase to be divided in separate parts where device code is built separately using `nvcc` compiler and later linked with the main app using the compiler of choice. In addition to linking the device code we also link the `cudart` library to make sure all the CUDA calls ae taken care of. This exercise demonstrates a similar scenerio by breaking down the code from Exercise 1 into different file such that all the CUDA code is located in `kernels.cu` and it is called in the `vecAdd.cpp` file inside `main` function using headers available in `kernels.h`.
 
To build this example make sure that module `cudatoolkit`is loaded and `g++` is available. The build steps are listed in `Makefile` located within the same directory. We build the CUDA code using same `nvcc` compiler and later link the object file with the main executable using `g++`. Users can try different compilers in place of `g++` to verify the flexibility of this method. To build this example follow the steps below:
 
 ```bash
 cd Ex-2
 make
 ```
 
 To run this:
 
 ```bash
 ./vec_add
 ```
 
 Expected output:
 ```bash
 final result: 1.000000
 ```
 
 ## Exercise 3: Simple CUDA + MPI C++ program
In this example we take the simple CUDA C++ code from Exercise 1 and add the usage of MPI. This example launches `n` MPI ranks and each rank performs the same vector addition operation as in the Exercise 1 by launching a CUDA kernel on one of the `k` devices available on the node and then using an MPI reduce operation results from different ranks are accumulated to verify the correctness. 

Each rank first checks for all the devices visible to it and then assigns itself one of the devices, PCI addresses of all the devices visible to all the ranks are printed out. This is to help understand rank to GPU bindings that are studied in next exercises.

Just like the first exercise, all the code is located in the same file i.e. `vecAdd.cu`. Keeping the file extension as `cu` makes it recgonizable as CUDA containing file by the NVIDIA programming environment and causes it to link `cudart` library simplifying things. When using `PrgEnv-nvidia` the `CC` wrapper by default links in the MPI implementation built for `PrgEnv-nvidia`

To build and test this example first make sure that `PrgEnv-nvidia` module has been loaded, then follow the steps below:

```bash
cd Ex-3
make
```

To run:

```bash
srun -n4 ./vec_add
```

Expected output:
```bash
Rank 3/4 from nid003053 sees 4 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
Rank 0/4 from nid003053 sees 4 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 3 GPUs are:
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 1/4 from nid003053 sees 4 GPUs, GPU assigned to me is: = 0000:41:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 2/4 from nid003053 sees 4 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 3: 0000:C1:00.0 **

****final result: 1.000000 ******
```
