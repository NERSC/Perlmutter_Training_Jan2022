# CUDA C++
This directory contains examples for building and running basic CUDA C++ codes on Perlmutter using different compilers. The exercises also cover linking with MPI libraries as well as performing rank to device mappings on a simple code example.

All the examples contain a `batch.sh` in their respective directories that should be used for running the examples through batch submission. Otherwise the run steps mentioned in the examples below can be used directly on interactive allocations. Comment out the following line when doing exercises outside of the reservation window of 9:30-12:30, Jan 5 2022.
```bash 
#SBATCH --reservation=perlmutter_day1
```

## Exercise 1: Simple CUDA C++ program
Most of the times CUDA kernels and code containing CUDA API calls are placed in a file with extension `.cu `. The `nvcc` compiler provided with CUDAtoolkit can by default recognize `.cu` files as containing CUDA code and can link in the required libraries while also compiling the host side code with a host compiler.
This example contains a simple vector addition kernel in `vecAdd.cu` that is called from a main function contained within the same file. This can be built simply with `nvcc` compiler. The build steps have been placed in a `Makefile` placed within the same directory. 

It must be noted that to build this example `nvcc` is being passed a `-arch=sm_80` flag, this is to make sure that code is built for devices with `Compute Capability 8.0` i.e. NVIDIA A100 devices that are available on Perlmutter. To build this example make sure that module `cudatoolkit` has been loaded and then follow the steps below:

```bash
cd Ex-1
make clean
make
```
To run:

```bash
sbatch batch.sh
```
which contains
```
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
 module load cudatoolkit
 module load gcc
 make clean
 make
 ```
 
 To run this:
 
 ```bash
 sbatch batch.sh
 ```
 which contains
 ```
 ./vec_add
 ```
 
 Expected output:
 ```bash
 final result: 1.000000
 ```
 
 ## Exercise 3: Simple CUDA + MPI C++ program
In this example we take the simple CUDA C++ code from Exercise 1 and add the usage of MPI. This example launches `n` MPI ranks and each rank performs the same vector addition operation as in the Exercise 1 by launching a CUDA kernel on one of the `k` devices available on the node and then using an MPI reduce operation results from different ranks are accumulated to verify the correctness. 

Each rank first checks for all the devices visible to it and then assigns itself one of the devices, PCI addresses of all the devices visible to all the ranks are printed out. This is to help understand rank to GPU bindings that are studied in next exercises.

Just like the first exercise, all the code is located in the same file i.e. `vecAdd.cu`. Keeping the file extension as `cu` makes it recgonizable as CUDA containing file by the NVIDIA programming environment and causes it to link `cudart` library simplifying things. When using `PrgEnv-nvidia` the `CC` wrapper by default links in the MPI implementation built for `PrgEnv-nvidia`.

To build and test this example first make sure that `PrgEnv-nvidia` module has been loaded, then follow the steps below:

```bash
# revert to default nvidia env if loaded cudatoolkit, gcc or PrgEnv-gnu from other exercisesm such as Ex-2, or Ex-4. etc.
module load PrgEnv-nvidia
module unload cudatoolkit
module unload gcc
```

```bash
cd Ex-3
make clean
make
```
It must be noted that in line 5 of `Makefile`, flag `-gpu=cc80` is being passed to `CC`, that is to esnure that CUDA code is built for the devices of compute capability 8.0.

To run:

```bash
sbatch batch.sh
```
which contains
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

 ## Exercise 4: Simple CUDA + MPI C++ program, separate compilation.
 Just as described in Exercise 2, users may need to use different compilers for their device code and the main application. When we have MPI in the mix, users can use the MPI build of their choice by loading the correct programming environment and then using the corresponding `CC` wrapper to build the main app and then link with the separately compiled CUDA code. 
 
 In this exercise, we use the same example from previous exercise but breakdown the code in separate files such that all the CUDA code is in a separate file i.e. `kernels.cu` and it is being called by the main app in `vecAdd.cpp` using the header `kernels.h`. It can be observed in the `Makefile` that CUDA code is built using the `nvcc` compiler and then linked using the MPI build of choice. It is important to link `cudart` library to ensure that all the CUDA API references are available. 
 
 Users can try loading different programming environments i.e. `PrgEnv-nvidia`, `PrgEnv-gnu` and test the sample code by following the below instructions. Make sure that you have the module `cudatoolkit` loaded as that is needed to find the path for `cudart` library.
 
 To build with PrgEnv-nvidia:
 
 ```bash
 module load PrgEnv-nvidia
 module load cudatoolkit
 cd Ex-4
 make clean
 make
 ```

 To build with PrgEnv-gnu:
 
 ```bash
 module load PrgEnv-gnu
 module load cudatoolkit
 cd Ex-4
 make clean
 make
 ```
 
 To run:
 ```bash
 sbatch batch.sh
 ```
 which contains
 ```
 srun -n4 ./vec_add
 ```
 
 Expected output:
 
```bash
Rank 3/4 from nid002457 sees 4 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
Rank 1/4 from nid002457 sees 4 GPUs, GPU assigned to me is: = 0000:41:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 2/4 from nid002457 sees 4 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 0/4 from nid002457 sees 4 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 3 GPUs are:
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **

****final result: 1.000000 ******
```

 ## Exercise 5: CUDA +MPI, GPU affinity example.
 Most applications assign one MPI rank per GPU and most of the time it is done using round robin method without considering physical location of the CPU core where the MPI rank is residing with respect to the location of the GPU. In this exercise we will use a simple slurm flag to bind MPI ranks to the GPU located closest to the NUMA region where the MPI rank was scheduled. For this we will use the same code as in Exercise 4 but run it in a different manner. 
 
 First make sure that you have `cudatoolkit` and a `PrgEnv-xx` (either `PrgEnv-nvidia` or `PrgEnv-gnu`) loaded. Then use the below steps to build the code:
 
 To build with PrgEnv-nvidia:
 ```bash
 module load PrgEnv-nvidia
 module load cudatoolkit
 cd Ex-5
 make clean
 make
 ```

 To build with PrgEnv-gnu:
 ```bash
 module load PrgEnv-gnu
 module load cudatoolkit
 cd Ex-5
 make clean
 make
 ```
 
 and then to run the example with regular GPU binding, run with:
 
 ```bash
sbatch batch_reg.sh
```
which contains
```bash
srun -n8 --cpu-bind=cores ./vec_add
```
which contains
```bash
srun -n8 --cpu-bind=cores ./vec_add
```
The above will bind MPI ranks to cores and each MPI rank will have all the GPUs visible. In such a scenerio it is not possible to determine the closest GPUs to any of the MPI ranks (cores) hence assignment is in round robin way. The output from above run command can be inspected to see cores to GPU mappings:

```bash
Rank 1/8 (PID:73658 on Core: 16) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:41:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 5/8 (PID:73662 on Core: 17) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:41:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 0/8 (PID:73657 on Core: 0) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 3 GPUs are:
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 2/8 (PID:73659 on Core: 32) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 3/8 (PID:73660 on Core: 48) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
Rank 6/8 (PID:73663 on Core: 33) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 3: 0000:C1:00.0 **
Rank 7/8 (PID:73664 on Core: 49) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 3 GPUs are:
**rank = 0: 0000:03:00.0 **
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
Rank 4/8 (PID:73661 on Core: 1) from nid003497 sees 4 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 3 GPUs are:
**rank = 1: 0000:41:00.0 **
**rank = 2: 0000:81:00.0 **
**rank = 3: 0000:C1:00.0 **


****final result: 1.000000 ******
```

In order to bind the MPI ranks to the GPUs located closest to the corresponding core (NUMA region), we use the `--gpu-bind=closest` flag, other options for `--gpu-bind` can also be considered to suite each application's need. Rerun the same example using the `--gpu-bind=closest` flag:

```bash
sbatch batch_close.sh
```
which contains
```bash
srun -n8 --cpu-bind=cores --gpu-bind=closest ./vec_add
```

and inspect the output as shown below:

```bash
NUMA node(s):        4
NUMA node0 CPU(s):   0-15,64-79
NUMA node1 CPU(s):   16-31,80-95
NUMA node2 CPU(s):   32-47,96-111
NUMA node3 CPU(s):   48-63,112-127


Rank 1/8 (PID:74002 on Core: 16) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 0 GPUs are:
Rank 3/8 (PID:74004 on Core: 48) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 0 GPUs are:
Rank 5/8 (PID:74006 on Core: 17) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:81:00.0
Other 0 GPUs are:
Rank 7/8 (PID:74008 on Core: 49) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:03:00.0
Other 0 GPUs are:
Rank 0/8 (PID:74001 on Core: 0) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 0 GPUs are:
Rank 4/8 (PID:74005 on Core: 1) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:C1:00.0
Other 0 GPUs are:
Rank 6/8 (PID:74007 on Core: 33) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:41:00.0
Other 0 GPUs are:
Rank 2/8 (PID:74003 on Core: 32) from nid003497 sees 1 GPUs, GPU assigned to me is: = 0000:41:00.0

****final result: 1.000000 ******
```
To view the NUMA regions on current node, use `lscpu | grep NUMA` and observe from the above output that cores in the same NUMA region always view the same GPU. It must be noted that when `--gpu-bind=closest` is used only the closest GPUs are visible to MPI ranks. 

You can also use the sbatch scripts `batch_reg.sh` and `batch_close.sh` to run the examples without and with GPU bindings. These scripts are located in the same folder (`Ex-5`). 
