# CUDA-aware MPI

Cray-MPICH is a CUDA-aware MPI implementation, but the CUDA-aware transport
layer is not included in the executable by default. To build with 
CUDA-awareness, load the relevant `craype-accel-*` module before linking.
(For Perlmutter, this is `craype-accel-nvidia80`)

module load PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
make

You can check whether an executable has support for CUDA-aware MPI with `ldd`:
look for a library like "libmpi_gtl_cuda.so".

At run time, you must also enable CUDA-aware MPI with:
export MPICH_GPU_SUPPORT_ENABLED=1

