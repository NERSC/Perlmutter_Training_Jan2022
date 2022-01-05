#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH -A ntrain3_g
#SBATCH --reservation=perlmutter_day1

ml PrgEnv-nvidia
ml cudatoolkit

echo "first we'll build without cuda-aware MPI:"
make clean ; make
ldd bcast_from_device
echo ""
echo "running this will result in an error:"
srun -n 2 ./bcast_from_device

echo "to build it for CUDA-aware MPI we need an accel target set:"
ml craype-accel-nvidia80
make clean ; make
echo ""
echo "notice that now libmpi_gtl_cuda is linked:"
ldd bcast_from_device
echo "You still must enable CUDA-aware MPI at run time with:"
export MPICH_GPU_SUPPORT_ENABLED=1
srun -n 2 ./bcast_from_device

