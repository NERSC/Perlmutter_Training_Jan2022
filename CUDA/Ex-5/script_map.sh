#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 1
#SBATCH --ntasks-per-node=8
#SBATCH -A m3902_g
#SBATCH --exclusive

lscpu

srun -n8 --cpu-bind=map_cpu:2,4,6,8,10,12,14,16 ./vec_add


