#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 1
#SBATCH --ntasks-per-node=8
#SBATCH -A m3902_g
#SBATCH --exclusive

lscpu

srun -n8 --cpu-bind=cores ./vec_add

