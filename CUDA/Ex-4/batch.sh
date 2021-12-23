#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH -A m3902_g

srun -n4 --gpu-bind=map_gpu:0,1,2,3 ./vec_add