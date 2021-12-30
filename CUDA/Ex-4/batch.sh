#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH -A ntrain3_g
#SBATCH --reservation=perlmutter_day1

srun -n4 ./vec_add
