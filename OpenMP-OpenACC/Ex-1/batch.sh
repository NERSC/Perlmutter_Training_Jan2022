#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH -A ntrain3_g
#SBATCH --reservation=perlmutter_day1

# setting NVCOMPILER_ACC_NOTIFY at runtime will print extra info about
# kernel launches and data transfers
# see https://docs.nersc.gov/performance/readiness/#runtime-environment-variables
export NVCOMPILER_ACC_NOTIFY=3

./vec_add.seq

./vec_add.openmp

./vec_add.openacc
