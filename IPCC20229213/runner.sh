#!/bin/bash

source /public1/soft/modules/module.sh
module add mpi/intel/17.0.5
module add mpich/3.1.4-intel-17.0.5
mpiicc -v
mpiicc ./pivot.c -mcmodel=medium -fopenmp -O3 -Wall -lm -o pivot.out -std=c11
# srun -p IPCC -N 2 -n 128 ./pivot.out uniformvector-4dim-1h.txt
srun -p IPCC -N 2 -n 128 ./pivot.out $1

