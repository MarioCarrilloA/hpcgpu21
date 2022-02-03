#!/bin/bash

#SBATCH -J deviceProperties
#SBATCH -N 1
#SBATCH --gres=gpu:V100:1
#SBATCH -o results-V100-%j.out
#SBATCH -e results-V100-%j.err
#SBATCH -t 30
#SBATCH -n 1

module purge
module add "nvidia/10.0"

BIN=exercise09
make

./${BIN}
