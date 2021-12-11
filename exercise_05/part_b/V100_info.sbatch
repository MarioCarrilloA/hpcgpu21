#!/bin/bash

#SBATCH -J deviceProperties
#SBATCH -N 1
#SBATCH --gres=gpu:V100:1
#SBATCH -o results-V100-%j.out
#SBATCH -e results-V100-%j.err
#SBATCH -t 30
#SBATCH -n 1
#SBATCH --mail-type=END
module purge
module add "nvidia/10.0"

BIN=exercise05_b1

make

echo "(m, n, k)"
echo "(4096,4096,1024)"
./$BIN -m 4096 -n 4096 -k 1024

echo "(4096,1024,2048)"
./$BIN -m 4096 -n 1024 -k 2048

echo "(4096,4096,4096)"
./$BIN -m 4096 -n 4096 -m 4096
