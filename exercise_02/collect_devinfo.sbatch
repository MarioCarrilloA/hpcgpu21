#!/bin/bash

#SBATCH -J deviceProperties
#SBATCH -N 1
#SBATCH --gres=gpu:K20m:1
#SBATCH -o deviceProperties-%j.out
#SBATCH -e deviceProperties-%j.err
#SBATCH -t 30
#SBATCH -n 1
#SBATCH --mail-type=END
module purge
module add "nvidia/7.5"

DEVINFO_BIN=DevProperties

make extras

./$DEVINFO_BIN
