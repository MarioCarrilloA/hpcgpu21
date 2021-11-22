#!/bin/bash

#SBATCH -J deviceProperties
#SBATCH -N 1
#SBATCH --gres=gpu:V100:1
#SBATCH -o devicePropertiesV100-%j.out
#SBATCH -e devicePropertiesV100-%j.err
#SBATCH -t 30
#SBATCH -n 1
#SBATCH --mail-type=END
module purge
module add "nvidia/10.0"

DEVINFO_BIN=DevProperties

make extras

./$DEVINFO_BIN
