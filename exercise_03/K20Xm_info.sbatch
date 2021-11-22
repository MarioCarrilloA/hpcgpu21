#!/bin/bash

#SBATCH -J deviceProperties
#SBATCH -N 1
#SBATCH --gres=gpu:K20Xm:2
#SBATCH -o devicePropertiesK20Xm-%j.out
#SBATCH -e devicePropertiesK20Xm-%j.err
#SBATCH -t 30
#SBATCH -n 1
#SBATCH --mail-type=END
module purge
module add "nvidia/10.0"

DEVINFO_BIN=DevProperties

make extras

./$DEVINFO_BIN
