#!/bin/bash
#SBATCH -J ExerciseJob
#SBATCH -N 1
#SBATCH --gres=gpu:K20m:1
#SBATCH -o ExerciseJob-%j.out
#SBATCH -e ExerciseJob-%j.err
#SBATCH -t 30
#SBATCH --mail-type=END
module purge
module add "nvidia/7.5"

echo "Executing on $HOSTNAME"
pwd

echo "Building program"
make all

echo "Executing program"
echo "-----------------"
./exercise00

sleep 5
