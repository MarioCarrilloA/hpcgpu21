#!/bin/bash
#SBATCH -J ExerciseJob
#SBATCH -N 1
#SBATCH --gres=gpu:K20m:1
#SBATCH -o ExerciseJob-%j.out
#SBATCH -e ExerciseJob-%j.err
#SBATCH -t 30
#SBATCH --mail-type=END

echo "Executing on $HOSTNAME"
./exercise00

sleep 5
