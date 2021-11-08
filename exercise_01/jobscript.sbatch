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

NUM_FUNCS=6
FUNCTIONS=(
    "GPU Kernel with floats"
    "GPU Kernel with doubles"
    "CPU host with floats"
    "CPU host with doubles"
    "GPU Kernel with floats, fabs instead of sqrt, pow"
    "GPU Kernel with dubles, fabs instead of sqrt, pow"
)
BIN_EX="exercise01"
NITERS=10

echo "Executing on $HOSTNAME"
if [ $BUILD_MODE == "fast-math" ]; then
    echo "Building with ${BUILD_MODE}!!!"
    make fast-math
else
    make all
fi

if [ ! -f "$BIN_EX" ]; then
    echo "Binary $BIN_EX was not built" 1>&2
fi

echo -e "RESULTS:\n"
for (( i=0; i<$NUM_FUNCS; i++ )); do
    FUNC_ID=$((i + 1))
    echo "==================================================="
    echo "  ${FUNCTIONS[$i]}"
    echo "==================================================="
    ./${BIN_EX} -f $FUNC_ID -n $NITERS
    echo -e "---------------------------------------------------\n"
done

sleep 1
