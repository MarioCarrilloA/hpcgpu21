#!/bin/bash

TITLE="GPU"
MODULE="nvidia/7.5"
GPU="Q4000"

rhrk-launch \
    --title $TITLE \
    --command "bash" \
    --module "$MODULE" \
    --jobslot 1  \
    --gpu $GPU  \
    --time 30 \
    --mode vgl
