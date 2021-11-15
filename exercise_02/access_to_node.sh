#!/bin/bash

TITLE="GPU"
MODULE="nvidia/7.5"
GPU="P2000"

rhrk-launch \
    --title $TITLE \
    --command "bash" \
    --module "$MODULE" \
    --jobslot 1  \
    --gpu $GPU  \
    --time 30 \
    --mode vgl
