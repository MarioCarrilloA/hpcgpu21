#!/bin/bash

TITLE="GPU"
MODULE="nvidia/7.5"

rhrk-launch --title $TITLE --command "bash" --module $MODULE --mode vgl
