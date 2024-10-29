#!/bin/bash

#SBATCH -c8 --mem=32g
#SBATCH --gpus 1
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

export USE_DIST=1
export TEST=1
export TEST_TTA=1

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29500 combined_crash_classifier.py