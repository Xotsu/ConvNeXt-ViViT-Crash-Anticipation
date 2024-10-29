#!/bin/bash

#SBATCH -c8 --mem=32g
#SBATCH --gpus 2
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

export USE_DIST=1
export TEST=0
export TEST_TTA=1

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29500 combined_crash_classifier.py