#!/usr/bin/env bash
# 2500-update screen for shared_xy only.
# Reuses the already completed baseline and path_xy 2500-update checkpoints.
set -eo pipefail

source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-shared-xy

cd /home/yrf/MPT

test -f diagnostics/path_xy_screen/baseline/summary.json
test -f diagnostics/path_xy_screen/path_xy/summary.json
test ! -e diagnostics/path_xy_screen/shared_xy
mkdir -p diagnostics/path_xy_screen/shared_xy

python3 -u train_compact_stage1.py \
  --architecture shared_xy \
  --map-layout single_12 \
  --path-xy-alpha 0.1 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/path_xy_screen/shared_xy \
  --seed 20260813 \
  --split-seed 20260814 \
  --train-environments 80 \
  --validation-environments 20 \
  --updates 2500 \
  --eval-every 250 \
  --eval-seed 20260815 \
  --patience-evals 1000 \
  --min-delta 1e-4 \
  --batch-size 16 \
  --validation-contexts 400 \
  --validation-batches 25 \
  --lr 1e-4 \
  2>&1 | tee diagnostics/path_xy_screen/shared_xy/run.log
