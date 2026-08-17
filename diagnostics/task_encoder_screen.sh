#!/usr/bin/env bash
# 2500-update screen: reuse Q1-protocol baseline, train task_encoder only.
set -eo pipefail

source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-task-encoder

cd /home/yrf/MPT

test -f diagnostics/path_xy_screen/baseline/summary.json
test ! -e diagnostics/task_encoder_screen/task_encoder
mkdir -p diagnostics/task_encoder_screen/task_encoder

python3 -u train_compact_stage1.py \
  --architecture task_encoder \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/task_encoder_screen/task_encoder \
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
  2>&1 | tee diagnostics/task_encoder_screen/task_encoder/run.log
