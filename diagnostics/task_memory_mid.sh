#!/usr/bin/env bash
# Continue baseline and task_memory from 2500 to 5000. Does not overwrite
# the original 2500-run directories.
set -eo pipefail

source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-task-memory-mid

cd /home/yrf/MPT

test -f diagnostics/path_xy_screen/baseline/stage1_last.pth
test -f diagnostics/task_memory_screen/task_memory/stage1_last.pth
test ! -e diagnostics/task_memory_mid/baseline
test ! -e diagnostics/task_memory_mid/task_memory
mkdir -p diagnostics/task_memory_mid/baseline diagnostics/task_memory_mid/task_memory

python3 -u train_compact_stage1.py \
  --architecture spatial_map \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/task_memory_mid/baseline \
  --resume diagnostics/path_xy_screen/baseline/stage1_last.pth \
  --seed 20260813 \
  --split-seed 20260814 \
  --train-environments 80 \
  --validation-environments 20 \
  --updates 5000 \
  --eval-every 250 \
  --save-every 250 \
  --eval-seed 20260815 \
  --patience-evals 1000 \
  --min-delta 1e-4 \
  --batch-size 16 \
  --validation-contexts 400 \
  --validation-batches 25 \
  --lr 1e-4 \
  2>&1 | tee diagnostics/task_memory_mid/baseline/run.log

python3 -u train_compact_stage1.py \
  --architecture task_memory \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/task_memory_mid/task_memory \
  --resume diagnostics/task_memory_screen/task_memory/stage1_last.pth \
  --seed 20260813 \
  --split-seed 20260814 \
  --train-environments 80 \
  --validation-environments 20 \
  --updates 5000 \
  --eval-every 250 \
  --save-every 250 \
  --eval-seed 20260815 \
  --patience-evals 1000 \
  --min-delta 1e-4 \
  --batch-size 16 \
  --validation-contexts 400 \
  --validation-batches 25 \
  --lr 1e-4 \
  2>&1 | tee diagnostics/task_memory_mid/task_memory/run.log
