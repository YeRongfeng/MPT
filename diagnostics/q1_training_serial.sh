#!/usr/bin/env bash
# Q1 Stage-1 controlled training: spatial_map/single_12 vs alignment_only.
# Frozen protocol copied from diagnostics/spatial_map_resolution/single_12.
# GPU 0, serial execution, no resume, fresh output dirs.
set -eo pipefail

source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-q1

cd /home/yrf/MPT

test ! -e diagnostics/q1_baseline
test ! -e diagnostics/q1_alignment_only
mkdir -p diagnostics/q1_baseline diagnostics/q1_alignment_only

python3 -u train_compact_stage1.py \
  --architecture spatial_map \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/q1_baseline \
  --seed 20260813 \
  --split-seed 20260814 \
  --train-environments 80 \
  --validation-environments 20 \
  --updates 13750 \
  --eval-every 250 \
  --eval-seed 20260815 \
  --patience-evals 1000 \
  --min-delta 1e-4 \
  --batch-size 16 \
  --validation-contexts 400 \
  --validation-batches 25 \
  --lr 1e-4 \
  2>&1 | tee diagnostics/q1_baseline/run.log

python3 -u train_compact_stage1.py \
  --architecture alignment_only \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/q1_alignment_only \
  --seed 20260813 \
  --split-seed 20260814 \
  --train-environments 80 \
  --validation-environments 20 \
  --updates 13750 \
  --eval-every 250 \
  --eval-seed 20260815 \
  --patience-evals 1000 \
  --min-delta 1e-4 \
  --batch-size 16 \
  --validation-contexts 400 \
  --validation-batches 25 \
  --lr 1e-4 \
  2>&1 | tee diagnostics/q1_alignment_only/run.log
