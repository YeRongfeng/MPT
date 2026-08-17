#!/usr/bin/env bash
# 2500-update Stage-1 architecture screen: spatial_map/single_12 vs path_xy.
# Frozen protocol copied from diagnostics/q1_protocol.json, except updates=2500.
# Does not launch Q2/Q3 or 13750-update training.
set -eo pipefail

source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-path-xy

cd /home/yrf/MPT

test ! -e diagnostics/path_xy_screen/baseline
test ! -e diagnostics/path_xy_screen/path_xy
mkdir -p diagnostics/path_xy_screen/baseline diagnostics/path_xy_screen/path_xy

python3 -u train_compact_stage1.py \
  --architecture spatial_map \
  --map-layout single_12 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/path_xy_screen/baseline \
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
  2>&1 | tee diagnostics/path_xy_screen/baseline/run.log

python3 -u train_compact_stage1.py \
  --architecture path_xy \
  --map-layout single_12 \
  --path-xy-alpha 0.1 \
  --dataFolder data/dataset1 \
  --output-dir diagnostics/path_xy_screen/path_xy \
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
  2>&1 | tee diagnostics/path_xy_screen/path_xy/run.log
