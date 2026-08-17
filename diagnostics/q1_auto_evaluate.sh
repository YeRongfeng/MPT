#!/usr/bin/env bash
# Waits until both Q1 training arms write completed summaries, then runs the
# frozen evaluation once on GPU 0. Training itself remains untouched.
set -eo pipefail
cd /home/yrf/MPT
while true; do
  if [ -f diagnostics/q1_baseline/summary.json ] && [ -f diagnostics/q1_alignment_only/summary.json ]; then
    if python3 - <<'PY'
import json
for p in ("diagnostics/q1_baseline/summary.json","diagnostics/q1_alignment_only/summary.json"):
    if json.load(open(p)).get("status") != "completed":
        raise SystemExit(1)
PY
    then
      break
    fi
  fi
  sleep 60
done
source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-q1-eval
python3 -u diagnostics/q1_evaluate.py \
  --run-dirs diagnostics/q1_baseline diagnostics/q1_alignment_only \
  --output diagnostics/q1_evaluation.json
