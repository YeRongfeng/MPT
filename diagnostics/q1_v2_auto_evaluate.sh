#!/usr/bin/env bash
set -eo pipefail
cd /home/yrf/MPT
while true; do
  if [ -f diagnostics/q1_alignment_only_v2/summary.json ]; then
    if python3 - <<'PY'
import json
if json.load(open('diagnostics/q1_alignment_only_v2/summary.json')).get('status') == 'completed':
    raise SystemExit(0)
raise SystemExit(1)
PY
    then break; fi
  fi
  sleep 60
done
source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-q1-v2-eval
python3 -u diagnostics/q1_evaluate.py \
  --run-dirs diagnostics/q1_alignment_only_v2 \
  --output diagnostics/q1_v2_evaluation.json
