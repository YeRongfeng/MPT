#!/usr/bin/env bash
# After both 2500-update screen arms finish, run generation metrics and CA-off.
set -eo pipefail
cd /home/yrf/MPT
while true; do
  if [ -f diagnostics/path_xy_screen/baseline/summary.json ] \
     && [ -f diagnostics/path_xy_screen/path_xy/summary.json ]; then
    if python3 - <<'PY'
import json
for path in (
    "diagnostics/path_xy_screen/baseline/summary.json",
    "diagnostics/path_xy_screen/path_xy/summary.json",
):
    if json.load(open(path)).get("status") != "completed":
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
export MPLCONFIGDIR=/tmp/matplotlib-path-xy-eval
python3 -u diagnostics/q1_evaluate.py \
  --run-dirs diagnostics/path_xy_screen/baseline diagnostics/path_xy_screen/path_xy \
  --output diagnostics/path_xy_screen/evaluation.json
python3 -u diagnostics/path_xy_diagnose.py \
  --run-dirs diagnostics/path_xy_screen/baseline diagnostics/path_xy_screen/path_xy \
  --output diagnostics/path_xy_screen/ca_off_diagnosis.json
