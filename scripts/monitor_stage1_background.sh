#!/usr/bin/env bash
set -u

repo_root="${1:-/home/sdu/MPT}"
log_path="${2:-/tmp/mpt-stage1-monitor.log}"
interval="${MPT_MONITOR_INTERVAL:-60}"
event_dir="$repo_root/data/stage1_final_random_uav/tensorboard/stage1"

mkdir -p "$(dirname "$log_path")"
printf 'monitor_started=%s repo=%s interval=%ss\n' "$(date --iso-8601=seconds)" "$repo_root" "$interval" >> "$log_path"

while :; do
    timestamp="$(date --iso-8601=seconds)"
    train_pids="$(pgrep -f 'scripts/run_stage1_with_local_runtime.py.*--workflow stage1' | tr '\n' ',' || true)"
    tensorboard_pids="$(pgrep -f 'tensorboard.*stage1_final_random_uav/tensorboard' | tr '\n' ',' || true)"
    latest_event="$(find "$event_dir" -maxdepth 1 -type f -name 'events.out.tfevents.*' -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -1)"
    printf '%s train_pids=%s tensorboard_pids=%s latest_event=%s\n' \
        "$timestamp" "${train_pids:-none}" "${tensorboard_pids:-none}" "${latest_event:-none}" >> "$log_path"
    sleep "$interval"
done
