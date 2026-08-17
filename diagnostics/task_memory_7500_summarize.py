#!/usr/bin/env python3
"""Merge 2500-5000 and 5000-7500 aligned evaluations into one curve summary."""

from __future__ import annotations

import json
from pathlib import Path

MID = Path("diagnostics/task_memory_mid/evaluation.json")
NEW = Path("diagnostics/task_memory_7500/evaluation.json")
OUT = Path("diagnostics/task_memory_7500/summary.json")


def extract(evaluation, arch_key_substr):
    rows = []
    for run_dir, arm in evaluation.items():
        if arch_key_substr not in run_dir:
            continue
        for ckpt in arm["checkpoints"]:
            path = ckpt["checkpoint"]
            if "stage1_update_" not in path and not path.endswith("stage1_last.pth"):
                continue
            if "stage1_best.pth" in path:
                continue
            update = ckpt.get("update")
            if update is None:
                continue
            ep1 = ckpt["endpoint_metrics"]["1"]
            ep8 = ckpt["endpoint_metrics"]["8"]
            rows.append(
                {
                    "arch": arm["architecture"],
                    "update": int(update),
                    "role": Path(path).name,
                    "val": ckpt["validation_meanflow"]["loss"],
                    "safe1": ep1["safe_at_k"],
                    "safe8": ep8["safe_at_k"],
                    "curv": ep8["curvature_ok_rate"],
                    "forb": ep8["forbidden_ok_rate"],
                    "stab": ep8["stability_ok_rate"],
                }
            )
    by_update = {}
    for row in rows:
        prev = by_update.get(row["update"])
        if prev is None or row["role"].startswith("stage1_update_"):
            by_update[row["update"]] = row
    return [by_update[k] for k in sorted(by_update)]


def interval_stats(baseline, task_memory, start, end):
    b = {r["update"]: r for r in baseline if start <= r["update"] <= end}
    t = {r["update"]: r for r in task_memory if start <= r["update"] <= end}
    common = sorted(set(b) & set(t))
    wins = {"s1": 0, "s8": 0, "c": 0, "n": len(common)}
    diffs = {"s1": [], "s8": [], "c": [], "val": []}
    points = []
    for u in common:
        db = b[u]
        dt = t[u]
        ds1 = dt["safe1"] - db["safe1"]
        ds8 = dt["safe8"] - db["safe8"]
        dc = dt["curv"] - db["curv"]
        dval = dt["val"] - db["val"]
        if ds1 > 0:
            wins["s1"] += 1
        if ds8 > 0:
            wins["s8"] += 1
        if dc > 0:
            wins["c"] += 1
        diffs["s1"].append(ds1)
        diffs["s8"].append(ds8)
        diffs["c"].append(dc)
        diffs["val"].append(dval)
        points.append(
            {
                "update": u,
                "baseline": {
                    "val": db["val"],
                    "safe1": db["safe1"],
                    "safe8": db["safe8"],
                    "curv": db["curv"],
                },
                "task_memory": {
                    "val": dt["val"],
                    "safe1": dt["safe1"],
                    "safe8": dt["safe8"],
                    "curv": dt["curv"],
                },
                "delta_pp": {
                    "safe1": ds1 * 100,
                    "safe8": ds8 * 100,
                    "curv": dc * 100,
                    "val": dval,
                },
            }
        )
    mean = {
        key: (sum(vals) / len(vals) if vals else None) for key, vals in diffs.items()
    }
    return {
        "start": start,
        "end": end,
        "matched": common,
        "wins_task_memory": wins,
        "mean_delta": {
            "safe1_pp": None if mean["s1"] is None else mean["s1"] * 100,
            "safe8_pp": None if mean["s8"] is None else mean["s8"] * 100,
            "curv_pp": None if mean["c"] is None else mean["c"] * 100,
            "val": mean["val"],
        },
        "points": points,
    }


def peak(rows, metric):
    if not rows:
        return None
    best = max(rows, key=lambda r: r[metric])
    return {"update": best["update"], metric: best[metric], "val": best["val"]}


def decide(stats_5000_7500, tm_peak_s1, bl_peak_s1, last_point):
    wins = stats_5000_7500["wins_task_memory"]
    n = wins["n"] or 1
    s8_freq = wins["s8"] / n
    last = last_point or {}
    last_s1_tm = last.get("task_memory", {}).get("safe1")
    last_s1_bl = last.get("baseline", {}).get("safe1")
    last_s8_tm = last.get("task_memory", {}).get("safe8")
    last_s8_bl = last.get("baseline", {}).get("safe8")
    tm_peak = tm_peak_s1["safe1"] if tm_peak_s1 else 0.0
    frequent = s8_freq >= 0.6 and wins["c"] / n >= 0.5
    last_ahead = (
        last_s1_tm is not None
        and last_s1_bl is not None
        and last_s1_tm >= last_s1_bl - 0.005
        and last_s8_tm is not None
        and last_s8_bl is not None
        and last_s8_tm >= last_s8_bl - 0.01
    )
    peak_moved = tm_peak >= 0.14
    if frequent and (last_ahead or peak_moved):
        return (
            "promote_task_memory_as_main_forward_candidate",
            "5000-7500 still frequently favors task_memory; Safe@1 peak or last checkpoint did not collapse to a pure speed-only story.",
        )
    if not frequent and last_s1_bl is not None and last_s1_tm is not None and last_s1_bl > last_s1_tm:
        return (
            "task_memory_mostly_faster_not_clearly_higher_capacity",
            "After 5000, baseline mostly catches or overtakes; task_memory looks more like earlier/better optimization than a strictly higher ceiling.",
        )
    return (
        "keep_but_not_yet_new_mainline",
        "The 5000-7500 curve is mixed: repeated mid-interval wins are not the same as a stable final Safe@1 lead.",
    )


def main():
    mid = json.loads(MID.read_text()) if MID.exists() else {}
    new = json.loads(NEW.read_text())
    baseline = extract(mid, "/baseline") + extract(new, "/baseline")
    task_memory = extract(mid, "/task_memory") + extract(new, "/task_memory")
    # Prefer the later evaluation for duplicated updates (5000).
    def dedup(rows):
        by_u = {}
        for row in rows:
            by_u[row["update"]] = row
        return [by_u[k] for k in sorted(by_u)]

    baseline = dedup(baseline)
    task_memory = dedup(task_memory)
    stats_2500_5000 = interval_stats(baseline, task_memory, 2500, 5000)
    stats_5000_7500 = interval_stats(baseline, task_memory, 5000, 7500)
    stats_2500_7500 = interval_stats(baseline, task_memory, 2500, 7500)
    tm_in_new = [r for r in task_memory if 5000 <= r["update"] <= 7500]
    bl_in_new = [r for r in baseline if 5000 <= r["update"] <= 7500]
    last = stats_5000_7500["points"][-1] if stats_5000_7500["points"] else None
    tm_peak = peak(tm_in_new, "safe1")
    bl_peak = peak(bl_in_new, "safe1")
    decision, reason = decide(stats_5000_7500, tm_peak, bl_peak, last)
    summary = {
        "status": "completed_7500_not_final",
        "protocol": "diagnostics/task_memory_7500_protocol.json",
        "curve": baseline + task_memory,
        "interval_2500_5000": {
            "wins_task_memory": stats_2500_5000["wins_task_memory"],
            "mean_delta": stats_2500_5000["mean_delta"],
        },
        "interval_5000_7500": stats_5000_7500,
        "interval_2500_7500": {
            "wins_task_memory": stats_2500_7500["wins_task_memory"],
            "mean_delta": stats_2500_7500["mean_delta"],
        },
        "peaks_5000_7500": {
            "task_memory_safe1": tm_peak,
            "baseline_safe1": bl_peak,
            "task_memory_safe8": peak(tm_in_new, "safe8"),
            "baseline_safe8": peak(bl_in_new, "safe8"),
        },
        "last_7500": last,
        "decision": decision,
        "decision_reason": reason,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
