"""Run reproduced baselines on Path MeanFlow tasks with the shared evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from map_config import discover_environments
from baselines.common import (
    evaluate_xy_path,
    infeasible_result,
    make_task,
    summarize_records,
    timed,
)


def _path_ids(env_dir: Path, limit: int) -> List[int]:
    names = sorted(env_dir.glob("path_*.p"))
    ids = [int(path.stem.split("_")[1]) for path in names]
    ids.sort()
    if limit > 0:
        return ids[: int(limit)]
    return ids


def _plan_one(method: str, task, args) -> Dict[str, Any]:
    if method in {"path_meanflow_stage1", "path_meanflow_stage2"}:
        from baselines.path_meanflow import plan_path_meanflow

        checkpoint = args.checkpoint
        if not checkpoint:
            checkpoint = (
                "data/path_meanflow/stage1_best.pth"
                if method.endswith("stage1")
                else "data/path_meanflow/stage2_best.pth"
            )
        return plan_path_meanflow(
            task,
            checkpoint=checkpoint,
            source_seed=args.mask_seed,
        )
    if method == "t_hybrid":
        from baselines.t_hybrid import plan_t_hybrid

        return plan_t_hybrid(task)
    if method == "uneven":
        if args.mask_mode != "full":
            raise ValueError("Original Uneven requires --mask-mode full")
        from baselines.uneven import plan_uneven

        return plan_uneven(task)
    raise ValueError(f"unknown method {method}")


def run_method(args) -> Dict[str, Any]:
    split_dir = Path(args.dataFolder) / args.split
    environments = discover_environments(split_dir)
    if args.environments > 0:
        environments = environments[: int(args.environments)]
    records: List[Dict[str, Any]] = []
    for env_name in environments:
        env_dir = split_dir / env_name
        for path_id in _path_ids(env_dir, args.paths_per_env):
            task = make_task(
                env_dir,
                path_id,
                mask_mode=args.mask_mode,
                mask_seed=args.mask_seed,
                p_mask=args.p_mask,
            )
            planned, elapsed = timed(lambda: _plan_one(args.method, task, args))
            record: Dict[str, Any] = {
                "method": args.method,
                "environment": env_name,
                "path_id": int(path_id),
                "mask_mode": args.mask_mode,
                "planning_time_s": elapsed,
                "found": planned["found"],
                "expansions": planned["expansions"],
                "input": planned["input"],
            }
            if not planned["found"] or planned["path"] is None:
                record.update(infeasible_result(planned["failure_reason"] or "no_path"))
            else:
                metrics = evaluate_xy_path(
                    planned["path"],
                    task,
                    curvature=planned.get("curvature"),
                )
                record.update(metrics)
            records.append(record)
            print(
                f"{args.method} {env_name} path_{path_id}: "
                f"found={record['found']} strict={record.get('strict_valid')} "
                f"time={elapsed:.3f}s"
            )
    summary = summarize_records(records)
    payload = {
        "method": args.method,
        "dataFolder": str(args.dataFolder),
        "split": args.split,
        "mask_mode": args.mask_mode,
        "p_mask": args.p_mask,
        "summary": summary,
        "records": records,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "uneven",
            "t_hybrid",
            "path_meanflow_stage1",
            "path_meanflow_stage2",
        ],
        required=True,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataFolder", default="data/dataset1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--environments", type=int, default=2)
    parser.add_argument("--paths-per-env", type=int, default=4)
    parser.add_argument(
        "--mask-mode",
        choices=["partial", "full", "stage2_independent"],
        default="partial",
    )
    parser.add_argument("--p_mask", type=float, default=1.0)
    parser.add_argument("--mask-seed", type=int, default=20260817)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = run_method(args)
    out = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent
        / "evaluation_results"
        / f"{args.method}_{args.split}_{args.mask_mode}_smoke.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=str))
    out.write_text(json.dumps(serializable, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
