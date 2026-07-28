"""Run the frozen Stage-1 initial-noise diagnostic across many conditions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/sim_dataset/val")
    parser.add_argument("--model-dir", default="data/sim")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--env-count", type=int, default=8)
    parser.add_argument("--paths-per-env", type=int, default=4)
    parser.add_argument("--num-restarts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help="Explicit environments. By default, sample original non-_optimized envs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep root. Defaults to diagnostics/stage1_noise_sweep_<timestamp>.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def discover_scenes(args: argparse.Namespace) -> list[tuple[str, int]]:
    dataset = Path(args.dataset)
    available_envs = sorted(
        path.name
        for path in dataset.iterdir()
        if path.is_dir() and path.name.startswith("env") and not path.name.endswith("_optimized")
    )
    if args.envs:
        missing = sorted(set(args.envs) - set(available_envs))
        if missing:
            raise FileNotFoundError(f"Unknown original environments: {missing}")
        selected_envs = list(args.envs)
    else:
        if args.env_count > len(available_envs):
            raise ValueError(
                f"Requested {args.env_count} environments, only {len(available_envs)} available"
            )
        rng = np.random.default_rng(args.seed)
        selected_envs = sorted(
            rng.choice(available_envs, size=args.env_count, replace=False).tolist()
        )

    scenes: list[tuple[str, int]] = []
    for env_offset, env_name in enumerate(selected_envs):
        path_ids = sorted(
            int(path.stem.split("_")[1])
            for path in (dataset / env_name).glob("path_*.p")
        )
        if args.paths_per_env > len(path_ids):
            raise ValueError(
                f"{env_name}: requested {args.paths_per_env} paths, only {len(path_ids)} available"
            )
        env_rng = np.random.default_rng(args.seed + 1009 * (env_offset + 1))
        selected_paths = sorted(
            env_rng.choice(path_ids, size=args.paths_per_env, replace=False).tolist()
        )
        scenes.extend((env_name, int(path_id)) for path_id in selected_paths)
    return scenes


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def aggregate(records: list[dict], failures: list[dict], config: dict) -> dict:
    restart_rows = [
        {**row, "environment": record["environment"], "path_index": record["path_index"]}
        for record in records
        for row in record["per_restart"]
    ]
    environment_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        environment_records[record["environment"]].append(record)

    def aggregate_group(group: list[dict]) -> dict:
        rows = [row for record in group for row in record["per_restart"]]
        initial_costs = [row["initial_cost"] for row in rows]
        optimized_costs = [row["best_cost"] for row in rows]
        gt_by_row = [
            record["ground_truth_reference_cost"]
            for record in group
            for _ in record["per_restart"]
        ]
        condition_initial_means = [record["initial_cost"]["mean"] for record in group]
        condition_optimized_means = [record["optimized_best_cost"]["mean"] for record in group]
        return {
            "condition_count": len(group),
            "restart_count": len(rows),
            "initial_cost": describe(initial_costs),
            "optimized_cost": describe(optimized_costs),
            "condition_initial_mean_cost": describe(condition_initial_means),
            "condition_optimized_mean_cost": describe(condition_optimized_means),
            "mean_relative_reduction": float(
                np.mean([row["relative_reduction"] for row in rows])
            ),
            "initial_below_gt_rate": float(
                np.mean(np.asarray(initial_costs) <= np.asarray(gt_by_row))
            ),
            "optimized_below_gt_rate": float(
                np.mean(np.asarray(optimized_costs) <= np.asarray(gt_by_row))
            ),
            "conditions_all_restarts_below_gt_rate": float(
                np.mean(
                    [
                        record["optimized_fraction_below_gt_reference"] == 1.0
                        for record in group
                    ]
                )
            ),
            "conditions_optimized_mean_below_gt_rate": float(
                np.mean(
                    [
                        record["optimized_best_cost"]["mean"]
                        <= record["ground_truth_reference_cost"]
                        for record in group
                    ]
                )
            ),
            "initial_z_energy_per_dim": describe(
                [row["initial_z_energy_per_dim"] for row in rows]
            ),
            "optimized_z_energy_per_dim": describe(
                [row["optimized_z_energy_per_dim"] for row in rows]
            ),
            "z_displacement_norm": describe(
                [row["z_displacement_norm"] for row in rows]
            ),
        }

    return {
        "config": config,
        "completed_conditions": len(records),
        "failed_conditions": failures,
        "overall": aggregate_group(records) if records else {},
        "by_environment": {
            env_name: aggregate_group(group)
            for env_name, group in sorted(environment_records.items())
        },
        "conditions": records,
        "restart_rows": restart_rows,
    }


def write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    scenes = discover_scenes(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(
        args.output_dir or f"diagnostics/stage1_noise_sweep_{timestamp}"
    )
    condition_root = sweep_root / "conditions"
    condition_root.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(sweep_root)
    config["scenes"] = [
        {"environment": env_name, "path_index": path_index}
        for env_name, path_index in scenes
    ]
    records: list[dict] = []
    failures: list[dict] = []
    print(f"Sweep root: {sweep_root}")
    print(
        f"Running {len(scenes)} conditions × {args.num_restarts} restarts "
        f"× {args.steps} optimization steps"
    )

    for condition_index, (env_name, path_index) in enumerate(scenes, start=1):
        condition_seed = args.seed + condition_index - 1
        print(
            f"\n[{condition_index}/{len(scenes)}] {env_name}/path_{path_index} "
            f"(seed={condition_seed})",
            flush=True,
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("diagnose_stage1_latent.py")),
            "--model-dir",
            args.model_dir,
            "--dataset",
            args.dataset,
            "--env",
            env_name,
            "--path-index",
            str(path_index),
            "--num-restarts",
            str(args.num_restarts),
            "--steps",
            str(args.steps),
            "--lr",
            str(args.lr),
            "--seed",
            str(condition_seed),
            "--device",
            args.device,
            "--log-every",
            str(max(args.steps, 1)),
            "--output-dir",
            str(condition_root),
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        try:
            subprocess.run(command, check=True)
            result_path = (
                condition_root
                / f"{env_name}_path{path_index}_seed{condition_seed}"
                / "summary.json"
            )
            with result_path.open() as handle:
                records.append(json.load(handle))
        except Exception as error:
            failure = {
                "environment": env_name,
                "path_index": path_index,
                "seed": condition_seed,
                "error": repr(error),
            }
            failures.append(failure)
            if not args.continue_on_error:
                summary = aggregate(records, failures, config)
                write_summary(sweep_root / "sweep_summary.json", summary)
                raise

        summary = aggregate(records, failures, config)
        write_summary(sweep_root / "sweep_summary.json", summary)
        overall = summary.get("overall", {})
        if overall:
            print(
                "Cumulative: "
                f"random-hit={overall['initial_below_gt_rate']:.1%}, "
                f"optimized-hit={overall['optimized_below_gt_rate']:.1%}, "
                f"mean-drop={overall['mean_relative_reduction']:.1%}"
            )

    print(f"\nCompleted. Summary: {sweep_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
