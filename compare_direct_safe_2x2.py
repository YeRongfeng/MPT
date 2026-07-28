#!/usr/bin/env python3
"""Aggregate complete A/B/C/D seed matrices without hiding missing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np


VARIANTS = ("A", "B", "C", "D")
CONTRASTS = {
    "canonicalization_on_raw_targets_B_minus_A": ("B", "A"),
    "canonicalization_on_safe_targets_D_minus_C": ("D", "C"),
    "safe_targets_in_global_frame_C_minus_A": ("C", "A"),
    "safe_targets_in_canonical_frame_D_minus_B": ("D", "B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("diagnostics/direct_safe_distribution"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/comparison_report.md"
        ),
    )
    return parser.parse_args()


def discover(root: Path, split: str) -> Dict[int, Dict[str, Mapping[str, object]]]:
    result: Dict[int, Dict[str, Mapping[str, object]]] = {}
    for variant in VARIANTS:
        for seed_folder in sorted((root / f"experiment_{variant}").glob("seed_*")):
            try:
                seed = int(seed_folder.name.split("_", 1)[1])
            except ValueError:
                continue
            summary_path = seed_folder / f"evaluation_{split}" / "summary.json"
            if summary_path.is_file():
                result.setdefault(seed, {})[variant] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
    return result


def validate_controls(root: Path, seed: int) -> Dict[str, object]:
    configs = {
        variant: json.loads(
            (
                root
                / f"experiment_{variant}"
                / f"seed_{seed}"
                / "config.json"
            ).read_text(encoding="utf-8")
        )
        for variant in VARIANTS
    }
    reference = configs["A"]
    errors: List[str] = []
    fixed_argument_keys = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "canvas_half_extent",
        "legacy_global_input",
    )
    for variant in VARIANTS[1:]:
        candidate = configs[variant]
        for key in ("parameter_count", "model_args", "loss_weights"):
            if candidate[key] != reference[key]:
                errors.append(f"{variant}: {key} differs from A")
        for key in fixed_argument_keys:
            if candidate["arguments"][key] != reference["arguments"][key]:
                errors.append(f"{variant}: argument {key} differs from A")
    if configs["C"]["teacher_dataset_sha256"] != configs["D"][
        "teacher_dataset_sha256"
    ]:
        errors.append("C/D teacher dataset hashes differ")
    if configs["A"]["teacher_dataset_sha256"] is not None or configs["B"][
        "teacher_dataset_sha256"
    ] is not None:
        errors.append("A/B unexpectedly reference a teacher dataset")
    return {"passed": not errors, "errors": errors}


def scalar(summary: Mapping[str, object], metric: str, k: str = "32") -> float:
    if metric in summary["prefix_metrics"][k]:
        return float(summary["prefix_metrics"][k][metric])
    return float(summary["single_candidate"][metric])


def main() -> None:
    args = parse_args()
    runs = discover(args.root, args.split)
    complete = {
        seed: variants
        for seed, variants in runs.items()
        if set(variants) == set(VARIANTS)
    }
    control_checks = {
        seed: validate_controls(args.root, seed) for seed in complete
    }
    invalid_control_seeds = [
        seed for seed, result in control_checks.items() if not result["passed"]
    ]
    if invalid_control_seeds:
        complete = {
            seed: variants
            for seed, variants in complete.items()
            if seed not in invalid_control_seeds
        }
    missing = {
        seed: sorted(set(VARIANTS) - set(variants))
        for seed, variants in runs.items()
        if set(variants) != set(VARIANTS)
    }
    metrics = (
        "safe_at_k",
        "valid_rate",
        "best_cost_at_k_median",
        "safe_mode_coverage_mean",
        "deduplicated_valid_count_mean",
    )
    aggregate: Dict[str, object] = {}
    if complete:
        for variant in VARIANTS:
            aggregate[variant] = {
                metric: {
                    "mean": float(
                        np.mean(
                            [
                                scalar(variants[variant], metric)
                                for variants in complete.values()
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            [
                                scalar(variants[variant], metric)
                                for variants in complete.values()
                            ],
                            ddof=1,
                        )
                    )
                    if len(complete) > 1
                    else 0.0,
                }
                for metric in metrics
            }
        aggregate["contrasts"] = {}
        for name, (positive, negative) in CONTRASTS.items():
            aggregate["contrasts"][name] = {
                metric: {
                    "mean_delta": float(
                        np.mean(
                            [
                                scalar(variants[positive], metric)
                                - scalar(variants[negative], metric)
                                for variants in complete.values()
                            ]
                        )
                    ),
                    "per_seed_delta": {
                        str(seed): (
                            scalar(variants[positive], metric)
                            - scalar(variants[negative], metric)
                        )
                        for seed, variants in complete.items()
                    },
                }
                for metric in metrics
            }
    decision = "insufficient_results"
    if complete:
        d = aggregate["D"]
        safe32 = d["safe_at_k"]["mean"]
        valid = d["valid_rate"]["mean"]
        mode = d["safe_mode_coverage_mean"]["mean"]
        if safe32 < 0.8:
            decision = "do_not_train_scorer_generator_coverage_is_low"
        elif valid < 0.5:
            decision = "candidate_ranking_may_be_worth_a_separate_followup"
        elif mode < 1.5:
            decision = "inspect_mode_collapse_before_candidate_ranking"
        else:
            decision = "generator_is_broadly_safe_scorer_not_yet_necessary"
    machine = {
        "split": args.split,
        "complete_seeds": sorted(complete),
        "incomplete_seeds": missing,
        "control_checks": control_checks,
        "invalid_control_seeds": invalid_control_seeds,
        "aggregate": aggregate,
        "scorer_decision_rule_result": decision,
        "warning": (
            "No research conclusion is emitted unless a seed has all four "
            "controlled variants."
        ),
    }
    json_path = args.output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# 直接安全分布 2×2 对照",
        "",
        f"- Split: `{args.split}`",
        f"- 完整 A/B/C/D seeds: {sorted(complete)}",
        f"- 不完整 seeds: {missing}",
        f"- 控制变量不合格 seeds: {invalid_control_seeds}",
        "",
    ]
    if not complete:
        report.extend(
            [
                "尚无完整的同 seed A/B/C/D 结果，因此不回答六个研究问题，"
                "也不对 canonicalization 或直接安全训练作正负结论。",
            ]
        )
    else:
        report.extend(
            [
                "| Variant | Safe@32 | Valid Rate | Best-Cost@32 | Modes | Dedup valid |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            value = aggregate[variant]
            report.append(
                f"| {variant} | {value['safe_at_k']['mean']:.1%} | "
                f"{value['valid_rate']['mean']:.1%} | "
                f"{value['best_cost_at_k_median']['mean']:.3f} | "
                f"{value['safe_mode_coverage_mean']['mean']:.2f} | "
                f"{value['deduplicated_valid_count_mean']['mean']:.2f} |"
            )
        report.extend(
            [
                "",
                f"候选评分器判定：`{decision}`。",
                "",
                "因果对比按 B−A、D−C、C−A、D−B 计算，完整数值和逐 seed "
                "delta 见同名 JSON。Best-Cost 使用 privileged Oracle，仅是"
                "候选集上限。",
            ]
        )
    args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; complete seeds={sorted(complete)}")


if __name__ == "__main__":
    main()
