#!/usr/bin/env python3
"""Stratify Stage-2 results by teacher safety and audit cost/safety alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from bspline_utils import DifferentiableBSpline
from dit.Models import PhysicalScaledEdgeResidualRepresentation
from experiment_a_representation import evaluate_control_points, summarize_array
from experiment_stage2_coupling import mode_label
from grad_optimizer import _sample_cost_map_on_dense_trajectory
from map_config import MAP_CONFIG, MAP_HALF_EXTENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="diagnostics/stage2_scale/N060_v6/optimizer_pairs.npz",
    )
    parser.add_argument(
        "--outputs",
        default="diagnostics/stage2_scale/N060_v6/model_outputs.npz",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["B_cost_only", "D_induced_coupling"],
    )
    parser.add_argument("--safe-length-ratio", type=float, default=1.17)
    parser.add_argument("--mode-threshold-m", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/stage2_teacher_student_gap",
    )
    return parser.parse_args()


def decode(
    residual: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
) -> torch.Tensor:
    representation = PhysicalScaledEdgeResidualRepresentation()
    with torch.no_grad():
        return (
            representation.decode(
                residual,
                starts[:, :2] / MAP_HALF_EXTENT,
                goals[:, :2] / MAP_HALF_EXTENT,
            )
            * MAP_HALF_EXTENT
        )


def load_heldout(
    pairs_path: Path,
    outputs_path: Path,
    variants: Iterable[str],
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    with np.load(pairs_path) as pairs:
        heldout = pairs["split"].astype(str) == "heldout"
        data: Dict[str, object] = {
            "map": torch.from_numpy(pairs["maps"][heldout].copy()),
            "cost_map": torch.from_numpy(
                pairs["privileged_cost_maps"][heldout].copy()
            ),
            "start": torch.from_numpy(pairs["starts"][heldout].copy()),
            "goal": torch.from_numpy(pairs["goals"][heldout].copy()),
            "r0": torch.from_numpy(
                pairs["stage1_residual"][heldout].copy()
            ),
            "r_star": torch.from_numpy(
                pairs["optimized_residual"][heldout].copy()
            ),
            "environment": pairs["environment"][heldout].astype(str),
            "path_num": pairs["path_num"][heldout].copy(),
            "condition_id": pairs["condition_id"][heldout].copy(),
            "source_id": pairs["source_id"][heldout].copy(),
        }
    data["cp0"] = decode(
        data["r0"], data["start"], data["goal"]
    )
    data["cp_star"] = decode(
        data["r_star"], data["start"], data["goal"]
    )
    predictions: Dict[str, torch.Tensor] = {}
    with np.load(outputs_path) as outputs:
        for variant in variants:
            predictions[f"{variant}_r"] = torch.from_numpy(
                outputs[f"{variant}_heldout_r"].copy()
            )
            predictions[f"{variant}_cp"] = torch.from_numpy(
                outputs[f"{variant}_heldout_cp"].copy()
            )
    return data, predictions


def evaluate(
    control_points: torch.Tensor,
    data: Dict[str, object],
    cp0_lengths: torch.Tensor,
    batch_size: int,
    safe_length_ratio: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    bspline = DifferentiableBSpline(26, 100, 3).to(device)
    chunks: Dict[str, List[torch.Tensor]] = {}
    esdf_rows: List[torch.Tensor] = []
    dense_rows: List[torch.Tensor] = []
    with torch.no_grad():
        for begin in range(0, control_points.shape[0], batch_size):
            end = min(begin + batch_size, control_points.shape[0])
            cp = control_points[begin:end].to(device)
            starts = data["start"][begin:end].to(device)
            goals = data["goal"][begin:end].to(device)
            maps = data["cost_map"][begin:end].to(device)
            result = evaluate_control_points(
                cp,
                starts,
                goals,
                maps,
                MAP_CONFIG.cost_map_info(),
                bspline,
                device,
            )
            dense = result["trajectory"]
            _, _, _, esdf = _sample_cost_map_on_dense_trajectory(
                dense,
                maps,
                MAP_CONFIG.cost_map_info(),
                device,
            )
            for key, value in result.items():
                if key != "trajectory":
                    chunks.setdefault(key, []).append(value.cpu())
            esdf_rows.append(esdf.cpu())
            dense_rows.append(dense.cpu())
    result_np = {
        key: torch.cat(values).numpy() for key, values in chunks.items()
    }
    esdf = torch.cat(esdf_rows)
    dense = torch.cat(dense_rows)
    path_length = torch.from_numpy(result_np["path_length"])
    result_np.update(
        {
            "esdf": esdf.numpy(),
            "dense": dense.numpy(),
            "dangerous_count": (esdf < 0.0).sum(dim=1).numpy(),
            "max_violation": (-esdf).clamp_min(0.0).amax(dim=1).numpy(),
            "length_ratio": (path_length / cp0_lengths).numpy(),
        }
    )
    result_np["strict_safe"] = (
        (result_np["dangerous_count"] == 0)
        & (result_np["length_ratio"] <= safe_length_ratio)
    )
    return result_np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float(
        np.mean(comparison > 0.0) + 0.5 * np.mean(comparison == 0.0)
    )


def finite_summary(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return summarize_array(values[np.isfinite(values)])


def recovery(
    initial: np.ndarray,
    output: np.ndarray,
    teacher: np.ndarray,
) -> np.ndarray:
    denominator = initial - teacher
    result = np.full_like(initial, np.nan, dtype=np.float64)
    valid = denominator > 1e-6
    result[valid] = (initial[valid] - output[valid]) / denominator[valid]
    return result


def condition_groups(condition_ids: np.ndarray) -> Dict[int, np.ndarray]:
    return {
        int(condition): np.flatnonzero(condition_ids == condition)
        for condition in np.unique(condition_ids)
    }


def safe_at_k(
    strict_safe: np.ndarray,
    groups: Dict[int, np.ndarray],
    selected_conditions: Iterable[int],
) -> float:
    selected = list(selected_conditions)
    if not selected:
        return float("nan")
    return float(np.mean([
        bool(np.any(strict_safe[groups[condition]]))
        for condition in selected
    ]))


def correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, predictions = load_heldout(
        Path(args.pairs), Path(args.outputs), args.variants
    )
    print(f"Loaded {len(data['condition_id'])} held-out candidates")

    bspline = DifferentiableBSpline(26, 100, 3)
    with torch.no_grad():
        dense0 = bspline(data["cp0"])
        cp0_lengths = torch.linalg.vector_norm(
            dense0[:, 1:] - dense0[:, :-1], dim=-1
        ).sum(dim=1)

    metrics: Dict[str, Dict[str, np.ndarray]] = {
        "stage1": evaluate(
            data["cp0"],
            data,
            cp0_lengths,
            args.batch_size,
            args.safe_length_ratio,
            device,
        ),
        "teacher": evaluate(
            data["cp_star"],
            data,
            cp0_lengths,
            args.batch_size,
            args.safe_length_ratio,
            device,
        ),
    }
    for variant in args.variants:
        metrics[variant] = evaluate(
            predictions[f"{variant}_cp"],
            data,
            cp0_lengths,
            args.batch_size,
            args.safe_length_ratio,
            device,
        )

    groups = condition_groups(data["condition_id"])
    teacher_safe_conditions = [
        condition for condition, indices in groups.items()
        if np.any(metrics["teacher"]["strict_safe"][indices])
    ]
    teacher_unsafe_conditions = [
        condition for condition in groups
        if condition not in teacher_safe_conditions
    ]
    condition_subset = {
        "teacher_safe": teacher_safe_conditions,
        "teacher_unsafe": teacher_unsafe_conditions,
    }
    summary: Dict[str, object] = {
        "definition": {
            "strict_safe": (
                "all 100 yaw-ESDF samples >= 0 and "
                f"path length ratio <= {args.safe_length_ratio}"
            ),
            "teacher_safe_condition": (
                "at least one of K teacher candidates is strict-safe"
            ),
        },
        "counts": {
            "conditions": len(groups),
            "candidates": len(data["condition_id"]),
            "teacher_safe_conditions": len(teacher_safe_conditions),
            "teacher_unsafe_conditions": len(teacher_unsafe_conditions),
            "teacher_safe_candidates": int(
                metrics["teacher"]["strict_safe"].sum()
            ),
        },
        "stratified": {},
        "cost_alignment": {},
    }

    rows: List[Dict[str, object]] = []
    for variant in args.variants:
        variant_summary: Dict[str, object] = {}
        student = metrics[variant]
        candidate_recovery = recovery(
            metrics["stage1"]["cost"],
            student["cost"],
            metrics["teacher"]["cost"],
        )
        teacher_error = torch.linalg.vector_norm(
            (
                predictions[f"{variant}_r"] - data["r_star"]
            ).flatten(start_dim=1),
            dim=1,
        )
        teacher_delta = torch.linalg.vector_norm(
            (data["r0"] - data["r_star"]).flatten(start_dim=1),
            dim=1,
        )
        relative_teacher_error = (
            teacher_error / teacher_delta.clamp_min(1e-8)
        ).numpy()
        relative_teacher_error[teacher_delta.numpy() <= 1e-5] = np.nan
        max_cp_deviation = torch.linalg.vector_norm(
            predictions[f"{variant}_cp"] - data["cp_star"], dim=-1
        ).amax(dim=1).numpy()
        r0_mode = mode_label(
            data["cp0"], args.mode_threshold_m
        ).numpy()
        teacher_mode = mode_label(
            data["cp_star"], args.mode_threshold_m
        ).numpy()
        student_mode = mode_label(
            predictions[f"{variant}_cp"], args.mode_threshold_m
        ).numpy()
        for subset_name, selected_conditions in condition_subset.items():
            indices = np.concatenate([
                groups[condition] for condition in selected_conditions
            ])
            variant_summary[subset_name] = {
                "condition_count": len(selected_conditions),
                "candidate_count": len(indices),
                "student_safe_at_k": safe_at_k(
                    student["strict_safe"], groups, selected_conditions
                ),
                "student_safe_candidate_rate": float(
                    np.mean(student["strict_safe"][indices])
                ),
                "recovery": finite_summary(candidate_recovery[indices]),
                "relative_teacher_error": finite_summary(
                    relative_teacher_error[indices]
                ),
                "max_control_point_deviation_m": finite_summary(
                    max_cp_deviation[indices]
                ),
                "mode_keep_vs_stage1": float(
                    np.mean(student_mode[indices] == r0_mode[indices])
                ),
                "mode_match_teacher": float(
                    np.mean(student_mode[indices] == teacher_mode[indices])
                ),
            }
        summary["stratified"][variant] = variant_summary

        for index in range(len(data["condition_id"])):
            newly_unsafe = np.flatnonzero(
                (metrics["teacher"]["esdf"][index] >= 0.0)
                & (student["esdf"][index] < 0.0)
            )
            rows.append(
                {
                    "variant": variant,
                    "environment": data["environment"][index],
                    "path_num": int(data["path_num"][index]),
                    "condition_id": int(data["condition_id"][index]),
                    "source_id": int(data["source_id"][index]),
                    "teacher_condition_safe": (
                        int(data["condition_id"][index])
                        in teacher_safe_conditions
                    ),
                    "teacher_candidate_safe": bool(
                        metrics["teacher"]["strict_safe"][index]
                    ),
                    "student_candidate_safe": bool(
                        student["strict_safe"][index]
                    ),
                    "recovery": float(candidate_recovery[index]),
                    "relative_teacher_error": float(
                        relative_teacher_error[index]
                    ),
                    "max_control_point_deviation_m": float(
                        max_cp_deviation[index]
                    ),
                    "mode_keep_vs_stage1": bool(
                        student_mode[index] == r0_mode[index]
                    ),
                    "mode_match_teacher": bool(
                        student_mode[index] == teacher_mode[index]
                    ),
                    "teacher_cost": float(
                        metrics["teacher"]["cost"][index]
                    ),
                    "student_cost": float(student["cost"][index]),
                    "teacher_min_esdf": float(
                        metrics["teacher"]["min_esdf"][index]
                    ),
                    "student_min_esdf": float(
                        student["min_esdf"][index]
                    ),
                    "newly_unsafe_point_count": int(len(newly_unsafe)),
                    "newly_unsafe_point_indices": " ".join(
                        map(str, newly_unsafe.tolist())
                    ),
                    "student_max_violation_m": float(
                        student["max_violation"][index]
                    ),
                }
            )

    for name, values in metrics.items():
        labels = values["strict_safe"].astype(bool)
        summary["cost_alignment"][name] = {
            "strict_safe_candidates": int(labels.sum()),
            "strict_safe_rate": float(labels.mean()),
            "cost_auroc_for_strict_safe": binary_auc(
                labels, -values["cost"]
            ),
            "safe_cost_distribution": (
                finite_summary(values["cost"][labels])
                if labels.any() else None
            ),
            "unsafe_cost_distribution": (
                finite_summary(values["cost"][~labels])
                if (~labels).any() else None
            ),
            "unsafe_max_violation_m": (
                finite_summary(values["max_violation"][~labels])
                if (~labels).any() else None
            ),
            "unsafe_dangerous_point_count": (
                finite_summary(values["dangerous_count"][~labels])
                if (~labels).any() else None
            ),
        }
    cost_drop = metrics["stage1"]["cost"] - metrics["teacher"]["cost"]
    dangerous_drop = (
        metrics["stage1"]["dangerous_count"]
        - metrics["teacher"]["dangerous_count"]
    )
    summary["cost_alignment"]["teacher_improvement_correlation"] = {
        "cost_drop_vs_dangerous_point_drop": correlation(
            cost_drop, dangerous_drop
        ),
        "cost_drop": finite_summary(cost_drop),
        "dangerous_point_drop": finite_summary(dangerous_drop),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    write_csv(output_dir / "per_candidate.csv", rows)

    teacher_alignment = summary["cost_alignment"]["teacher"]
    report = [
        "# Teacher safety / student imitation 分层分析",
        "",
        f"- Held-out：{len(groups)} 个条件，{len(data['condition_id'])} 个候选。",
        f"- Teacher-safe / unsafe 条件："
        f"{len(teacher_safe_conditions)}/{len(teacher_unsafe_conditions)}。",
        f"- 严格安全 teacher 候选："
        f"{summary['counts']['teacher_safe_candidates']}。",
        "- Strict-safe：100 个 yaw-ESDF 采样点全部非负，且路径长度比不超过 "
        f"{args.safe_length_ratio:.2f}。",
        "",
        "## Teacher-safe 条件上的学生复现",
        "",
        "| Student | safe@4 | Recovery 中位数 | Teacher error 中位数 | "
        "最大控制点误差中位数 (m) | 模式保持 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in args.variants:
        row = summary["stratified"][variant]["teacher_safe"]
        report.append(
            f"| {variant} | {row['student_safe_at_k']:.1%} | "
            f"{row['recovery']['median']:.3f} | "
            f"{row['relative_teacher_error']['median']:.3f} | "
            f"{row['max_control_point_deviation_m']['median']:.3f} | "
            f"{row['mode_keep_vs_stage1']:.1%} |"
        )
    report.extend([
        "",
        "## Cost 与 strict-safe",
        "",
        "| Trajectory | 安全候选数 | Cost AUROC | 安全 cost 中位数 | "
        "不安全 cost 中位数 | 不安全最大违规中位数 (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, values in summary["cost_alignment"].items():
        if name == "teacher_improvement_correlation":
            continue
        safe_distribution = values["safe_cost_distribution"]
        unsafe_distribution = values["unsafe_cost_distribution"]
        report.append(
            f"| {name} | {values['strict_safe_candidates']} | "
            f"{values['cost_auroc_for_strict_safe']:.3f} | "
            f"{safe_distribution['median'] if safe_distribution else float('nan'):.3f} | "
            f"{unsafe_distribution['median'] if unsafe_distribution else float('nan'):.3f} | "
            f"{values['unsafe_max_violation_m']['median']:.3f} |"
        )
    correlation_row = summary["cost_alignment"][
        "teacher_improvement_correlation"
    ]["cost_drop_vs_dangerous_point_drop"]
    report.extend([
        "",
        "Teacher cost 下降量与危险点减少量："
        f"Pearson `{correlation_row['pearson']:.3f}`，"
        f"Spearman `{correlation_row['spearman']:.3f}`。",
        "",
        "逐候选重新变得不安全的轨迹点索引、最大违规和控制点偏差见 "
        "`per_candidate.csv`。",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Completed: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
