"""Archived training workflow for Stage-2 path-aligned KL reweighting."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG, SAFETY_COST_CONFIG
from posterior_pipeline import (
    _direct_cost_forward,
    _log_numeric_tree,
    _make_tensorboard_writer,
    _require_current_mask_semantics,
    _require_demo_target_semantics,
    _require_main_method_model,
    load_model,
    make_partial_dataset,
    normalize_poses,
    stage2_environment_selection,
)
from .stage2_reweighting import (
    STAGE2_REWEIGHTING_SEMANTICS,
    PathAlignedCandidateCritic,
    Stage2ReweightingConfig,
    bounded_kl_weights,
    critic_energy,
    empirical_kl_from_uniform,
    load_critic_checkpoint,
    save_critic_checkpoint,
    sha256_file,
)


STAGE2_REWEIGHTING_TRAINING_SEMANTICS = (
    "environment_disjoint_four_mask_fixed_candidate_critic_training_v1"
)
STAGE2_CANDIDATE_ARCHIVE_VERSION = 1
FORMAL_SELECTION_ENVIRONMENTS = (
    "env000062",
    "env000066",
    "env000076",
    "env000088",
)
PARTITION_MASK_BASE = {
    "fit": 30_000,
    "selection": 40_000,
    "validation": 50_000,
}
DEPLOYABLE_FIELDS = (
    "observed_map",
    "start_condition",
    "goal_condition",
    "candidate_path",
    "candidate_curvature",
)


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def expanded_environment_manifest(args) -> dict[str, object]:
    """Reproduce the admitted 64/4/16/16 Stage-2 environment partition."""
    base = stage2_environment_selection(args)
    unused = list(base["unused"])
    additional_count = int(args.stage2_additional_fit_environments)
    reserved_count = int(args.stage2_reserved_environments)
    if additional_count + reserved_count != len(unused):
        raise ValueError(
            "additional fit and reserved counts must exactly partition the "
            f"{len(unused)} base unused environments"
        )
    ordered = sorted(
        unused,
        key=lambda environment: (
            stable_int(
                args.stage2_split_seed,
                "p2-additional-fit-environment",
                environment,
            ),
            environment,
        ),
    )
    additional = sorted(ordered[:additional_count])
    reserved = sorted(ordered[additional_count:])
    expanded = sorted(list(base["train"]) + additional)
    selection = list(FORMAL_SELECTION_ENVIRONMENTS)
    if int(args.stage2_selection_environments) != len(selection):
        raise ValueError(
            "The admitted protocol fixes four internal selection environments"
        )
    if not set(selection) <= set(expanded):
        raise ValueError(
            "Stage-2 split settings no longer contain the admitted selection terrains"
        )
    fit = sorted(set(expanded) - set(selection))
    validation = list(base["validation"])
    partitions = [set(fit), set(selection), set(validation), set(reserved)]
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            if left & right:
                raise ValueError("Stage-2 physical environment partitions overlap")
    return {
        "semantic_version": "stage2_path_aligned_environment_split_v1",
        "base_split": base,
        "additional_fit_environments": additional,
        "fit_environments": fit,
        "selection_environments": selection,
        "validation_environments": validation,
        "reserved_environments": reserved,
    }


def _indices_by_environment(dataset) -> dict[str, list[int]]:
    result = {str(environment): [] for environment in dataset.env_list}
    for dataset_index, (environment_index, _) in enumerate(dataset.indexDict):
        environment = str(dataset.env_list[int(environment_index)])
        result[environment].append(int(dataset_index))
    return result


def select_contexts(
    dataset,
    environments: Sequence[str],
    *,
    count_per_environment: int,
    seed: int,
) -> list[dict[str, object]]:
    by_environment = _indices_by_environment(dataset)
    contexts = []
    for environment in environments:
        candidates = by_environment.get(str(environment), [])
        if len(candidates) < int(count_per_environment):
            raise ValueError(
                f"{environment} has {len(candidates)} paths, "
                f"{count_per_environment} required"
            )
        ordered = sorted(
            candidates,
            key=lambda index: (
                stable_int(seed, environment, index),
                index,
            ),
        )
        for dataset_index in ordered[: int(count_per_environment)]:
            _, path_num = dataset.indexDict[int(dataset_index)]
            contexts.append(
                {
                    "environment": str(environment),
                    "dataset_index": int(dataset_index),
                    "path_num": int(path_num),
                }
            )
    return contexts


def stack_items(items):
    return {
        key: torch.stack([item[key].float() for item in items])
        for key in ("map", "start_pose", "goal_pose", "cost_map", "mask")
    }


def fixed_sources(
    contexts: Sequence[Mapping[str, object]],
    *,
    candidates: int,
    num_edges: int,
    seed: int,
) -> torch.Tensor:
    chunks = []
    for context in contexts:
        generator = torch.Generator().manual_seed(
            stable_int(
                seed,
                "path-aligned-reweighting-candidates",
                0,
                context["environment"],
                context["path_num"],
            )
            % (2**63 - 1)
        )
        chunks.append(
            torch.randn(
                int(candidates), int(num_edges), 2, generator=generator
            )
        )
    return torch.cat(chunks, dim=0)


def validity_from_output(output):
    return trajectory_validity_metrics(
        output["geometry"]["position"],
        output["cost_map"],
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=output["geometry"]["yaw"],
        analytic_curvature=output["geometry"]["curvature"],
        analytic_curvature_audit=output["curvature_audit"],
        mask=output["mask"],
        signed_mask_distance_map=output["signed_mask_distance"],
        start_pose=output["start_pose"],
        goal_pose=output["goal_pose"],
        condition_ids=output["condition_ids"],
    )


def chunks(values: Sequence, size: int):
    for offset in range(0, len(values), int(size)):
        yield values[offset : offset + int(size)]


@torch.no_grad()
def collect_partition_groups(
    stage1,
    dataset,
    contexts,
    *,
    partition: str,
    args,
    device,
):
    groups = []
    candidates = int(args.stage2_candidates)
    mask_variants = [
        PARTITION_MASK_BASE[partition] + index + 1
        for index in range(int(args.stage2_mask_copies))
    ]
    stage1.eval()
    for variant in mask_variants:
        for context_batch in chunks(
            contexts, int(args.stage2_collection_batch_size)
        ):
            items = [
                dataset.get_item(
                    int(context["dataset_index"]),
                    mask_variant=int(variant),
                    return_mask_metadata=True,
                )
                for context in context_batch
            ]
            batch = stack_items(items)
            source = fixed_sources(
                context_batch,
                candidates=candidates,
                num_edges=stage1.num_edges,
                seed=int(args.stage2_source_seed),
            )
            started = time.perf_counter()
            output = _direct_cost_forward(
                stage1,
                batch,
                device,
                sources_per_context=candidates,
                source=source,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            validity = validity_from_output(output)
            batch_size = len(context_batch)
            position = output["geometry"]["position"].reshape(
                batch_size, candidates, -1, 2
            )
            curvature = output["geometry"]["curvature"].reshape(
                batch_size, candidates, -1
            )
            strict = validity["strict_valid"].reshape(batch_size, candidates)
            constraint_ok = torch.stack(
                [
                    validity["forbidden_region_ok"],
                    validity["stability_ok"],
                    validity["curvature_ok"],
                ],
                dim=-1,
            ).reshape(batch_size, candidates, 3)
            violation_max = validity["violation_max_normalized"].reshape(
                batch_size, candidates
            )
            violation_integral = validity[
                "violation_integral_normalized_m"
            ].reshape(batch_size, candidates)
            start_condition, goal_condition = normalize_poses(
                batch["start_pose"],
                batch["goal_pose"],
                stage1.coordinate_scale,
                device,
            )
            for index, context in enumerate(context_batch):
                groups.append(
                    {
                        "partition": partition,
                        "environment": str(context["environment"]),
                        "dataset_index": int(context["dataset_index"]),
                        "path_num": int(context["path_num"]),
                        "mask_variant": int(variant),
                        "observed_map": batch["map"][index].cpu(),
                        "start_condition": start_condition[index].cpu(),
                        "goal_condition": goal_condition[index].cpu(),
                        "candidate_path": position[index].cpu(),
                        "candidate_curvature": curvature[index].cpu(),
                        "strict_safe": strict[index].cpu(),
                        "constraint_ok": constraint_ok[index].cpu(),
                        "violation_max": violation_max[index].cpu(),
                        "violation_integral": violation_integral[index].cpu(),
                        "generation_ms": elapsed_ms / batch_size,
                    }
                )
    return groups


def collect_candidate_archive(args, stage1, manifest, device):
    common = {
        "compute_stability_map": True,
        "mask_seed": int(args.mask_seed),
        "p_mask": float(args.stage2_p_mask),
        "mask_mode": "stage2_independent",
        "vehicle_radius_meters": float(args.vehicle_radius_meters),
        "dynamic_mask_noise": False,
    }
    train_environments = sorted(
        manifest["fit_environments"] + manifest["selection_environments"]
    )
    train_set, _ = make_partial_dataset(
        args.dataFolder,
        "train",
        environment_names=train_environments,
        **common,
    )
    validation_set, _ = make_partial_dataset(
        args.dataFolder,
        "val",
        environment_names=manifest["validation_environments"],
        **common,
    )
    fit_contexts = select_contexts(
        train_set,
        manifest["fit_environments"],
        count_per_environment=int(args.stage2_fit_contexts_per_environment),
        seed=stable_int(args.stage2_protocol_seed, "p2-fit-contexts"),
    )
    selection_contexts = select_contexts(
        train_set,
        manifest["selection_environments"],
        count_per_environment=int(
            args.stage2_selection_contexts_per_environment
        ),
        seed=stable_int(args.stage2_protocol_seed, "p2-selection-contexts"),
    )
    validation_contexts = select_contexts(
        validation_set,
        manifest["validation_environments"],
        count_per_environment=int(
            args.stage2_validation_contexts_per_environment
        ),
        seed=stable_int(args.stage2_protocol_seed, "p2-admission-contexts"),
    )
    if args.max_contexts is not None:
        limit = int(args.max_contexts)
        fit_contexts = fit_contexts[:limit]
        selection_contexts = selection_contexts[:limit]
        validation_contexts = validation_contexts[:limit]
    groups = []
    for partition, dataset, contexts in (
        ("fit", train_set, fit_contexts),
        ("selection", train_set, selection_contexts),
        ("validation", validation_set, validation_contexts),
    ):
        partition_groups = collect_partition_groups(
            stage1,
            dataset,
            contexts,
            partition=partition,
            args=args,
            device=device,
        )
        groups.extend(partition_groups)
        print(
            f"Stage 2 candidates {partition}: "
            f"{len(partition_groups)} mask-conditions",
            flush=True,
        )
    return {
        "archive_version": STAGE2_CANDIDATE_ARCHIVE_VERSION,
        "training_semantics": STAGE2_REWEIGHTING_TRAINING_SEMANTICS,
        "stage2_reweighting_semantics": STAGE2_REWEIGHTING_SEMANTICS,
        "source_checkpoint_sha256": sha256_file(args.prior_checkpoint),
        "candidate_count": int(args.stage2_candidates),
        "mask_copies": int(args.stage2_mask_copies),
        "mask_seed": int(args.mask_seed),
        "p_mask": float(args.stage2_p_mask),
        "vehicle_radius_meters": float(args.vehicle_radius_meters),
        "deployable_fields": list(DEPLOYABLE_FIELDS),
        "privileged_fields_are_labels_only": True,
        "same_sources_across_mask_variants": True,
        "environment_manifest": manifest,
        "contexts": {
            "fit": fit_contexts,
            "selection": selection_contexts,
            "validation": validation_contexts,
        },
        "groups": groups,
    }


def validate_candidate_archive(archive, args, manifest):
    if archive.get("archive_version") != STAGE2_CANDIDATE_ARCHIVE_VERSION:
        raise ValueError("Stage-2 candidate archive version mismatch")
    if archive.get("training_semantics") != STAGE2_REWEIGHTING_TRAINING_SEMANTICS:
        raise ValueError("Stage-2 candidate training semantics mismatch")
    if archive.get("source_checkpoint_sha256") != sha256_file(
        args.prior_checkpoint
    ):
        raise ValueError("Candidate archive and Stage-1 checkpoint differ")
    if int(archive.get("candidate_count", -1)) != int(args.stage2_candidates):
        raise ValueError("Candidate archive K differs from training config")
    if int(archive.get("mask_copies", -1)) != int(args.stage2_mask_copies):
        raise ValueError("Candidate archive mask count differs from config")
    if int(archive.get("mask_seed", -1)) != int(args.mask_seed):
        raise ValueError("Candidate archive mask seed differs from config")
    if not np.isclose(
        float(archive.get("p_mask", -1.0)), float(args.stage2_p_mask)
    ):
        raise ValueError("Candidate archive mask probability differs from config")
    if not np.isclose(
        float(archive.get("vehicle_radius_meters", -1.0)),
        float(args.vehicle_radius_meters),
    ):
        raise ValueError("Candidate archive vehicle radius differs from config")
    if archive.get("environment_manifest") != manifest:
        raise ValueError("Candidate archive environment manifest changed")
    if not archive.get("privileged_fields_are_labels_only"):
        raise ValueError("Candidate archive does not enforce privileged labels only")


class CandidateGroupDataset(Dataset):
    def __init__(self, groups):
        self.groups = list(groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate_groups(groups):
    tensor_keys = (
        *DEPLOYABLE_FIELDS,
        "strict_safe",
        "constraint_ok",
        "violation_max",
        "violation_integral",
    )
    result = {
        key: torch.stack([group[key] for group in groups])
        for key in tensor_keys
    }
    result["environment"] = [group["environment"] for group in groups]
    return result


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def continuous_risk(batch, integral_weight):
    return torch.log1p(batch["violation_max"].clamp_min(0.0)) + float(
        integral_weight
    ) * torch.log1p(batch["violation_integral"].clamp_min(0.0))


def pairwise_ranking_loss(score, strict_safe, target_risk):
    priority = (~strict_safe.bool()).float() * 2.0 + target_risk
    target_delta = priority[:, :, None] - priority[:, None, :]
    score_delta = score[:, :, None] - score[:, None, :]
    pair_mask = torch.triu(
        torch.ones_like(target_delta, dtype=torch.bool), diagonal=1
    ) & (target_delta.abs() > 1e-6)
    if not bool(pair_mask.any()):
        return score.sum() * 0.0
    return F.softplus(
        -target_delta.sign()[pair_mask] * score_delta[pair_mask]
    ).mean()


def critic_training_loss(critic, batch, args, positive_weight):
    predicted_risk, safe_logit = critic(
        *(batch[key] for key in DEPLOYABLE_FIELDS)
    )
    target_risk = continuous_risk(batch, args.stage2_integral_weight)
    regression = F.smooth_l1_loss(predicted_risk, target_risk)
    classification = F.binary_cross_entropy_with_logits(
        safe_logit,
        batch["strict_safe"].float(),
        pos_weight=positive_weight,
    )
    score = critic_energy(
        predicted_risk,
        safe_logit,
        args.stage2_classification_score_weight,
    )
    ranking = pairwise_ranking_loss(
        score, batch["strict_safe"], target_risk
    )
    total = (
        regression
        + float(args.stage2_classification_weight) * classification
        + float(args.stage2_ranking_weight) * ranking
    )
    return total, {
        "loss": float(total.detach()),
        "regression": float(regression.detach()),
        "classification": float(classification.detach()),
        "ranking": float(ranking.detach()),
    }


def mode_labels(path: torch.Tensor, threshold: float) -> torch.Tensor:
    start = path[:, :1]
    chord = path[:, -1:] - start
    relative = path - start
    cross = chord[..., 0] * relative[..., 1] - chord[..., 1] * relative[..., 0]
    chord_norm = torch.linalg.vector_norm(chord, dim=-1).clamp_min(1e-6)
    score = torch.mean(cross / chord_norm, dim=1)
    return torch.where(
        score > threshold,
        torch.ones_like(score, dtype=torch.long),
        torch.where(
            score < -threshold,
            -torch.ones_like(score, dtype=torch.long),
            torch.zeros_like(score, dtype=torch.long),
        ),
    )


def entropy(probability: torch.Tensor) -> float:
    probability = probability[probability > 0]
    return float(-(probability * probability.log()).sum())


def true_lexicographic_energy(group, integral_weight):
    strict = group["strict_safe"].bool()
    risk = torch.log1p(group["violation_max"].clamp_min(0.0)) + float(
        integral_weight
    ) * torch.log1p(group["violation_integral"].clamp_min(0.0))
    span = risk.max() - risk.min()
    normalized = (
        torch.zeros_like(risk)
        if float(span) <= 1e-12
        else (risk - risk.min()) / span
    )
    return normalized + 2.0 * (~strict).float()


@torch.no_grad()
def predict_groups(critic, groups, args, device):
    loader = DataLoader(
        CandidateGroupDataset(groups),
        batch_size=int(args.stage2_critic_batch_size),
        shuffle=False,
        collate_fn=collate_groups,
    )
    critic.eval()
    scores = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        risk, safe_logit = critic(*(batch[key] for key in DEPLOYABLE_FIELDS))
        score = critic_energy(
            risk,
            safe_logit,
            args.stage2_classification_score_weight,
        )
        scores.extend(score.cpu())
    return scores


def bootstrap_environment_gain(rows, *, samples: int, seed: int):
    by_environment = defaultdict(list)
    for row in rows:
        by_environment[row["environment"]].append(row["safe_mass_gain"])
    per_environment = np.asarray(
        [np.mean(by_environment[key]) for key in sorted(by_environment)],
        dtype=np.float64,
    )
    generator = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        draws[index] = generator.choice(
            per_environment, size=len(per_environment), replace=True
        ).mean()
    return {
        "mean": float(per_environment.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "positive_environments": int(np.sum(per_environment > 1e-12)),
        "environment_count": int(len(per_environment)),
    }


def evaluate_groups(groups, scores, args):
    rows = []
    for group, score in zip(groups, scores):
        safe = group["strict_safe"].bool()
        constraint_ok = group["constraint_ok"].float()
        uniform = torch.full_like(score, 1.0 / len(score))
        learned = bounded_kl_weights(
            score,
            float(args.stage2_kl_budget),
        )
        oracle = bounded_kl_weights(
            true_lexicographic_energy(group, args.stage2_integral_weight),
            float(args.stage2_kl_budget),
        )
        labels = mode_labels(
            group["candidate_path"], float(args.stage2_mode_threshold_m)
        )
        unique_modes = labels.unique()
        baseline_mode_mass = torch.stack(
            [uniform[labels == mode].sum() for mode in unique_modes]
        )
        learned_mode_mass = torch.stack(
            [learned[labels == mode].sum() for mode in unique_modes]
        )
        baseline_coverage = int(
            (baseline_mode_mass >= float(args.stage2_mode_mass_threshold)).sum()
        )
        learned_coverage = int(
            (learned_mode_mass >= float(args.stage2_mode_mass_threshold)).sum()
        )
        selected = int(score.argmin())
        learned_safe = float(learned @ safe.float())
        uniform_safe = float(uniform @ safe.float())
        oracle_safe = float(oracle @ safe.float())
        rows.append(
            {
                "environment": group["environment"],
                "uniform_safe_mass": uniform_safe,
                "weighted_safe_mass": learned_safe,
                "oracle_safe_mass": oracle_safe,
                "safe_mass_gain": learned_safe - uniform_safe,
                "constraint_mass_change": (
                    learned @ constraint_ok - uniform @ constraint_ok
                ).tolist(),
                "kl": float(empirical_kl_from_uniform(learned.unsqueeze(0))[0]),
                "ess_fraction": float(1.0 / learned.square().sum() / len(learned)),
                "mode_entropy_loss": entropy(baseline_mode_mass)
                - entropy(learned_mode_mass),
                "mode_coverage_retention": learned_coverage
                / max(baseline_coverage, 1),
                "source0_safe": bool(safe[0]),
                "top1_safe": bool(safe[selected]),
            }
        )
    uniform = np.asarray([row["uniform_safe_mass"] for row in rows])
    learned = np.asarray([row["weighted_safe_mass"] for row in rows])
    oracle = np.asarray([row["oracle_safe_mass"] for row in rows])
    changes = np.asarray([row["constraint_mass_change"] for row in rows])
    source0 = np.asarray([row["source0_safe"] for row in rows], dtype=bool)
    top1 = np.asarray([row["top1_safe"] for row in rows], dtype=bool)
    learned_gain = float(np.mean(learned - uniform))
    oracle_gain = float(np.mean(oracle - uniform))
    return {
        "conditions": len(rows),
        "environments": len({row["environment"] for row in rows}),
        "uniform_safe_mass": float(uniform.mean()),
        "weighted_safe_mass": float(learned.mean()),
        "oracle_weighted_safe_mass": float(oracle.mean()),
        "safe_mass_gain": learned_gain,
        "oracle_safe_mass_gain": oracle_gain,
        "oracle_gain_retention": learned_gain / max(oracle_gain, 1e-12),
        "constraint_mass_change": {
            name: float(changes[:, index].mean())
            for index, name in enumerate(
                ("forbidden", "stability", "curvature")
            )
        },
        "kl_max": float(max(row["kl"] for row in rows)),
        "ess_fraction_min": float(min(row["ess_fraction"] for row in rows)),
        "ess_fraction_mean": float(np.mean([row["ess_fraction"] for row in rows])),
        "mode_entropy_loss_mean": float(
            np.mean([row["mode_entropy_loss"] for row in rows])
        ),
        "mode_coverage_retention": float(
            np.mean([row["mode_coverage_retention"] for row in rows])
        ),
        "source0_safe_rate": float(source0.mean()),
        "top1_safe_rate": float(top1.mean()),
        "invalid_to_strict_rate": float(np.mean(top1[~source0]))
        if np.any(~source0)
        else 0.0,
        "strict_to_invalid_rate": float(np.mean(~top1[source0]))
        if np.any(source0)
        else 0.0,
        "bootstrap": bootstrap_environment_gain(
            rows,
            samples=int(args.stage2_bootstrap_samples),
            seed=stable_int(args.seed, "stage2-reweighting-bootstrap"),
        ),
    }


def admission_gate(selection, validation, args):
    gate = {
        "selection_gain_positive": selection["safe_mass_gain"] > 0.0,
        "validation_gain_at_least_threshold": validation["safe_mass_gain"]
        >= float(args.stage2_min_safe_mass_gain),
        "validation_bootstrap_ci_low_positive": validation["bootstrap"][
            "ci95_low"
        ]
        > 0.0,
        "retains_half_oracle_gain": validation["oracle_gain_retention"]
        >= float(args.stage2_min_oracle_gain_retention),
        "no_constraint_mass_regression": min(
            validation["constraint_mass_change"].values()
        )
        >= -float(args.stage2_max_constraint_mass_regression),
        "top1_strict_regression_controlled": validation[
            "strict_to_invalid_rate"
        ]
        <= float(args.stage2_max_regression_rate),
        "ess_fraction_at_least_threshold": validation["ess_fraction_min"]
        >= float(args.stage2_min_ess_fraction),
        "kl_at_most_budget": validation["kl_max"]
        <= float(args.stage2_kl_budget) + 1e-6,
        "mode_coverage_retained": validation["mode_coverage_retention"]
        >= float(args.stage2_min_mode_coverage_retention),
        "gain_not_single_environment": validation["bootstrap"][
            "positive_environments"
        ]
        >= 4,
    }
    gate["passed"] = all(gate.values())
    return gate


def training_config(args):
    keys = (
        "seed",
        "mask_seed",
        "stage2_split_seed",
        "stage2_train_environments",
        "stage2_validation_environments",
        "stage2_candidates",
        "stage2_mask_copies",
        "stage2_source_seed",
        "stage2_protocol_seed",
        "stage2_p_mask",
        "vehicle_radius_meters",
        "stage2_additional_fit_environments",
        "stage2_reserved_environments",
        "stage2_selection_environments",
        "stage2_fit_contexts_per_environment",
        "stage2_selection_contexts_per_environment",
        "stage2_validation_contexts_per_environment",
        "stage2_critic_epochs",
        "stage2_critic_lr",
        "stage2_critic_batch_size",
        "stage2_critic_hidden_dim",
        "stage2_critic_weight_decay",
        "stage2_critic_grad_clip_norm",
        "stage2_kl_budget",
        "stage2_integral_weight",
        "stage2_classification_weight",
        "stage2_ranking_weight",
        "stage2_classification_score_weight",
        "stage2_selection_mode",
        "stage2_mode_threshold_m",
        "stage2_mode_mass_threshold",
        "stage2_bootstrap_samples",
        "stage2_min_safe_mass_gain",
        "stage2_min_oracle_gain_retention",
        "stage2_max_constraint_mass_regression",
        "stage2_max_regression_rate",
        "stage2_min_ess_fraction",
        "stage2_min_mode_coverage_retention",
        "max_contexts",
    )
    return {key: getattr(args, key) for key in keys}


def require_resume_protocol(checkpoint, args):
    if checkpoint.get("resume_capable") is not True:
        raise ValueError(
            "--resume must point to stage2_critic_last.pth with optimizer state"
        )
    saved = checkpoint.get("training_config")
    if not isinstance(saved, Mapping):
        raise ValueError("Resumed Stage-2 critic lacks its frozen protocol")
    current = training_config(args)
    for key, value in current.items():
        if key == "stage2_critic_epochs":
            continue
        if saved.get(key) != value:
            raise ValueError(
                f"Stage-2 resume config mismatch for {key}: "
                f"checkpoint={saved.get(key)!r}, requested={value!r}"
            )


def rng_checkpoint_metadata(data_generator):
    metadata = {
        "torch_rng_state": torch.get_rng_state(),
        "data_generator_state": data_generator.get_state(),
    }
    if torch.cuda.is_available():
        metadata["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return metadata


def restore_rng_state(checkpoint, data_generator):
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )
    if checkpoint.get("data_generator_state") is not None:
        data_generator.set_state(checkpoint["data_generator_state"].cpu())


def run_stage2_reweighting_training(args, device):
    """Train only the deployable critic while keeping Stage 1 frozen."""
    output = Path(args.fileDir)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    manifest = expanded_environment_manifest(args)
    config = Stage2ReweightingConfig(
        candidates=int(args.stage2_candidates),
        kl_budget=float(args.stage2_kl_budget),
        critic_hidden_dim=int(args.stage2_critic_hidden_dim),
        map_channels=4,
        classification_score_weight=float(
            args.stage2_classification_score_weight
        ),
        selection_mode=str(args.stage2_selection_mode),
    )

    if args.resume:
        critic, saved_config, checkpoint = load_critic_checkpoint(
            args.resume, device
        )
        if saved_config != config:
            raise ValueError("Resumed Stage-2 critic config differs")
        require_resume_protocol(checkpoint, args)
        args.prior_checkpoint = checkpoint.get("source_checkpoint")
        if not args.prior_checkpoint:
            raise ValueError("Resumed critic lacks its Stage-1 checkpoint")
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
    else:
        if not args.prior_checkpoint:
            raise ValueError(
                "Path-aligned Stage 2 requires --prior_checkpoint"
            )
        critic = PathAlignedCandidateCritic(
            map_channels=config.map_channels,
            hidden_dim=config.critic_hidden_dim,
        ).to(device)
        checkpoint = {}
        start_epoch = 0

    stage1, _, stage1_checkpoint = load_model(
        Path(args.prior_checkpoint), device
    )
    _require_main_method_model(stage1)
    _require_current_mask_semantics(stage1_checkpoint)
    _require_demo_target_semantics(stage1_checkpoint)
    if stage1_checkpoint.get("vehicle_radius_meters") is not None and not np.isclose(
        float(stage1_checkpoint["vehicle_radius_meters"]),
        float(args.vehicle_radius_meters),
    ):
        raise ValueError("Stage-1 vehicle radius differs from Stage-2 config")
    stage1.eval()
    stage1.use_gradient_checkpoint = False
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)

    archive_path = Path(
        args.stage2_candidate_archive
        or checkpoint.get("candidate_archive", output / "stage2_candidates.pt")
    )
    if archive_path.exists():
        archive = torch.load(archive_path, map_location="cpu")
    else:
        archive = collect_candidate_archive(args, stage1, manifest, device)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(archive, archive_path)
    validate_candidate_archive(archive, args, manifest)
    del stage1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fit_groups = [
        group for group in archive["groups"] if group["partition"] == "fit"
    ]
    selection_groups = [
        group
        for group in archive["groups"]
        if group["partition"] == "selection"
    ]
    validation_groups = [
        group
        for group in archive["groups"]
        if group["partition"] == "validation"
    ]
    if not fit_groups or not selection_groups or not validation_groups:
        raise ValueError("Stage-2 candidate archive lacks a required partition")

    data_generator = torch.Generator().manual_seed(int(args.seed) + 60_000)
    if args.resume:
        restore_rng_state(checkpoint, data_generator)
    loader = DataLoader(
        CandidateGroupDataset(fit_groups),
        batch_size=int(args.stage2_critic_batch_size),
        shuffle=True,
        generator=data_generator,
        collate_fn=collate_groups,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    safe_count = sum(int(group["strict_safe"].sum()) for group in fit_groups)
    candidate_count = sum(len(group["strict_safe"]) for group in fit_groups)
    positive_weight = torch.tensor(
        min((candidate_count - safe_count) / max(safe_count, 1), 50.0),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(args.stage2_critic_lr),
        weight_decay=float(args.stage2_critic_weight_decay),
    )
    if checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    writer = _make_tensorboard_writer(
        output,
        "stage2_path_aligned_reweighting",
        purge_step=start_epoch if args.resume else None,
    )
    trace = list(checkpoint.get("training_trace", []))
    for epoch in range(start_epoch, int(args.stage2_critic_epochs)):
        critic.train()
        sums = defaultdict(float)
        batches = 0
        progress = tqdm(
            loader,
            desc=f"Stage 2 critic epoch {epoch + 1}",
        )
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = critic_training_loss(
                critic, batch, args, positive_weight
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(args.stage2_critic_grad_clip_norm)
            )
            optimizer.step()
            batches += 1
            for key, value in terms.items():
                sums[key] += float(value)
            sums["gradient_norm_pre_clip"] += float(gradient_norm)
            progress.set_postfix(
                loss=f"{terms['loss']:.4f}",
                rank=f"{terms['ranking']:.4f}",
            )
        row = {
            key: value / max(batches, 1) for key, value in sums.items()
        }
        row["epoch"] = epoch + 1
        trace.append(row)
        _log_numeric_tree(
            writer,
            "stage2_reweighting/train",
            row,
            epoch + 1,
        )
        writer.add_scalar(
            "stage2_reweighting/train/learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch + 1,
        )
        save_critic_checkpoint(
            output / "stage2_critic_last.pth",
            critic,
            config,
            source_checkpoint=args.prior_checkpoint,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            resume_capable=True,
            candidate_archive=str(archive_path.resolve()),
            environment_manifest=manifest,
            training_config=training_config(args),
            training_trace=trace,
            final_test_opened=False,
            **rng_checkpoint_metadata(data_generator),
        )
        writer.flush()

    partition_metrics = {}
    for partition, groups in (
        ("fit", fit_groups),
        ("selection", selection_groups),
        ("validation", validation_groups),
    ):
        scores = predict_groups(critic, groups, args, device)
        partition_metrics[partition] = evaluate_groups(
            groups, scores, args
        )
        _log_numeric_tree(
            writer,
            f"stage2_reweighting/{partition}",
            partition_metrics[partition],
            int(args.stage2_critic_epochs),
        )
    gate = admission_gate(
        partition_metrics["selection"],
        partition_metrics["validation"],
        args,
    )
    final_checkpoint = output / "stage2_critic_last.pth"
    save_critic_checkpoint(
        final_checkpoint,
        critic,
        config,
        source_checkpoint=args.prior_checkpoint,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=int(args.stage2_critic_epochs) - 1,
        resume_capable=True,
        candidate_archive=str(archive_path.resolve()),
        environment_manifest=manifest,
        training_config=training_config(args),
        training_trace=trace,
        metrics=partition_metrics,
        admission_gate=gate,
        checkpoint_selection="fixed_last_epoch",
        final_test_opened=False,
        **rng_checkpoint_metadata(data_generator),
    )
    if gate["passed"]:
        save_critic_checkpoint(
            output / "stage2_critic_best.pth",
            critic,
            config,
            source_checkpoint=args.prior_checkpoint,
            optimizer_state_dict=None,
            epoch=int(args.stage2_critic_epochs) - 1,
            resume_capable=False,
            candidate_archive=str(archive_path.resolve()),
            environment_manifest=manifest,
            training_config=training_config(args),
            metrics=partition_metrics,
            admission_gate=gate,
            checkpoint_selection="fixed_last_epoch_admitted",
            final_test_opened=False,
        )
    method = {
        "method": "frozen_path_meanflow_path_aligned_privileged_reweighting",
        "stage1": "frozen_conditional_path_meanflow",
        "stage2": "deployable_path_aligned_candidate_critic",
        "stage2_reweighting_semantics": STAGE2_REWEIGHTING_SEMANTICS,
        "training_semantics": STAGE2_REWEIGHTING_TRAINING_SEMANTICS,
        "critic_config": config.to_dict(),
        "training_config": training_config(args),
        "environment_manifest": manifest,
        "candidate_archive": str(archive_path.resolve()),
        "privileged_fields_are_labels_only": True,
        "stage1_frozen": True,
        "inference_uses_privileged_map": False,
        "inference_uses_optimizer": False,
        "checkpoint_selection": "fixed_last_epoch",
        "metrics": partition_metrics,
        "admission_gate": gate,
        "final_test_opened": False,
    }
    write_json(output / "stage2_method.json", method)
    writer.close()
    print(
        "Stage 2 path-aligned reweighting "
        f"validation gain={partition_metrics['validation']['safe_mass_gain']:+.2%} "
        f"CI=[{partition_metrics['validation']['bootstrap']['ci95_low']:+.2%}, "
        f"{partition_metrics['validation']['bootstrap']['ci95_high']:+.2%}] "
        f"gate={'PASS' if gate['passed'] else 'FAIL'}",
        flush=True,
    )
    return final_checkpoint
