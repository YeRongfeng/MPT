#!/usr/bin/env python3
"""Train one controlled A/B/C/D Trajectory MeanFlow variant from scratch."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bspline_utils import DifferentiableBSpline
from direct_safe_distribution.splits import (
    load_manifest,
    sha256_file,
    verify_selected_split_files,
)
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from geometry.canonicalization import (
    canonicalize_map,
    canonicalize_pose,
    canonicalize_trajectory,
)
from map_config import MAP_CONFIG
from train_dit import diffusion_loss


VARIANTS = {
    "A": ("raw", False),
    "B": ("raw", True),
    "C": ("safe", False),
    "D": ("safe", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/split_manifest.json"
        ),
    )
    parser.add_argument("--model-template", type=Path, required=True)
    parser.add_argument(
        "--teacher-dataset",
        type=Path,
        default=Path(
            "diagnostics/direct_safe_distribution/"
            "safety_teacher_dataset/candidates.npz"
        ),
    )
    parser.add_argument(
        "--safe-target-policy",
        choices=("strict", "strict_or_improved"),
        default="strict",
    )
    parser.add_argument("--max-paths-per-map", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--canvas-half-extent",
        type=float,
        default=MAP_CONFIG.size_meters * math.sqrt(2.0),
        help="Common A/B/C/D physical half extent in meters.",
    )
    parser.add_argument(
        "--legacy-global-input",
        action="store_true",
        help="Global-only compatibility mode: no raster resampling, scale=10.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("diagnostics/direct_safe_distribution"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MapCache:
    def __init__(self, root: Path, max_items: int = 8):
        self.root = root
        self.max_items = max_items
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, map_id: str) -> torch.Tensor:
        if map_id in self.cache:
            value = self.cache.pop(map_id)
            self.cache[map_id] = value
            return value
        with (self.root / map_id / "map.p").open("rb") as handle:
            raw = np.asarray(pickle.load(handle)["tensor"], dtype=np.float32)
        value = torch.from_numpy(raw[..., 1:4].copy()).permute(2, 0, 1)
        self.cache[map_id] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


class DirectTargetDataset(Dataset):
    """Raw or teacher targets with one common global/canonical preprocessing."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, object],
        split: str,
        target_kind: str,
        canonical: bool,
        canvas_half_extent: float,
        teacher_dataset: Optional[Path] = None,
        safe_target_policy: str = "strict",
        max_paths_per_map: Optional[int] = None,
        legacy_global_input: bool = False,
    ):
        if split not in ("train", "validation"):
            raise ValueError("Training dataset may only open train or validation")
        if target_kind not in ("raw", "safe"):
            raise ValueError(f"Unknown target kind: {target_kind}")
        if target_kind == "safe" and split != "train":
            raise ValueError("Safety teacher targets are train-map-only")
        if legacy_global_input and canonical:
            raise ValueError("Legacy no-resample mode is global-only")
        self.root = Path(manifest["dataset_folder"])
        self.map_ids = list(manifest["splits"][split])
        self.target_kind = target_kind
        self.canonical = canonical
        self.canvas_half_extent = float(canvas_half_extent)
        self.legacy_global_input = legacy_global_input
        self.map_cache = _MapCache(self.root)
        self.bspline = DifferentiableBSpline(26, 100, 3)
        self.representation = PhysicalScaledEdgeResidualRepresentation()
        self.records: List[Tuple[object, ...]] = []
        if target_kind == "raw":
            for map_id in self.map_ids:
                files = sorted(
                    (self.root / map_id).glob("path_*.p"),
                    key=lambda path: int(path.stem.split("_")[-1]),
                )
                if max_paths_per_map is not None:
                    files = files[:max_paths_per_map]
                self.records.extend(
                    (map_id, int(path.stem.split("_")[-1]), path)
                    for path in files
                )
        else:
            if teacher_dataset is None:
                raise ValueError("teacher_dataset is required for safe targets")
            with np.load(teacher_dataset) as data:
                map_id = data["map_id"].astype(str)
                if safe_target_policy == "strict":
                    mask = data["training_eligible_strict"].astype(bool)
                else:
                    mask = data["training_eligible_improved"].astype(bool)
                if not set(map_id[mask]).issubset(set(self.map_ids)):
                    raise ValueError("Teacher dataset contains non-train map IDs")
                for index in np.flatnonzero(mask):
                    self.records.append(
                        (
                            str(map_id[index]),
                            int(data["path_num"][index]),
                            np.asarray(data["start_pose"][index], np.float32),
                            np.asarray(data["goal_pose"][index], np.float32),
                            np.asarray(
                                data["optimized_residual"][index], np.float32
                            ),
                            bool(data["strict_safe"][index]),
                        )
                    )
        if not self.records:
            raise ValueError(
                f"No {target_kind} records for split={split}, policy="
                f"{safe_target_policy}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _target(self, index: int) -> Tuple[str, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        map_id, path_num = str(record[0]), int(record[1])
        if self.target_kind == "raw":
            with Path(record[2]).open("rb") as handle:
                path = np.asarray(pickle.load(handle)["path"], dtype=np.float32)
            trajectory = torch.from_numpy(path)
            start = trajectory[0].clone()
            goal = trajectory[-1].clone()
        else:
            start = torch.from_numpy(record[2].copy())
            goal = torch.from_numpy(record[3].copy())
            residual = torch.from_numpy(record[4].copy()).unsqueeze(0)
            control = self.representation.decode(
                residual,
                (start[:2] / MAP_CONFIG.half_extent).unsqueeze(0),
                (goal[:2] / MAP_CONFIG.half_extent).unsqueeze(0),
            )[0] * MAP_CONFIG.half_extent
            xy = self.bspline(control.unsqueeze(0))[0]
            trajectory = torch.cat(
                (xy, torch.zeros(len(xy), 1, dtype=xy.dtype)), dim=1
            )
        return map_id, path_num, start, goal, trajectory

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        map_id, path_num, start, goal, trajectory = self._target(index)
        normal_map = self.map_cache.get(map_id)
        if self.legacy_global_input:
            transformed_map = canonicalize_map(
                normal_map,
                start,
                goal,
                source_bounds=(-10.0, 9.8, -10.0, 9.8),
                enabled=False,
            )
            map_value = transformed_map.values
            valid_mask = transformed_map.valid_mask
            transformed_start = start
            transformed_goal = goal
            transformed_trajectory = trajectory
        else:
            output_bounds = (
                -self.canvas_half_extent,
                self.canvas_half_extent,
                -self.canvas_half_extent,
                self.canvas_half_extent,
            )
            if self.canonical:
                map_result = canonicalize_map(
                    normal_map,
                    start,
                    goal,
                    source_bounds=(-10.0, 9.8, -10.0, 9.8),
                    output_bounds=output_bounds,
                )
                transformed_start = canonicalize_pose(start, start, goal)
                transformed_goal = canonicalize_pose(goal, start, goal)
                transformed_trajectory = canonicalize_trajectory(
                    trajectory.unsqueeze(0),
                    start.unsqueeze(0),
                    goal.unsqueeze(0),
                )[0]
            else:
                identity_start = torch.tensor(
                    [0.0, 0.0, 0.0], dtype=start.dtype
                )
                identity_goal = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=goal.dtype
                )
                map_result = canonicalize_map(
                    normal_map,
                    identity_start,
                    identity_goal,
                    source_bounds=(-10.0, 9.8, -10.0, 9.8),
                    output_bounds=output_bounds,
                )
                transformed_start = start
                transformed_goal = goal
                transformed_trajectory = trajectory
            map_value = map_result.values
            valid_mask = map_result.valid_mask
        return {
            "map": map_value,
            "trajectory": transformed_trajectory,
            "start_pose": transformed_start,
            "goal_pose": transformed_goal,
            "valid_fraction": valid_mask.mean(),
            "record_index": torch.tensor(index),
            "path_num": torch.tensor(path_num),
        }


def collate_direct(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def build_model_args(
    template: Mapping[str, object], coordinate_scale: float
) -> Dict[str, object]:
    args = dict(template["model_args"])
    args["coordinate_scale"] = coordinate_scale
    args["privileged_local_dim"] = 0
    return args


def run_epoch(
    model: PathDiffusionTransformer,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    total_epochs: int,
    seed: int,
    loss_weights: Mapping[str, float],
    grad_clip: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator(device=device).manual_seed(seed + epoch * 1009)
    totals: Dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, _, _, diagnostics = diffusion_loss(
                model,
                batch,
                device,
                loss_weights=dict(loss_weights),
                epoch=epoch,
                total_epochs=total_epochs,
                is_training=training,
                current_stage=1,
                prediction_type=model.prediction_type,
                sampling_generator=generator,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        count = len(batch["map"])
        sample_count += count
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
        totals["valid_fraction"] = totals.get(
            "valid_fraction", 0.0
        ) + float(batch["valid_fraction"].mean()) * count
        for key in ("main", "main_raw_mse", "boundary", "oob_rate", "max_oob"):
            if key in diagnostics:
                value = diagnostics[key]
                if torch.is_tensor(value):
                    value = float(value.detach().mean())
                totals[key] = totals.get(key, 0.0) + float(value) * count
    return {key: value / sample_count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    target_kind, canonical = VARIANTS[args.variant]
    if args.legacy_global_input and canonical:
        raise ValueError("--legacy-global-input cannot be canonical")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    device = torch.device(args.device)
    manifest = load_manifest(args.manifest, verify_files=False)
    verify_selected_split_files(manifest, ("train", "validation"))
    template = json.loads(args.model_template.read_text(encoding="utf-8"))
    coordinate_scale = (
        MAP_CONFIG.half_extent
        if args.legacy_global_input
        else args.canvas_half_extent
    )
    model_args = build_model_args(template, coordinate_scale)
    model = PathDiffusionTransformer(**model_args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_dataset = DirectTargetDataset(
        manifest=manifest,
        split="train",
        target_kind=target_kind,
        canonical=canonical,
        canvas_half_extent=coordinate_scale,
        teacher_dataset=args.teacher_dataset if target_kind == "safe" else None,
        safe_target_policy=args.safe_target_policy,
        max_paths_per_map=args.max_paths_per_map,
        legacy_global_input=args.legacy_global_input,
    )
    # Validation is always raw and never uses a privileged teacher.
    validation_dataset = DirectTargetDataset(
        manifest=manifest,
        split="validation",
        target_kind="raw",
        canonical=canonical,
        canvas_half_extent=coordinate_scale,
        max_paths_per_map=args.max_paths_per_map,
        legacy_global_input=args.legacy_global_input,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        collate_fn=collate_direct,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_direct,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_weights = dict(
        template.get("stage1_config", {}).get(
            "loss_weights",
            {
                "main": 0.01,
                "main_norm_p": 1.0,
                "boundary": 0.1,
                "boundary_safe_bound": 0.98,
            },
        )
    )
    output_dir = args.output_root / f"experiment_{args.variant}" / (
        f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "variant": args.variant,
        "target_kind": target_kind,
        "canonical": canonical,
        "from_scratch": True,
        "arguments": vars(args),
        "model_args": model_args,
        "parameter_count": parameter_count,
        "loss_weights": loss_weights,
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "train_map_ids": manifest["splits"]["train"],
        "validation_map_ids": manifest["splits"]["validation"],
        "test_maps_opened": False,
        "teacher_dataset_sha256": (
            sha256_file(args.teacher_dataset)
            if target_kind == "safe"
            else None
        ),
        "coordinate_statement": (
            "continuous canonical formula is exact; raster grid_sample is "
            "approximate and validity padding is measured"
        ),
        "inference_boundary_wrapper": (
            "rotate residual to global, multiply by canvas_scale/10, then "
            "apply the unchanged PhysicalScaledEdgeResidualRepresentation "
            "radial_project_residual against original map bounds"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            seed=args.seed,
            loss_weights=loss_weights,
            grad_clip=args.grad_clip,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            seed=args.seed + 1_000_000,
            loss_weights=loss_weights,
            grad_clip=args.grad_clip,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_raw_monitor": validation_metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{args.variant} epoch {epoch + 1}/{args.epochs}: "
            f"train={train_metrics['loss']:.6f}, "
            f"val_raw={validation_metrics['loss']:.6f}"
        )

    data_protocol = {
        "split_manifest_sha256": manifest["manifest_sha256"],
        "training_map_ids": manifest["splits"]["train"],
        "allowed_splits": ["train"],
        "validation_map_ids": manifest["splits"]["validation"],
        "test_maps_opened": False,
    }
    checkpoint = {
        "epoch": args.epochs - 1,
        "stage": "direct_full_training",
        "variant": args.variant,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_args": model_args,
        "data_protocol": data_protocol,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "final_model.pth")
    (output_dir / "model_params.json").write_text(
        json.dumps(
            {
                "model_args": model_args,
                "map_config": MAP_CONFIG.to_dict(),
                "split_manifest_sha256": manifest["manifest_sha256"],
                "variant": args.variant,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed {output_dir}")


if __name__ == "__main__":
    main()
