#!/usr/bin/env python3
"""Precompute yaw-aware stability/ESDF maps for selected dataset environments."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataLoader_dit import (
    STABILITY_MAP_FORMAT_VERSION,
    STABILITY_MAP_SEMANTIC_VERSION,
    compute_map_yaw_bins,
    generate_sdf_from_yaw_stability,
    stability_cache_is_compatible,
    stability_source_map_sha256,
)
from map_config import MAP_CONFIG, MAP_YAW_BINS, SAFETY_COST_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=MAP_CONFIG.dataset_root,
        help="Dataset containing train/val split directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split directories to process.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment names; omit to process every valid environment.",
    )
    parser.add_argument("--output-name", default="stability_map.npz")
    parser.add_argument("--yaw-bins", type=int, default=MAP_YAW_BINS)
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=SAFETY_COST_CONFIG.yaw_esdf_weight,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing compatible or incompatible cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate targets and print the plan without writing files.",
    )
    return parser.parse_args()


def discover_envs(split_folder: Path) -> list[str]:
    return [
        path.name
        for path in sorted(split_folder.iterdir())
        if path.is_dir() and (path / "map.p").is_file()
    ]


def load_map_geometry(map_file: Path) -> tuple[np.ndarray, float, tuple[float, ...]]:
    with map_file.open("rb") as handle:
        map_data = pickle.load(handle)
    tensor = np.asarray(map_data["tensor"])
    if tensor.ndim != 3 or tensor.shape[-1] < 4:
        raise ValueError(f"{map_file}: expected (H,W,>=4), got {tensor.shape}")
    resolution = float(map_data.get("resolution", MAP_CONFIG.resolution))
    bounds = tuple(map_data.get("bounds", MAP_CONFIG.bounds))
    if tuple(tensor.shape[:2]) != MAP_CONFIG.map_shape:
        raise ValueError(
            f"{map_file}: shape {tensor.shape[:2]} != {MAP_CONFIG.map_shape}"
        )
    if not np.isclose(resolution, MAP_CONFIG.resolution, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"{map_file}: resolution {resolution} != {MAP_CONFIG.resolution}"
        )
    if not np.allclose(bounds, MAP_CONFIG.bounds, atol=1e-6, rtol=0.0):
        raise ValueError(f"{map_file}: bounds {bounds} != {MAP_CONFIG.bounds}")
    return tensor, resolution, bounds


def cache_is_compatible(
    output_file: Path,
    map_shape: tuple[int, int],
    resolution: float,
    yaw_bins: int,
    yaw_weight: float,
    source_map_sha256: str,
) -> bool:
    try:
        with np.load(output_file) as cached:
            return stability_cache_is_compatible(
                cached,
                map_shape=map_shape,
                resolution=resolution,
                yaw_bins=yaw_bins,
                yaw_weight=yaw_weight,
                source_map_sha256=source_map_sha256,
            )
    except (KeyError, OSError, ValueError):
        return False


def process_one_env(
    env_path: Path,
    output_name: str,
    yaw_bins: int,
    yaw_weight: float,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[str, Path]:
    map_file = env_path / "map.p"
    if not map_file.is_file():
        return "missing_map", map_file
    tensor, resolution, _ = load_map_geometry(map_file)
    source_hash = stability_source_map_sha256(map_file)
    output_file = env_path / output_name
    if output_file.exists() and not overwrite:
        if cache_is_compatible(
            output_file,
            tuple(tensor.shape[:2]),
            resolution,
            yaw_bins,
            yaw_weight,
            source_hash,
        ):
            return "skip", output_file
        return "incompatible", output_file
    if dry_run:
        return "planned", output_file

    yaw_stability = compute_map_yaw_bins(
        tensor[:, :, 1],
        tensor[:, :, 2],
        tensor[:, :, 3],
        yaw_bins=yaw_bins,
    )
    cost_map = generate_sdf_from_yaw_stability(
        yaw_stability,
        voxel_size_xy=resolution,
        yaw_weight=yaw_weight,
    )
    if hasattr(yaw_stability, "detach"):
        yaw_stability = yaw_stability.detach().cpu().numpy()
    if hasattr(cost_map, "detach"):
        cost_map = cost_map.detach().cpu().numpy()

    temporary = output_file.with_name(output_file.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        yaw_stability=np.asarray(yaw_stability, dtype=np.float32),
        cost_map=np.asarray(cost_map, dtype=np.float32),
        yaw_bins=np.int32(yaw_bins),
        voxel_size_xy=np.float32(resolution),
        yaw_weight=np.float32(yaw_weight),
        map_shape=np.asarray(tensor.shape[:2], dtype=np.int32),
        map_bounds=np.asarray(MAP_CONFIG.bounds, dtype=np.float32),
        map_size_meters=np.float32(MAP_CONFIG.size_meters),
        format_version=np.int32(STABILITY_MAP_FORMAT_VERSION),
        semantic_version=np.asarray(STABILITY_MAP_SEMANTIC_VERSION),
        source_map_sha256=np.asarray(source_hash),
    )
    os.replace(temporary, output_file)
    return "ok", output_file


def resolve_tasks(
    dataset_root: Path,
    splits: Sequence[str],
    requested_envs: Sequence[str] | None,
) -> list[tuple[str, str, Path]]:
    tasks: list[tuple[str, str, Path]] = []
    for split in splits:
        split_folder = dataset_root / split
        if not split_folder.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_folder}")
        environments = (
            list(requested_envs)
            if requested_envs is not None
            else discover_envs(split_folder)
        )
        for environment in environments:
            env_path = split_folder / environment
            if not env_path.is_dir():
                raise FileNotFoundError(f"Missing environment: {env_path}")
            tasks.append((split, environment, env_path))
    return tasks


def main() -> None:
    args = parse_args()
    if args.yaw_bins <= 1:
        raise ValueError("--yaw-bins must be greater than one")
    if args.yaw_weight <= 0:
        raise ValueError("--yaw-weight must be positive")
    tasks = resolve_tasks(args.dataset_root, args.splits, args.envs)
    print(f"Dataset: {args.dataset_root}")
    print(f"Splits: {args.splits}; environments/tasks: {len(tasks)}")
    print(
        f"Geometry: {MAP_CONFIG.size_meters:g} m, "
        f"{MAP_CONFIG.grid_size}x{MAP_CONFIG.grid_size}, "
        f"resolution={MAP_CONFIG.resolution:g} m"
    )
    print(
        f"Yaw ESDF: bins={args.yaw_bins}, yaw_weight={args.yaw_weight:g}, "
        f"output={args.output_name}"
    )

    counts: dict[str, int] = {}
    with tqdm(tasks, desc="Generating stability maps") as progress:
        for split, environment, env_path in progress:
            status, output = process_one_env(
                env_path=env_path,
                output_name=args.output_name,
                yaw_bins=args.yaw_bins,
                yaw_weight=args.yaw_weight,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            counts[status] = counts.get(status, 0) + 1
            if status in {"missing_map", "incompatible"}:
                raise RuntimeError(
                    f"{split}/{environment}: {status}: {output}. "
                    "Use --overwrite only after verifying the intended geometry."
                )
            progress.set_postfix(last=f"{split}/{environment}", status=status)
    print("Completed:", ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
