"""Shared dataset loading and hard-validity evaluation for baseline planners."""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

sys.modules.setdefault("numpy._core", np)
if hasattr(np, "core"):
    sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

from dataLoader_dit import (  # noqa: E402
    erode_mask_for_vehicle,
    generate_random_mask,
    normalize_mask,
)
from grad_optimizer import trajectory_validity_metrics  # noqa: E402
from map_config import (  # noqa: E402
    DENSE_TRAJECTORY_POINTS,
    MAP_CONFIG,
    MAP_RESOLUTION,
    SAFETY_COST_CONFIG,
)


def wrap_angle(yaw: np.ndarray) -> np.ndarray:
    return (np.asarray(yaw, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def world_to_grid(x: float, y: float, *, shape: Tuple[int, int]) -> Tuple[int, int]:
    origin_x, origin_y = MAP_CONFIG.origin_xy
    col = int((float(x) - origin_x) / MAP_RESOLUTION)
    row = int((float(y) - origin_y) / MAP_RESOLUTION)
    row = max(0, min(int(shape[0]) - 1, row))
    col = max(0, min(int(shape[1]) - 1, col))
    return row, col


def grid_to_world(row: int, col: int) -> Tuple[float, float]:
    origin_x, origin_y = MAP_CONFIG.origin_xy
    x = origin_x + (float(col) + 0.5) * MAP_RESOLUTION
    y = origin_y + (float(row) + 0.5) * MAP_RESOLUTION
    return x, y


def resample_xy(path_xy: np.ndarray, num_points: int = DENSE_TRAJECTORY_POINTS) -> np.ndarray:
    points = np.asarray(path_xy, dtype=np.float64)[..., :2]
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError(f"path_xy must be (N,2) with N>=2, got {points.shape}")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    chord = np.concatenate([[0.0], np.cumsum(segment)])
    if float(chord[-1]) < 1e-6:
        return np.repeat(points[:1], num_points, axis=0).astype(np.float32)
    samples = np.linspace(0.0, chord[-1], int(num_points))
    x = np.interp(samples, chord, points[:, 0])
    y = np.interp(samples, chord, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def resample_se2(path: np.ndarray, num_points: int = DENSE_TRAJECTORY_POINTS) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 2 or path.shape[0] < 2:
        raise ValueError(f"path must be (N,2|3) with N>=2, got {path.shape}")
    xy = resample_xy(path[:, :2], num_points)
    if path.shape[1] < 3:
        yaw, _ = yaw_and_curvature_from_xy(xy)
        return np.concatenate([xy, yaw[:, None]], axis=1)
    segment = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    chord = np.concatenate([[0.0], np.cumsum(segment)])
    samples = np.linspace(0.0, chord[-1], int(num_points))
    unwrapped = np.unwrap(path[:, 2])
    yaw = np.interp(samples, chord, unwrapped)
    return np.concatenate([xy, wrap_angle(yaw)[:, None]], axis=1).astype(np.float32)


def yaw_and_curvature_from_xy(path_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(path_xy, dtype=np.float64)[..., :2]
    delta = np.gradient(xy, axis=0)
    second = np.gradient(delta, axis=0)
    speed = np.linalg.norm(delta, axis=1)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    cross = delta[:, 0] * second[:, 1] - delta[:, 1] * second[:, 0]
    curvature = cross / np.maximum(speed, 1e-6) ** 3
    yaw[0] = yaw[1] if xy.shape[0] > 1 else 0.0
    yaw[-1] = yaw[-2] if xy.shape[0] > 1 else yaw[0]
    return wrap_angle(yaw).astype(np.float32), curvature.astype(np.float32)


def path_length(path_xy: np.ndarray) -> float:
    points = np.asarray(path_xy, dtype=np.float64)[..., :2]
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


@dataclass
class PlanningTask:
    environment: str
    path_id: int
    start: np.ndarray
    goal: np.ndarray
    expert_path: np.ndarray
    elevation: np.ndarray
    normals: np.ndarray
    mask: np.ndarray
    occupancy: np.ndarray
    cost_map: Optional[np.ndarray]
    mask_metadata: Dict[str, Any]


def load_environment(env_dir: Path) -> Dict[str, Any]:
    env_dir = Path(env_dir)
    with open(env_dir / "map.p", "rb") as handle:
        env = pickle.load(handle)
    tensor = np.asarray(env["tensor"], dtype=np.float32)
    stability_file = env_dir / "stability_map.npz"
    cost_map = None
    if stability_file.is_file():
        with np.load(stability_file) as stability:
            cost_map = np.asarray(stability["cost_map"], dtype=np.float32)
    return {
        "environment": env_dir.name,
        "tensor": tensor,
        "elevation": tensor[:, :, 0],
        "normals": tensor[:, :, 1:4],
        "map_mask": (
            normalize_mask(env["mask"], tensor.shape[:2], source=f"{env_dir}/map.p['mask']")
            if "mask" in env
            else None
        ),
        "cost_map": cost_map,
        "bounds": tuple(env.get("bounds", MAP_CONFIG.bounds)),
        "resolution": float(env.get("resolution", MAP_RESOLUTION)),
    }


def load_expert_path(env_dir: Path, path_id: int) -> np.ndarray:
    with open(Path(env_dir) / f"path_{int(path_id)}.p", "rb") as handle:
        payload = pickle.load(handle)
    return np.asarray(payload["path"], dtype=np.float32)


def build_task_mask(
    elevation: np.ndarray,
    expert_xy: np.ndarray,
    *,
    mask_mode: str,
    path_id: int,
    mask_seed: int,
    p_mask: float,
    map_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if mask_mode == "full":
        mask = np.ones(elevation.shape, dtype=np.float32)
        return mask, {"accepted_type": "full_map", "masked_fraction": 0.0}
    if map_mask is not None:
        mask = erode_mask_for_vehicle(map_mask)
        return mask, {"accepted_type": "map_data"}
    mask, metadata = generate_random_mask(
        elevation.shape,
        int(mask_seed) + int(path_id) * 1_000_003,
        expert_xy,
        p_mask=float(p_mask),
        require_trajectory_clear=(mask_mode != "stage2_independent"),
        return_metadata=True,
    )
    return mask, metadata


def occupancy_from_mask(mask: np.ndarray) -> np.ndarray:
    """True where the vehicle center cannot go. Unknown/forbidden are obstacles."""
    return np.asarray(mask, dtype=np.float32) < 0.5


def make_task(
    env_dir: Path,
    path_id: int,
    *,
    mask_mode: str = "partial",
    mask_seed: int = 20260817,
    p_mask: float = 1.0,
) -> PlanningTask:
    env = load_environment(env_dir)
    expert = load_expert_path(env_dir, path_id)
    mask, metadata = build_task_mask(
        env["elevation"],
        expert[:, :2],
        mask_mode=mask_mode,
        path_id=path_id,
        mask_seed=mask_seed,
        p_mask=p_mask,
        map_mask=env["map_mask"],
    )
    return PlanningTask(
        environment=env["environment"],
        path_id=int(path_id),
        start=expert[0].astype(np.float32),
        goal=expert[-1].astype(np.float32),
        expert_path=expert,
        elevation=env["elevation"],
        normals=env["normals"],
        mask=mask,
        occupancy=occupancy_from_mask(mask),
        cost_map=env["cost_map"],
        mask_metadata=metadata,
    )


def evaluate_xy_path(
    path_xy: np.ndarray,
    task: PlanningTask,
    *,
    yaw: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    if task.cost_map is None:
        raise FileNotFoundError(
            f"stability_map.npz missing for {task.environment}; cannot run hard validity"
        )
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] >= 3:
        se2 = resample_se2(path)
        dense = se2[:, :2]
        est_yaw = se2[:, 2]
        _, est_curvature = yaw_and_curvature_from_xy(dense)
    else:
        dense = resample_xy(path)
        est_yaw, est_curvature = yaw_and_curvature_from_xy(dense)
    yaw = est_yaw if yaw is None or np.asarray(yaw).shape[0] != dense.shape[0] else np.asarray(yaw)
    curvature = (
        est_curvature
        if curvature is None or np.asarray(curvature).shape[0] != dense.shape[0]
        else np.asarray(curvature)
    )
    torch_device = torch.device(device)
    validity = trajectory_validity_metrics(
        torch.as_tensor(dense, dtype=torch.float32, device=torch_device)[None],
        torch.as_tensor(task.cost_map, dtype=torch.float32, device=torch_device),
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=torch.as_tensor(yaw, dtype=torch.float32, device=torch_device)[None],
        analytic_curvature=torch.as_tensor(
            curvature, dtype=torch.float32, device=torch_device
        )[None],
        d_safe=SAFETY_COST_CONFIG.hard_stability_margin_meters,
        mask=torch.as_tensor(task.mask, dtype=torch.float32, device=torch_device),
        start_pose=torch.as_tensor(task.start, dtype=torch.float32, device=torch_device)[None],
        goal_pose=torch.as_tensor(task.goal, dtype=torch.float32, device=torch_device)[None],
        condition_ids=torch.zeros(1, dtype=torch.long, device=torch_device),
    )
    scalar = {}
    for key, value in validity.items():
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            scalar[key] = tensor.item()
        elif tensor.numel() == 1:
            scalar[key] = tensor.reshape(-1)[0].item()
        elif tensor.ndim == 1 and tensor.numel() <= 4:
            scalar[key] = [float(item) for item in tensor.detach().cpu()]
    scalar["path_length_m"] = path_length(dense)
    scalar["chord_length_m"] = float(np.linalg.norm(task.goal[:2] - task.start[:2]))
    return scalar


def infeasible_result(reason: str) -> Dict[str, Any]:
    return {
        "strict_valid": False,
        "forbidden_region_ok": False,
        "stability_ok": False,
        "curvature_ok": False,
        "endpoint_yaw_ok": False,
        "failure_reason": reason,
        "path_length_m": None,
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    planned = [item for item in records if item.get("found", False)]
    n = max(1, len(records))

    def _rate(key: str) -> float:
        values = [bool(item.get(key, False)) for item in records]
        return float(sum(values) / n)

    lengths = [
        float(item["path_length_m"])
        for item in planned
        if item.get("strict_valid") and item.get("path_length_m") is not None
    ]
    times = [float(item["planning_time_s"]) for item in records if "planning_time_s" in item]
    times_sorted = sorted(times)
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))] if times_sorted else None
    return {
        "count": len(records),
        "found_rate": float(len(planned) / n),
        "feasible_at_1": _rate("strict_valid"),
        "forbidden_pass": _rate("forbidden_region_ok"),
        "stability_pass": _rate("stability_ok"),
        "curvature_pass": _rate("curvature_ok"),
        "yaw_pass": _rate("endpoint_yaw_ok"),
        "mean_feasible_length_m": float(np.mean(lengths)) if lengths else None,
        "mean_planning_time_s": float(np.mean(times)) if times else None,
        "p95_planning_time_s": p95,
    }


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, float(time.perf_counter() - start)
