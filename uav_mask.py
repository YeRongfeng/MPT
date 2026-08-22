"""Physical-looking UAV observation masks for map-conditioned planning.

The returned mask has one meaning: cells that are both observed by the UAV
and not occupied by a detected local obstacle.  Sparse point-return dropout
is deliberately not used to carve holes in the observation footprint; it is a
rendering detail, not a change in the geometric map support.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np


def _grid_centers(
    shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    xmin, _, ymin, _ = bounds
    cols = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    grid_x = xmin + (cols[None, :] + 0.5) * float(resolution)
    grid_y = ymin + (rows[:, None] + 0.5) * float(resolution)
    return np.broadcast_to(grid_x, (height, width)), np.broadcast_to(grid_y, (height, width))


def _resample_path(path_xy: np.ndarray, spacing_m: float) -> np.ndarray:
    path = np.asarray(path_xy, dtype=np.float32)[..., :2]
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] <= 1e-6:
        return path[:1].copy()
    distances = np.arange(0.0, cumulative[-1] + 0.5 * spacing_m, spacing_m)
    distances = np.clip(distances, 0.0, cumulative[-1])
    return np.stack(
        [np.interp(distances, cumulative, path[:, axis]) for axis in range(2)],
        axis=1,
    ).astype(np.float32)


def generate_uav_observation_mask(
    shape: Tuple[int, int],
    trajectory_xy: np.ndarray,
    seed: int,
    *,
    bounds: Tuple[float, float, float, float] = (-10.0, 10.0, -10.0, 10.0),
    resolution: float = 0.2,
    p_mask: float = 1.0,
    scan_radius_m: float = 2.5,
    corridor_spacing_m: float = 3.0,
    random_scan_count: int = 7,
    vehicle_radius_m: float = 0.35,
    require_trajectory_clear: bool = True,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Generate a deterministic accumulated UAV observation mask.

    The UAV scan centers are a union of perturbed anchors around the vehicle
    reference route and independent map-wide probes.  Each center contributes
    a nadir circular footprint.  Small circular obstacle footprints are
    sampled only inside the observed region; in Stage 1 they are kept away
    from the reference route so the demonstration remains feasible.
    """
    shape = tuple(int(value) for value in shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"shape must be a positive (H,W), got {shape}")
    path = np.asarray(trajectory_xy, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2:
        raise ValueError(f"trajectory_xy must be (N,>=2), got {path.shape}")
    if not 0.0 <= float(p_mask) <= 1.0:
        raise ValueError("p_mask must be in [0,1]")
    if float(scan_radius_m) <= 0.0 or float(corridor_spacing_m) <= 0.0:
        raise ValueError("scan_radius_m and corridor_spacing_m must be positive")
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    if xmax <= xmin or ymax <= ymin or float(resolution) <= 0.0:
        raise ValueError("invalid bounds or resolution")

    rng = np.random.default_rng(int(seed))
    if not bool(rng.binomial(1, float(p_mask))):
        full = np.ones(shape, dtype=np.float32)
        metadata = {
            "mask_active": False,
            "semantic_mode": "complete",
            "mask_generation_semantics": "uav_footprint_observation_circle_obstacles_v1",
            "accepted_type": "complete",
            "sampling_attempts": 1,
            "attempts": 1,
            "resamples": 0,
            "proposed_type_counts": {"complete": 1},
            "rejected_type_counts": {},
            "rejection_reason_counts": {},
            "raw_connectivity_counts": {"connected": 1, "disconnected": 0},
            "trajectory_obstacle_mode": "none",
            "demo_blocked_fraction": 0.0,
            "demo_max_contiguous_blocked_fraction": 0.0,
            "masked_fraction": 0.0,
            "observed_fraction": 1.0,
            "obstacle_fraction": 0.0,
            "scan_center_count": 0,
        }
        return (full, metadata) if return_metadata else full

    grid_x, grid_y = _grid_centers(shape, bounds, resolution)
    anchors = _resample_path(path, float(corridor_spacing_m))
    # Keep the UAV survey independent from the car while retaining a guaranteed
    # corridor: perturb anchors by less than one fifth of the footprint radius.
    jitter_limit = min(0.20 * float(scan_radius_m), 0.5)
    anchors = anchors + rng.uniform(-jitter_limit, jitter_limit, anchors.shape).astype(np.float32)
    anchors[:, 0] = np.clip(anchors[:, 0], xmin, xmax)
    anchors[:, 1] = np.clip(anchors[:, 1], ymin, ymax)

    margin = min(float(scan_radius_m), 1.0)
    random_centers = np.column_stack(
        [
            rng.uniform(xmin + margin, xmax - margin, int(random_scan_count)),
            rng.uniform(ymin + margin, ymax - margin, int(random_scan_count)),
        ]
    ).astype(np.float32)

    # Connect every random excursion with bounded interpolation.  The mask is
    # built from this continuous survey route, rather than isolated disks, so
    # adjacent scans cannot leave visual or geometric breaks.
    survey_step = min(0.80 * float(scan_radius_m), float(corridor_spacing_m))
    survey_points = [anchors[0].copy()]

    def append_segment(target: np.ndarray) -> None:
        start = np.asarray(survey_points[-1], dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        count = max(1, int(np.ceil(float(np.linalg.norm(target - start)) / survey_step)))
        for step in range(1, count + 1):
            fraction = step / float(count)
            survey_points.append((start + fraction * (target - start)).astype(np.float32))

    random_order = rng.permutation(len(random_centers))
    random_cursor = 0
    for anchor_index, anchor in enumerate(anchors[1:], start=1):
        if random_cursor < len(random_order) and (anchor_index == 1 or rng.random() < 0.55):
            append_segment(random_centers[random_order[random_cursor]])
            random_cursor += 1
        if rng.random() < 0.65:
            local = anchor + rng.normal(
                0.0, 0.35 * float(scan_radius_m), size=2
            ).astype(np.float32)
            local[0] = np.clip(local[0], xmin, xmax)
            local[1] = np.clip(local[1], ymin, ymax)
            append_segment(local)
        append_segment(anchor)
    for random_index in random_order[random_cursor:]:
        append_segment(random_centers[random_index])
    centers = np.asarray(survey_points, dtype=np.float32)

    def disk(center: np.ndarray, radius_m: float) -> np.ndarray:
        return np.hypot(grid_x - float(center[0]), grid_y - float(center[1])) <= float(radius_m)

    def disks(centers_to_render: np.ndarray, radius_m: float) -> np.ndarray:
        centers_to_render = np.asarray(centers_to_render, dtype=np.float32)
        dx = grid_x[..., None] - centers_to_render[None, None, :, 0]
        dy = grid_y[..., None] - centers_to_render[None, None, :, 1]
        return np.any(dx * dx + dy * dy <= float(radius_m) ** 2, axis=2)

    observed = disks(centers, float(scan_radius_m))

    # The reference route is used only as a coverage constraint, not as the
    # UAV route.  It is sampled densely for obstacle clearance checks.
    dense_distances = np.linspace(
        0.0,
        float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        max(200, path.shape[0] * 8),
    )
    dense = np.stack(
        [
            np.interp(
                dense_distances,
                np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))),
                path[:, axis],
            )
            for axis in range(2)
        ],
        axis=1,
    )
    obstacle = np.zeros(shape, dtype=bool)
    obstacle_count = int(rng.integers(2, 5))
    path_for_distance = dense[:: max(1, len(dense) // 200)]
    for _ in range(obstacle_count):
        for _attempt in range(64):
            center = np.asarray(
                [rng.uniform(xmin + 0.5, xmax - 0.5), rng.uniform(ymin + 0.5, ymax - 0.5)],
                dtype=np.float32,
            )
            radius = float(rng.uniform(0.25, 0.55))
            if not bool(np.any(disk(center, radius))):
                continue
            if require_trajectory_clear:
                if float(np.min(np.linalg.norm(path_for_distance - center[None, :], axis=1))) < (
                    radius + float(vehicle_radius_m) + 0.35
                ):
                    continue
            candidate = disk(center, radius)
            obstacle |= candidate & observed
            break

    mask = (observed & ~obstacle).astype(np.float32)
    metadata = {
        "mask_active": True,
        "semantic_mode": "uav_observation_with_local_obstacles",
        "mask_generation_semantics": "uav_footprint_observation_circle_obstacles_v1",
        "accepted_type": "uav_observation",
        "sampling_attempts": 1,
        "attempts": 1,
        "resamples": 0,
        "proposed_type_counts": {"uav_observation": 1},
        "rejected_type_counts": {},
        "rejection_reason_counts": {},
        "raw_connectivity_counts": {"connected": 1, "disconnected": 0},
        "trajectory_obstacle_mode": (
            "near_route_nonblocking" if require_trajectory_clear else "on_route_allowed"
        ),
        "demo_blocked_fraction": 0.0,
        "demo_max_contiguous_blocked_fraction": 0.0,
        "masked_fraction": float(1.0 - mask.mean()),
        "observed_fraction": float(observed.mean()),
        "obstacle_fraction": float(obstacle.mean()),
        "scan_center_count": int(len(centers)),
        "corridor_anchor_count": int(len(anchors)),
        "random_scan_count": int(len(random_centers)),
        "survey_path_count": int(len(centers)),
        "survey_max_step_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).max()),
        "scan_radius_m": float(scan_radius_m),
        "obstacle_count": int(obstacle_count),
    }
    return (mask, metadata) if return_metadata else mask
