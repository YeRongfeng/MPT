"""Interactive final model for an accumulating, downward-facing UAV LiDAR map.

The viewer is the executable interface for the final observation contract. The
UAV follows a randomized survey with mandatory anchors around the vehicle
reference trajectory, scans from underneath, and unions each scan into an
accumulated support mask; vehicle progress is separate and is permitted only
inside known support.
``simulate_uav_observation`` is retained as the earlier oblique-sensor helper
for compatibility with its focused unit tests.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from scipy.ndimage import binary_dilation


@dataclass(frozen=True)
class SensorConfig:
    # Legacy oblique-sensor parameters are kept for the compatibility helper.
    range_m: float = 12.0
    tilt_deg: float = 35.0
    vertical_fov_deg: float = 65.0
    horizontal_fov_deg: float = 100.0
    altitude_m: float = 8.0
    point_density: float = 0.55
    support_radius_cells: int = 1
    occlusion_epsilon_m: float = 0.03
    # Parameters used by the physical preview.
    scan_radius_m: float = 3.5
    scan_step_m: float = 0.8
    survey_spacing_m: float = 1.5


def _grid_centers(
    shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = shape
    xmin, xmax, ymin, ymax = bounds
    cols = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    x = xmin + (cols[None, :] + 0.5) * float(resolution)
    y = ymin + (rows[:, None] + 0.5) * float(resolution)
    return np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width))


def _disk(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _sample_height(
    elevation: np.ndarray,
    xy: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> float:
    xmin, xmax, ymin, ymax = bounds
    col = int(np.clip((float(xy[0]) - xmin) / resolution, 0, elevation.shape[1] - 1))
    row = int(np.clip((float(xy[1]) - ymin) / resolution, 0, elevation.shape[0] - 1))
    return float(elevation[row, col])


def simulate_uav_observation(
    elevation: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    sensor_xy: Tuple[float, float],
    sensor_yaw: float,
    config: SensorConfig,
    rng: np.random.Generator,
) -> Dict[str, Union[np.ndarray, float]]:
    """Legacy oblique observation helper retained for existing tests."""
    elevation = np.asarray(elevation, dtype=np.float32)
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got {elevation.shape}")
    if not 0.0 <= config.point_density <= 1.0:
        raise ValueError("point_density must be in [0, 1]")
    grid_x, grid_y = _grid_centers(elevation.shape, bounds, resolution)
    dx = grid_x - float(sensor_xy[0])
    dy = grid_y - float(sensor_xy[1])
    horizontal_distance = np.hypot(dx, dy)
    sensor_z = _sample_height(elevation, sensor_xy, bounds, resolution) + float(config.altitude_m)
    vertical_drop = np.maximum(sensor_z - elevation, float(resolution))
    off_nadir = np.arctan2(horizontal_distance, vertical_drop)
    bearing = np.arctan2(dy, dx)
    bearing_error = np.arctan2(
        np.sin(bearing - float(sensor_yaw)), np.cos(bearing - float(sensor_yaw))
    )
    candidate = (
        (horizontal_distance <= float(config.range_m))
        & (np.abs(off_nadir - np.deg2rad(config.tilt_deg)) <= np.deg2rad(config.vertical_fov_deg) / 2.0)
        & (np.abs(bearing_error) <= np.deg2rad(config.horizontal_fov_deg) / 2.0)
    )
    rows, cols = np.nonzero(candidate)
    visible = np.zeros_like(candidate, dtype=bool)
    if len(rows):
        target_x, target_y = grid_x[rows, cols], grid_y[rows, cols]
        target_z = elevation[rows, cols]
        distance = horizontal_distance[rows, cols]
        step_count = np.maximum(2, np.ceil(distance / float(resolution)).astype(np.int32))
        max_steps = int(step_count.max())
        fractions = np.arange(1, max_steps + 1, dtype=np.float32)[None, :] / (
            step_count[:, None].astype(np.float32) + 1.0
        )
        valid_fraction = fractions < 1.0
        ray_x = float(sensor_xy[0]) + fractions * (target_x[:, None] - float(sensor_xy[0]))
        ray_y = float(sensor_xy[1]) + fractions * (target_y[:, None] - float(sensor_xy[1]))
        xmin, xmax, ymin, ymax = bounds
        ray_col = np.clip(((ray_x - xmin) / resolution).astype(np.int32), 0, elevation.shape[1] - 1)
        ray_row = np.clip(((ray_y - ymin) / resolution).astype(np.int32), 0, elevation.shape[0] - 1)
        terrain_z = elevation[ray_row, ray_col]
        ray_z = sensor_z + fractions * (target_z[:, None] - sensor_z)
        blocked = np.any(valid_fraction & (terrain_z > ray_z + config.occlusion_epsilon_m), axis=1)
        visible[rows, cols] = ~blocked
    returns = visible & (rng.random(elevation.shape) <= float(config.point_density))
    support = binary_dilation(returns, structure=_disk(config.support_radius_cells)) & candidate
    return {
        "candidate": candidate,
        "visible": visible,
        "returns": returns,
        "support": support,
        "sensor_z": float(sensor_z),
        "coverage": float(support.mean()),
        "return_count": float(returns.sum()),
    }


def simulate_downward_lidar_scan(
    elevation: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    sensor_xy: Tuple[float, float],
    config: SensorConfig,
    rng: np.random.Generator,
) -> Dict[str, Union[np.ndarray, float]]:
    """Simulate one nadir LiDAR footprint and its sparse returns.

    A downward-mounted scanner has no forward fan: its instantaneous footprint
    is a disk on the ground.  ``point_density`` models incomplete returns and
    support dilation models the usable local map around observed points.
    """
    elevation = np.asarray(elevation, dtype=np.float32)
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got {elevation.shape}")
    if not 0.0 <= config.point_density <= 1.0:
        raise ValueError("point_density must be in [0, 1]")
    if config.scan_radius_m <= 0.0:
        raise ValueError("scan_radius_m must be positive")
    grid_x, grid_y = _grid_centers(elevation.shape, bounds, resolution)
    footprint = np.hypot(grid_x - float(sensor_xy[0]), grid_y - float(sensor_xy[1])) <= float(config.scan_radius_m)
    sensor_z = _sample_height(elevation, sensor_xy, bounds, resolution) + float(config.altitude_m)
    returns = footprint & (rng.random(elevation.shape) <= float(config.point_density))
    support = binary_dilation(returns, structure=_disk(config.support_radius_cells)) & footprint
    return {
        "candidate": footprint,
        "returns": returns,
        "support": support,
        "sensor_z": float(sensor_z),
        "coverage": float(support.mean()),
        "return_count": float(returns.sum()),
    }


def load_environment(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    with open(Path(path) / "map.p", "rb") as handle:
        payload = pickle.load(handle)
    tensor = np.asarray(payload["tensor"], dtype=np.float32)
    bounds = tuple(float(value) for value in payload.get("bounds", (-10.0, 10.0, -10.0, 10.0)))
    resolution = float(payload.get("resolution", (bounds[1] - bounds[0]) / tensor.shape[1]))
    return tensor[:, :, 0], bounds, resolution


def load_reference_path(path: Path) -> np.ndarray:
    """Load the vehicle reference route from ``path_0.p``."""
    with open(Path(path) / "path_0.p", "rb") as handle:
        payload = pickle.load(handle)
    route = np.asarray(payload["path"], dtype=np.float32)
    if route.ndim != 2 or route.shape[1] < 2 or route.shape[0] < 2:
        raise ValueError(f"path_0.p must contain an (N, >=2) path, got {route.shape}")
    return route[:, :2]


def _path_position_at_distance(path_xy: np.ndarray, distance: float) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    return np.asarray(
        [np.interp(distance, cumulative, path_xy[:, axis]) for axis in range(2)], dtype=np.float32
    )


def _densify_polyline(route: np.ndarray, max_step_m: float) -> np.ndarray:
    """Insert linear samples so the rendered/scanned UAV route has no gaps."""
    route = np.asarray(route, dtype=np.float32)
    if route.ndim != 2 or route.shape[0] < 2:
        raise ValueError("route must contain at least two points")
    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive")
    points = [route[0].copy()]
    for start, target in zip(route[:-1], route[1:]):
        count = max(1, int(np.ceil(float(np.linalg.norm(target - start)) / max_step_m)))
        for step in range(1, count + 1):
            points.append(start + (step / float(count)) * (target - start))
    return np.asarray(points, dtype=np.float32)


def build_uav_survey_path(
    bounds: Tuple[float, float, float, float],
    vehicle_path: np.ndarray,
    rng: np.random.Generator,
    spacing_m: float = 2.5,
    corridor_radius_m: float = 3.5,
) -> np.ndarray:
    """Build a randomized survey route with guaranteed path-corridor anchors.

    Random waypoints explore the map, while jittered anchors sampled along the
    vehicle reference path ensure that the UAV footprint includes that path's
    neighbourhood. The anchors are intentionally perturbed and interleaved
    with local/global excursions, so this is not a copy of the car trajectory.
    """
    xmin, xmax, ymin, ymax = bounds
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if corridor_radius_m <= 0.0:
        raise ValueError("corridor_radius_m must be positive")
    vehicle_path = np.asarray(vehicle_path, dtype=np.float32)
    if vehicle_path.ndim != 2 or vehicle_path.shape[0] < 2 or vehicle_path.shape[1] < 2:
        raise ValueError("vehicle_path must have shape (N, >=2)")
    path_xy = vehicle_path[:, :2]
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    anchor_distances = np.arange(0.0, cumulative[-1] + 0.5 * spacing_m, spacing_m)
    anchors = np.asarray(
        [_path_position_at_distance(path_xy, distance) for distance in anchor_distances],
        dtype=np.float32,
    )

    # Jitter is bounded well inside the LiDAR footprint, preserving coverage
    # of the original route while making the actual flight line irregular.
    jitter_limit = min(0.35 * float(corridor_radius_m), 0.75)
    anchors += rng.uniform(-jitter_limit, jitter_limit, size=anchors.shape).astype(np.float32)
    anchors[:, 0] = np.clip(anchors[:, 0], xmin, xmax)
    anchors[:, 1] = np.clip(anchors[:, 1], ymin, ymax)

    random_count = max(12, int(np.ceil(1.5 * len(anchors))))
    margin = min(0.5 * float(corridor_radius_m), 1.0)
    random_points = np.column_stack(
        [
            rng.uniform(xmin + margin, xmax - margin, size=random_count),
            rng.uniform(ymin + margin, ymax - margin, size=random_count),
        ]
    ).astype(np.float32)

    def append_continuous_segment(route_points, target):
        start = np.asarray(route_points[-1], dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        segment_length = float(np.linalg.norm(target - start))
        segment_count = max(1, int(np.ceil(segment_length / spacing_m)))
        for step in range(1, segment_count + 1):
            fraction = step / float(segment_count)
            point = start + fraction * (target - start)
            if step < segment_count:
                point += rng.normal(0.0, 0.12 * spacing_m, size=2).astype(np.float32)
                point[0] = np.clip(point[0], xmin, xmax)
                point[1] = np.clip(point[1], ymin, ymax)
            route_points.append(point.tolist())

    # Follow the anchor corridor in order, but insert random local excursions
    # and bounded interpolation so the flight is irregular without jumps.
    waypoints = [anchors[0].tolist()]
    random_order = rng.permutation(len(random_points))
    random_cursor = 0
    for anchor_index, anchor in enumerate(anchors[1:], start=1):
        if random_cursor < len(random_order) and (anchor_index == 1 or rng.random() < 0.55):
            append_continuous_segment(waypoints, random_points[random_order[random_cursor]])
            random_cursor += 1
        if rng.random() < 0.65:
            local = anchor + rng.normal(0.0, 0.8 * float(corridor_radius_m), size=2).astype(np.float32)
            local[0] = np.clip(local[0], xmin, xmax)
            local[1] = np.clip(local[1], ymin, ymax)
            append_continuous_segment(waypoints, local)
        append_continuous_segment(waypoints, anchor)
    for index in random_order[random_cursor:]:
        append_continuous_segment(waypoints, random_points[index])
    route = np.asarray(waypoints, dtype=np.float32)
    if route.ndim != 2 or route.shape[0] < 2:
        raise ValueError("could not construct a UAV survey route inside map bounds")
    return _densify_polyline(route, max_step_m=0.75 * float(spacing_m))


def _mask_contains_xy(
    mask: np.ndarray,
    xy: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    resolution: float,
) -> bool:
    xmin, xmax, ymin, ymax = bounds
    col = int(np.floor((float(xy[0]) - xmin) / resolution))
    row = int(np.floor((float(xy[1]) - ymin) / resolution))
    if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1]:
        return False
    return bool(mask[row, col])


class UAVObservationViewer:
    def __init__(
        self,
        environments: List[Path],
        *,
        seed: int,
        config: SensorConfig,
        output_dir: Path,
    ) -> None:
        self.environments = environments
        self.environment_index = 0
        self.rng = np.random.default_rng(int(seed))
        self.config = config
        self.output_dir = Path(output_dir)
        self._load_current_environment()

        self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 5.4))
        self.figure.subplots_adjust(bottom=0.22, wspace=0.18)
        self._buttons = []
        self._add_button("Next scan", 0.025, self._next_scan)
        self._add_button("Advance car", 0.115, self._advance_vehicle)
        self._add_button("Reset", 0.205, self._reset_scans)
        self._add_button("Final area", 0.295, self._scan_to_end)
        self._add_button("Radius +", 0.385, lambda _: self._change_config(scan_radius_m=0.5))
        self._add_button("Radius -", 0.475, lambda _: self._change_config(scan_radius_m=-0.5))
        self._add_button("Density +", 0.565, lambda _: self._change_config(point_density=0.1))
        self._add_button("Density -", 0.655, lambda _: self._change_config(point_density=-0.1))
        self._add_button("Next env", 0.745, self._next_environment)
        self._add_button("Save", 0.835, self._save)
        self._draw(self.current_result)

    def _add_button(self, label: str, left: float, callback) -> None:
        axis = self.figure.add_axes([left, 0.07, 0.07, 0.07])
        button = Button(axis, label)
        button.on_clicked(callback)
        self._buttons.append(button)

    def _load_current_environment(self) -> None:
        env_path = self.environments[self.environment_index]
        self.elevation, self.bounds, self.resolution = load_environment(env_path)
        self.vehicle_path = load_reference_path(env_path)
        self.path_distance = np.linalg.norm(np.diff(self.vehicle_path, axis=0), axis=1)
        self.path_cumulative = np.concatenate(([0.0], np.cumsum(self.path_distance)))
        self.uav_path = build_uav_survey_path(
            self.bounds,
            self.vehicle_path,
            self.rng,
            spacing_m=self.config.survey_spacing_m,
            corridor_radius_m=self.config.scan_radius_m,
        )
        self.scan_index = 0
        self.vehicle_distance_m = 0.0
        self.uav_index = 0
        self.vehicle_xy = self.vehicle_path[0].copy()
        self.uav_xy = self.uav_path[self.uav_index].copy()
        self.uav_trail = [self.uav_xy.copy()]
        self.accumulated_returns = np.zeros_like(self.elevation, dtype=bool)
        self.accumulated_support = np.zeros_like(self.elevation, dtype=bool)
        # The vehicle starts in an already mapped local patch; subsequent
        # scans extend that known map ahead of it.
        initial_map = simulate_downward_lidar_scan(
            self.elevation, self.bounds, self.resolution, tuple(self.vehicle_xy), self.config, self.rng
        )
        self.accumulated_returns |= initial_map["returns"]
        self.accumulated_support |= initial_map["support"]
        self.current_result = simulate_downward_lidar_scan(
            self.elevation, self.bounds, self.resolution, tuple(self.uav_xy), self.config, self.rng
        )

    def _next_scan(self, _event) -> None:
        self.uav_index = min(self.uav_index + 1, len(self.uav_path) - 1)
        self.uav_xy = self.uav_path[self.uav_index].copy()
        self.uav_trail.append(self.uav_xy.copy())
        self.current_result = simulate_downward_lidar_scan(
            self.elevation, self.bounds, self.resolution, tuple(self.uav_xy), self.config, self.rng
        )
        self.accumulated_returns |= self.current_result["returns"]
        self.accumulated_support |= self.current_result["support"]
        self.scan_index += 1
        self._draw(self.current_result)

    def _scan_to_end(self, _event) -> None:
        """Complete the final UAV survey in one interaction."""
        while self.uav_index < len(self.uav_path) - 1:
            self.uav_index += 1
            self.uav_xy = self.uav_path[self.uav_index].copy()
            self.uav_trail.append(self.uav_xy.copy())
            self.current_result = simulate_downward_lidar_scan(
                self.elevation,
                self.bounds,
                self.resolution,
                tuple(self.uav_xy),
                self.config,
                self.rng,
            )
            self.accumulated_returns |= self.current_result["returns"]
            self.accumulated_support |= self.current_result["support"]
            self.scan_index += 1
        self._draw(self.current_result)

    def _advance_vehicle(self, _event) -> None:
        """Move the vehicle only when the next route cell is already mapped."""
        candidate_distance = min(
            self.vehicle_distance_m + self.config.scan_step_m, float(self.path_cumulative[-1])
        )
        candidate_xy = _path_position_at_distance(self.vehicle_path, candidate_distance)
        if not _mask_contains_xy(self.accumulated_support, tuple(candidate_xy), self.bounds, self.resolution):
            print("vehicle held: candidate position is outside the accumulated UAV map")
            return
        self.vehicle_distance_m = candidate_distance
        self.vehicle_xy = candidate_xy
        self._draw(self.current_result)

    def _reset_scans(self, _event) -> None:
        self.scan_index = 0
        self.vehicle_distance_m = 0.0
        self.uav_index = 0
        self.vehicle_xy = self.vehicle_path[0].copy()
        self.uav_xy = self.uav_path[self.uav_index].copy()
        self.uav_trail = [self.uav_xy.copy()]
        self.accumulated_returns.fill(False)
        self.accumulated_support.fill(False)
        initial_map = simulate_downward_lidar_scan(
            self.elevation, self.bounds, self.resolution, tuple(self.vehicle_xy), self.config, self.rng
        )
        self.accumulated_returns |= initial_map["returns"]
        self.accumulated_support |= initial_map["support"]
        self.current_result = simulate_downward_lidar_scan(
            self.elevation, self.bounds, self.resolution, tuple(self.uav_xy), self.config, self.rng
        )
        self._draw(self.current_result)

    def _change_config(self, **changes: float) -> None:
        values = {key: getattr(self.config, key) for key in changes}
        for key, delta in changes.items():
            value = values[key] + float(delta)
            if key == "point_density":
                value = float(np.clip(value, 0.05, 1.0))
            elif key == "scan_radius_m":
                value = float(np.clip(value, 1.0, 8.0))
            values[key] = value
        self.config = replace(self.config, **values)
        self._reset_scans(None)

    def _next_environment(self, _event) -> None:
        self.environment_index = (self.environment_index + 1) % len(self.environments)
        self._load_current_environment()
        self._draw(self.current_result)

    def _draw(self, result: Dict[str, Union[np.ndarray, float]]) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        env_name = self.environments[self.environment_index].name
        for axis in self.axes:
            axis.clear()
            axis.set_aspect("equal")
        self.axes[0].imshow(self.elevation, origin="lower", extent=self.bounds, cmap="terrain")
        if self.accumulated_support.any():
            self.axes[0].imshow(
                np.ma.masked_where(~self.accumulated_support, self.accumulated_support),
                origin="lower", extent=self.bounds, cmap="Blues", alpha=0.42,
            )
        self.axes[0].plot(self.vehicle_path[:, 0], self.vehicle_path[:, 1], color="crimson", lw=1.0, alpha=0.55, label="reference corridor")
        self.axes[0].plot(self.uav_path[:, 0], self.uav_path[:, 1], color="deepskyblue", lw=0.8, alpha=0.75, label="UAV survey")
        trail = np.asarray(self.uav_trail)
        if len(trail) > 1:
            self.axes[0].plot(trail[:, 0], trail[:, 1], "--", color="black", lw=1.0, alpha=0.75)
        self.axes[0].scatter([self.vehicle_xy[0]], [self.vehicle_xy[1]], c="crimson", s=30, marker="o", label="vehicle")
        self.axes[0].scatter([self.uav_xy[0]], [self.uav_xy[1]], c="black", s=38, marker="^", label="UAV")
        self.axes[0].add_patch(Circle(tuple(self.uav_xy), self.config.scan_radius_m, fill=False, color="black", lw=1.0))
        self.axes[0].set_title(f"Terrain + accumulated support ({env_name})")
        self.axes[0].legend(loc="lower left", fontsize=8)

        self.axes[1].imshow(result["returns"], origin="lower", extent=self.bounds, cmap="gray_r", vmin=0, vmax=1)
        self.axes[1].scatter([self.uav_xy[0]], [self.uav_xy[1]], c="red", s=25, marker="^")
        self.axes[1].add_patch(Circle(tuple(self.uav_xy), self.config.scan_radius_m, fill=False, color="red", lw=1.0))
        self.axes[1].set_title(f"Current downward returns ({int(result['return_count'])})")

        self.axes[2].imshow(self.accumulated_support, origin="lower", extent=self.bounds, cmap="gray_r", vmin=0, vmax=1)
        self.axes[2].set_title(f"Accumulated support ({100.0 * self.accumulated_support.mean():.1f}% coverage)")
        for axis in self.axes:
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(ymin, ymax)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
        self.figure.suptitle(
            f"scan={self.scan_index}  vehicle={self.vehicle_distance_m:.1f} m  "
            f"UAV waypoint={self.uav_index}/{len(self.uav_path) - 1}  "
            f"radius={self.config.scan_radius_m:.1f} m  density={self.config.point_density:.2f}  "
            f"UAV z={result['sensor_z']:.2f} m"
        )
        self.figure.canvas.draw_idle()

    def _save(self, _event) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.environments[self.environment_index].name}_step{self.scan_index:03d}_{self.rng.integers(0, 1_000_000):06d}"
        self.figure.savefig(self.output_dir / f"{stem}.png", dpi=160)
        np.savez_compressed(
            self.output_dir / f"{stem}.npz",
            current_returns=self.current_result["returns"],
            current_support=self.current_result["support"],
            accumulated_returns=self.accumulated_returns,
            accumulated_support=self.accumulated_support,
            candidate=self.current_result["candidate"],
            vehicle_path=self.vehicle_path,
            vehicle_xy=self.vehicle_xy,
            uav_xy=self.uav_xy,
            uav_trail=np.asarray(self.uav_trail),
            vehicle_distance_m=np.asarray(self.vehicle_distance_m),
            uav_path=self.uav_path,
            uav_index=np.asarray(self.uav_index),
            scan_index=np.asarray(self.scan_index),
        )
        print(f"saved {self.output_dir / stem}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/dataset1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--environment", default=None)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--scan-radius-m", type=float, default=3.5)
    parser.add_argument("--survey-spacing-m", type=float, default=1.5)
    parser.add_argument("--point-density", type=float, default=0.55)
    parser.add_argument("--output-dir", default="/tmp/uav_observation_preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.dataset_root) / args.split
    if args.environment:
        environments = [split_dir / args.environment]
    else:
        environments = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not environments:
        raise FileNotFoundError(f"no environments found under {split_dir}")
    config = SensorConfig(
        scan_radius_m=float(args.scan_radius_m),
        survey_spacing_m=float(args.survey_spacing_m),
        point_density=float(args.point_density),
    )
    viewer = UAVObservationViewer(environments, seed=int(args.seed), config=config, output_dir=Path(args.output_dir))
    print("Final UAV model: Next scan advances only the UAV map; Advance car moves only inside accumulated support; Reset clears history.")
    plt.show()
    del viewer


if __name__ == "__main__":
    main()
