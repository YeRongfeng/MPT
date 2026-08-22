"""Call official T-Hybrid A* over ROS on a Path MeanFlow task.

The paper baseline is the upstream ROS node (planner, smoother, path
topics), not the diagnostic CLI. Occupancy is published as OccupancyGrid;
roll/pitch/roughness still come from elevation/normals as T-Hybrid voxels.
The shared evaluator scores the resulting path with our hard checks.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from baselines.common import grid_to_world
from map_config import MAP_CONFIG, MAP_RESOLUTION, SAFETY_COST_CONFIG

REPO = Path(__file__).resolve().parents[1]
CATKIN_WS = REPO / "baselines" / "t_hybrid_ws"
DEVEL_SETUP = CATKIN_WS / "devel" / "setup.bash"
RUNTIME_DIR = CATKIN_WS / "runtime"
CLIENT = REPO / "baselines" / "t_hybrid_ros" / "plan_once.py"
ROS_PYTHON = "/usr/bin/python3"
NOETIC_SETUP = "/opt/ros/noetic/setup.bash"

_LAUNCH_PROC: Optional[subprocess.Popen] = None

# Matches baselines/t_hybrid_cli/include/constants.h
CELL_SIZE = float(MAP_RESOLUTION)
MAP_LENGTH = float(MAP_CONFIG.size_meters)


def world_to_thybrid(x: float, y: float) -> Tuple[float, float]:
    origin_x, origin_y = MAP_CONFIG.origin_xy
    return float(x) - float(origin_x), float(y) - float(origin_y)


def thybrid_to_world(x: float, y: float) -> Tuple[float, float]:
    origin_x, origin_y = MAP_CONFIG.origin_xy
    return float(x) + float(origin_x), float(y) + float(origin_y)


def _voxel_id(col: int, row: int, volnum: int) -> int:
    return int(col + row * volnum)


def _roughness(elevation: np.ndarray, normals: np.ndarray, row: int, col: int) -> float:
    height, width = elevation.shape
    normal = np.asarray(normals[row, col], dtype=np.float64)
    if not np.all(np.isfinite(normal)) or float(np.linalg.norm(normal)) < 1e-6:
        return 1.0
    x0, y0 = grid_to_world(row, col)
    z0 = float(elevation[row, col])
    distances = []
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            rr = row + drow
            cc = col + dcol
            if rr < 0 or cc < 0 or rr >= height or cc >= width:
                continue
            x, y = grid_to_world(rr, cc)
            z = float(elevation[rr, cc])
            if not np.isfinite(z):
                continue
            distances.append(
                abs(
                    normal[0] * (x - x0)
                    + normal[1] * (y - y0)
                    + normal[2] * (z - z0)
                )
            )
    if not distances:
        return 1.0
    return float(np.mean(distances))


def write_occupancy(path: Path, occupancy: np.ndarray) -> None:
    occupied = np.asarray(occupancy, dtype=bool)
    height, width = occupied.shape
    lines = [f"{width} {height}"]
    for row in range(height):
        lines.append(" ".join("1" if occupied[row, col] else "0" for col in range(width)))
    path.write_text("\n".join(lines) + "\n")


def write_terrain(
    path: Path,
    occupancy: np.ndarray,
    elevation: np.ndarray,
    normals: np.ndarray,
) -> None:
    occupied = np.asarray(occupancy, dtype=bool)
    height, width = occupied.shape
    volnum = int(round(MAP_LENGTH / CELL_SIZE))
    lines = []
    for row in range(height):
        for col in range(width):
            voxel_id = _voxel_id(col, row, volnum)
            if occupied[row, col] or not np.isfinite(elevation[row, col]):
                lines.append(f"{voxel_id} 0 0 0 0 1")
                continue
            normal = np.asarray(normals[row, col], dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if not np.all(np.isfinite(normal)) or norm < 1e-6:
                lines.append(f"{voxel_id} 0 0 0 0 1")
                continue
            normal = normal / norm
            rough = _roughness(elevation, normals, row, col)
            lines.append(
                f"{voxel_id} 1 {rough:.8g} {normal[0]:.8g} {normal[1]:.8g} {normal[2]:.8g}"
            )
    path.write_text("\n".join(lines) + "\n")


def _parse_path(stdout: str) -> np.ndarray:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    found_index = next((i for i, line in enumerate(lines) if line.startswith("found")), None)
    if found_index is None:
        return np.zeros((0, 3), dtype=np.float32)
    count_parts = lines[found_index].split()
    if len(count_parts) >= 2 and count_parts[1] == "0":
        return np.zeros((0, 3), dtype=np.float32)
    points = []
    for line in lines[found_index + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x_t, y_t, yaw = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        x, y = thybrid_to_world(x_t, y_t)
        points.append((x, y, yaw))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _ros_command(command: str) -> str:
    parts = [f"source '{NOETIC_SETUP}'"]
    if DEVEL_SETUP.is_file():
        parts.append(f"source '{DEVEL_SETUP}'")
    parts.append(command)
    return "; ".join(parts)


def _run_ros(command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", _ros_command(command)],
        **kwargs,
    )


def _node_running() -> bool:
    proc = _run_ros("rosnode list", capture_output=True, text=True, timeout=10)
    return proc.returncode == 0 and "Thybrid_astar" in (proc.stdout or "")


def _ensure_ros_node(terrain_path: Path) -> Optional[str]:
    global _LAUNCH_PROC
    if not DEVEL_SETUP.is_file():
        return "t_hybrid_ros_not_built"
    if _node_running():
        return None
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME_DIR / "roslaunch.log"
    launch = (
        "roslaunch Thybrid_astar path_plan_eval.launch "
        f"terrain_data:='{terrain_path}' use_rviz:=false"
    )
    with open(log_path, "ab") as log:
        _LAUNCH_PROC = subprocess.Popen(
            ["bash", "-lc", _ros_command(launch)],
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=os.environ.copy(),
        )
    deadline = time.time() + 40.0
    while time.time() < deadline:
        if _node_running():
            return None
        if _LAUNCH_PROC.poll() is not None:
            return f"t_hybrid_roslaunch_exit_{_LAUNCH_PROC.returncode}"
        time.sleep(0.5)
    return "t_hybrid_ros_node_timeout"


def plan_t_hybrid(task, *, timeout_s: float = 60.0) -> Dict[str, Any]:
    info = "ros_occupancy_and_terrain_voxels"
    if task.occupancy is None or task.elevation is None or task.normals is None:
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": "missing_map",
            "input": info,
        }
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    occ_path = RUNTIME_DIR / "occupancy.txt"
    terrain_path = RUNTIME_DIR / "terrainData.txt"
    write_occupancy(occ_path, task.occupancy)
    write_terrain(terrain_path, task.occupancy, task.elevation, task.normals)
    boot = _ensure_ros_node(terrain_path)
    if boot:
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": boot,
            "input": info,
        }
    start_t = world_to_thybrid(float(task.start[0]), float(task.start[1]))
    goal_t = world_to_thybrid(float(task.goal[0]), float(task.goal[1]))
    client = (
        f"'{ROS_PYTHON}' '{CLIENT}' '{occ_path}' "
        f"{start_t[0]} {start_t[1]} {float(task.start[2])} "
        f"{goal_t[0]} {goal_t[1]} {float(task.goal[2])} "
        f"--resolution {CELL_SIZE} --timeout {timeout_s}"
    )
    try:
        proc = _run_ros(
            client,
            capture_output=True,
            text=True,
            timeout=timeout_s + 15.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": "timeout",
            "input": info,
        }
    path = _parse_path(proc.stdout)
    if proc.returncode != 0 or path.shape[0] < 2:
        reason = "no_path"
        if proc.returncode != 0:
            reason = f"t_hybrid_exit_{proc.returncode}"
        if "timeout" in (proc.stderr or ""):
            reason = "timeout"
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": reason,
            "input": info,
            "stderr": (proc.stderr or "")[-500:],
        }
    return {
        "found": True,
        "path": path,
        "expansions": int(path.shape[0]),
        "failure_reason": None,
        "input": info,
        "stderr": (proc.stderr or "")[-300:],
        "vehicle_radius_m": SAFETY_COST_CONFIG.vehicle_radius_meters,
    }
