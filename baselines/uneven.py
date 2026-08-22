"""Run the original Uneven Planner chain on one shared evaluation task."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from baselines.export_elevation_pcd import elevation_to_xyz, write_pcd


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "baselines" / "uneven_planner_ws"
RUNTIME = REPO / "baselines" / "evaluation_results" / "uneven_runtime"
LAUNCH = "uneven_baseline_cli plan_once.launch"


def _runtime_inputs(task) -> tuple[Path, Path]:
    task_dir = RUNTIME / task.environment
    task_dir.mkdir(parents=True, exist_ok=True)
    pcd_path = task_dir / "elevation.pcd"
    map_path = task_dir / "empty.map"
    if not pcd_path.is_file():
        write_pcd(pcd_path, elevation_to_xyz(task.elevation))
    if not map_path.is_file():
        map_path.write_text("")
    return pcd_path, map_path


def _parse_path(stdout: str) -> np.ndarray | None:
    points = []
    for line in stdout.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0] == "PATH":
            try:
                points.append(tuple(float(value) for value in fields[1:]))
            except ValueError:
                continue
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float32)


def plan_uneven(task, *, timeout_s: float = 180.0) -> Dict[str, Any]:
    pcd_path, map_path = _runtime_inputs(task)
    start = np.asarray(task.start, dtype=np.float64)
    goal = np.asarray(task.goal, dtype=np.float64)
    command = [
        "roslaunch",
        "uneven_baseline_cli",
        "plan_once.launch",
        f"pcd:={pcd_path}",
        f"map_file:={map_path}",
        f"start_x:={start[0]}",
        f"start_y:={start[1]}",
        f"start_yaw:={start[2]}",
        f"goal_x:={goal[0]}",
        f"goal_y:={goal[1]}",
        f"goal_yaw:={goal[2]}",
        "samples:=200",
    ]
    env = os.environ.copy()
    env.setdefault("ROS_IP", "127.0.0.1")
    env.setdefault("ROS_HOSTNAME", "localhost")
    env.setdefault("ROS_HOME", str(REPO / "baselines" / "ros_home"))
    env["CMAKE_PREFIX_PATH"] = f"{WORKSPACE / 'devel'}:{env.get('CMAKE_PREFIX_PATH', '')}"
    env["ROS_PACKAGE_PATH"] = f"{WORKSPACE / 'src'}:{env.get('ROS_PACKAGE_PATH', '')}"
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": "timeout",
            "input": "full_elevation_pcd_original_uneven_ros_chain",
        }

    path = _parse_path(completed.stdout)
    if completed.returncode != 0 or path is None:
        return {
            "found": False,
            "path": None,
            "expansions": None,
            "failure_reason": f"ros_exit_{completed.returncode}",
            "input": "full_elevation_pcd_original_uneven_ros_chain",
        }
    return {
        "found": True,
        "path": path,
        "expansions": int(path.shape[0]),
        "failure_reason": None,
        "input": "full_elevation_pcd_original_uneven_ros_chain",
    }
