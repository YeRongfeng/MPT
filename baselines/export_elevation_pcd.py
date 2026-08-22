"""Write a dataset elevation grid as an ASCII PCD for external planners."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.common import grid_to_world, load_environment


def elevation_to_xyz(elevation: np.ndarray) -> np.ndarray:
    height, width = elevation.shape
    points = []
    for row in range(height):
        for col in range(width):
            z = float(elevation[row, col])
            if not np.isfinite(z):
                continue
            x, y = grid_to_world(row, col)
            points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def write_pcd(path: Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    with open(path, "w") as handle:
        handle.write(header)
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    env = load_environment(Path(args.env_dir))
    points = elevation_to_xyz(env["elevation"])
    write_pcd(Path(args.output), points)
    print(f"wrote {args.output} with {len(points)} points")


if __name__ == "__main__":
    main()
