"""Data and geometry adapters for the official Neural A* implementation.

The ICML-2021 code consumes binary grids rather than terrain tensors.  This
module keeps that conversion independent from the historical Neural A*
dependencies so dataset preparation and unit tests work in the MPT runtime.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from baselines.common import PlanningTask, grid_to_world, world_to_grid


# This is the action order used by planning_experiment.utils.mechanism.Moore.
MOORE_ACTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 1),
    (-1, -1),
    (1, 1),
    (1, -1),
)


def _one_hot(shape: Tuple[int, int], location: Tuple[int, int]) -> np.ndarray:
    result = np.zeros((1, 1, *shape), dtype=np.float32)
    result[(0, 0, *location)] = 1.0
    return result


def task_to_maps(task: PlanningTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an MPT task to ``(maze, start_map, goal_map)``.

    The returned maze follows the official convention: traversable cells are
    one and obstacles are zero.  Coordinates use the existing MPT row/column
    mapping, so no transposition or map resizing is hidden in this adapter.
    """

    occupancy = np.asarray(task.occupancy, dtype=bool)
    if occupancy.ndim != 2 or occupancy.shape[0] != occupancy.shape[1]:
        raise ValueError(f"Neural A* requires a square 2-D grid, got {occupancy.shape}")
    maze = (~occupancy).astype(np.float32)[None, None]
    start = world_to_grid(float(task.start[0]), float(task.start[1]), shape=occupancy.shape)
    goal = world_to_grid(float(task.goal[0]), float(task.goal[1]), shape=occupancy.shape)
    if occupancy[start] or occupancy[goal]:
        raise ValueError(f"start/goal must be traversable: start={start}, goal={goal}")
    return maze, _one_hot(occupancy.shape, start), _one_hot(occupancy.shape, goal)


def moore_labels(
    maze: np.ndarray,
    goal: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build official-format shortest-path policy and distance labels.

    The original datasets use unit-cost Moore moves.  Unreachable cells keep
    distance ``-1`` (the official evaluator treats the minimum distance as the
    impossible-distance sentinel), while obstacle policies remain all zero.
    """

    grid = np.asarray(maze, dtype=np.float32)
    if grid.ndim == 3:
        grid = grid[0]
    if grid.ndim == 4:
        grid = grid[0, 0]
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"maze must be square, got {grid.shape}")
    height, width = grid.shape
    gy, gx = map(int, goal)
    if not (0 <= gy < height and 0 <= gx < width) or grid[gy, gx] == 0:
        raise ValueError(f"goal must be traversable and in bounds: {goal}")

    distances = np.full((height, width), -1.0, dtype=np.float32)
    distances[gy, gx] = 0.0
    queue = deque([(gy, gx)])
    while queue:
        row, col = queue.popleft()
        for drow, dcol in MOORE_ACTIONS:
            nr, nc = row + drow, col + dcol
            if 0 <= nr < height and 0 <= nc < width:
                if grid[nr, nc] != 0 and distances[nr, nc] < 0:
                    distances[nr, nc] = distances[row, col] + 1.0
                    queue.append((nr, nc))

    policies = np.zeros((8, 1, height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            if grid[row, col] == 0 or distances[row, col] <= 0:
                continue
            for action, (drow, dcol) in enumerate(MOORE_ACTIONS):
                nr, nc = row + drow, col + dcol
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and distances[nr, nc] >= 0
                    and distances[nr, nc] == distances[row, col] - 1.0
                ):
                    policies[action, 0, row, col] = 1.0
                    break
    return policies[None], distances[None, None]


def task_to_official_sample(
    task: PlanningTask,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one sample in the official ``MazeDataset`` tensor layout."""

    maze, start_map, goal_map = task_to_maps(task)
    goal = np.argwhere(goal_map[0, 0] > 0.5)
    policies, distances = moore_labels(maze, tuple(goal[0]))
    return maze, goal_map, policies, distances, start_map


def save_npz(
    samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
    *,
    train_fraction: float = 0.8,
) -> None:
    """Write MPT samples using the four-array official ``.npz`` layout.

    The fifth start-map value is intentionally not serialized: official
    Neural A* samples a start from the distance map at runtime.  A deterministic
    start is retained by ``task_to_official_sample`` for inference adapters.
    """

    if len(samples) < 2:
        raise ValueError("at least two samples are required for train/valid splits")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    n_train = max(1, min(len(samples) - 1, int(round(len(samples) * train_fraction))))
    n_valid = max(1, len(samples) - n_train)
    groups = (list(samples[:n_train]), list(samples[n_train:n_train + n_valid]))
    if not groups[1]:
        groups = (list(samples[:-1]), [samples[-1]])

    arrays: List[np.ndarray] = []
    for group in groups:
        arrays.extend([
            np.concatenate([sample[0] for sample in group], axis=0),
            np.concatenate([sample[1] for sample in group], axis=0),
            np.concatenate([sample[2] for sample in group], axis=0),
            np.concatenate([sample[3] for sample in group], axis=0),
        ])
    # Keep the test split equal to validation for a usable smoke dataset.
    arrays.extend(arrays[4:8])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, *arrays)


def path_map_to_world(
    path_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> np.ndarray:
    """Recover an ordered grid path from Neural A*'s binary path map."""

    allowed = np.asarray(path_map) > 0.5
    if allowed.ndim == 3:
        allowed = allowed[0]
    if allowed.ndim == 4:
        allowed = allowed[0, 0]
    if allowed.ndim != 2:
        raise ValueError(f"path_map must be 2-D, got {allowed.shape}")
    allowed = allowed.copy()
    allowed[start] = True
    allowed[goal] = True
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for drow, dcol in MOORE_ACTIONS:
            nxt = (current[0] + drow, current[1] + dcol)
            if (
                0 <= nxt[0] < allowed.shape[0]
                and 0 <= nxt[1] < allowed.shape[1]
                and allowed[nxt]
                and nxt not in parent
            ):
                parent[nxt] = current
                queue.append(nxt)
    if goal not in parent:
        return np.empty((0, 2), dtype=np.float32)
    cells = []
    current = goal
    while current is not None:
        cells.append(current)
        current = parent[current]
    cells.reverse()
    return np.asarray([grid_to_world(row, col) for row, col in cells], dtype=np.float32)
