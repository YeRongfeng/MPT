import numpy as np

from baselines.neural_astar_data import (
    moore_labels,
    path_map_to_world,
)


def test_moore_labels_use_official_action_layout_and_skip_walls():
    maze = np.ones((1, 5, 5), dtype=np.float32)
    maze[0, 2, 2] = 0.0
    policies, distances = moore_labels(maze, (0, 4))

    assert policies.shape == (1, 8, 1, 5, 5)
    assert distances.shape == (1, 1, 5, 5)
    assert distances[0, 0, 2, 2] == -1.0
    assert distances[0, 0, 0, 0] == 4.0
    assert policies[:, :, :, 2, 2].sum() == 0.0


def test_path_map_to_world_recovers_diagonal_path():
    path_map = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        path_map[i, i] = 1.0
    path = path_map_to_world(path_map, (0, 0), (4, 4))

    assert path.shape == (5, 2)
    np.testing.assert_allclose(path[0], (-9.9, -9.9), atol=1e-6)
    np.testing.assert_allclose(path[-1], (-9.1, -9.1), atol=1e-6)
