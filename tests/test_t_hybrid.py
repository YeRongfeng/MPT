"""Coordinate and terrain-file helpers for the T-Hybrid adapter."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from baselines.t_hybrid import (
    thybrid_to_world,
    world_to_thybrid,
    write_occupancy,
    write_terrain,
    _parse_path,
    _voxel_id,
)
from map_config import MAP_CONFIG


class THybridAdapterTests(unittest.TestCase):
    def test_world_roundtrip_corners(self) -> None:
        origin_x, origin_y = MAP_CONFIG.origin_xy
        x, y = world_to_thybrid(origin_x, origin_y)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        back = thybrid_to_world(10.0, 4.0)
        self.assertAlmostEqual(back[0], origin_x + 10.0)
        self.assertAlmostEqual(back[1], origin_y + 4.0)

    def test_voxel_id_matches_t_hybrid_formula(self) -> None:
        # ID = car_X + car_Y * volnum, volnum = 100 on the 20 m / 0.2 m map.
        self.assertEqual(_voxel_id(0, 0, 100), 0)
        self.assertEqual(_voxel_id(3, 2, 100), 203)

    def test_occupancy_and_terrain_files(self) -> None:
        occupancy = np.zeros((2, 2), dtype=bool)
        occupancy[0, 1] = True
        elevation = np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32)
        normals = np.zeros((2, 2, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        with TemporaryDirectory() as tmp:
            occ_path = Path(tmp) / "occ.txt"
            ter_path = Path(tmp) / "terrain.txt"
            write_occupancy(occ_path, occupancy)
            write_terrain(ter_path, occupancy, elevation, normals)
            occ = occ_path.read_text().splitlines()
            self.assertEqual(occ[0], "2 2")
            self.assertEqual(occ[1], "0 1")
            terrain = ter_path.read_text().splitlines()
            ids = [int(line.split()[0]) for line in terrain]
            static = {int(line.split()[0]): float(line.split()[1]) for line in terrain}
            self.assertEqual(ids, [0, 1, 100, 101])
            self.assertEqual(static[1], 0.0)
            self.assertEqual(static[0], 1.0)

    def test_parse_path_skips_logs(self) -> None:
        stdout = "start store vol information\nfound 2\n1.0 2.0 0.3\n3.0 4.0 1.2\n"
        path = _parse_path(stdout)
        self.assertEqual(path.shape, (2, 3))
        origin_x, origin_y = MAP_CONFIG.origin_xy
        self.assertAlmostEqual(float(path[0, 0]), origin_x + 1.0)
        self.assertAlmostEqual(float(path[0, 1]), origin_y + 2.0)


if __name__ == "__main__":
    unittest.main()
