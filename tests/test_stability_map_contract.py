import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataLoader_dit import (
    STABILITY_MAP_FORMAT_VERSION,
    STABILITY_MAP_SEMANTIC_VERSION,
    compute_map_yaw_bins,
    generate_sdf_from_yaw_stability,
    stability_cache_is_compatible,
    stability_source_map_sha256,
)
from grad_optimizer import (
    _sample_cost_map_at_xy_yaw,
    endpoint_stability_feasibility,
)


class StabilityMapContractTest(unittest.TestCase):
    def test_compute_map_yaw_bins_orients_nz_upward_before_angles(self):
        nx = torch.full((2, 3), 0.75)
        ny = torch.full((2, 3), -0.25)
        nz = torch.full((2, 3), 0.35)
        positive = compute_map_yaw_bins(nx, ny, nz, yaw_bins=36)
        negative = compute_map_yaw_bins(nx, ny, -nz, yaw_bins=36)
        self.assertTrue(torch.equal(positive, negative))

    def test_compute_map_yaw_bins_preserves_row_column_axes(self):
        nx = torch.zeros(2, 3)
        ny = torch.zeros(2, 3)
        nz = torch.ones(2, 3)
        # One steep cell is unsafe for every yaw. It must stay at row=0,col=2.
        nx[0, 2] = 1.0
        nz[0, 2] = 0.01
        result = compute_map_yaw_bins(nx, ny, nz, yaw_bins=8)
        self.assertEqual(result.shape, (2, 3, 8))
        self.assertTrue(torch.equal(result[0, 2], torch.zeros(8)))
        self.assertTrue(torch.equal(result[1, 0], torch.ones(8)))

    def test_uniform_stability_fields_have_constant_signed_distance(self):
        for value, sign in ((1.0, 1.0), (0.0, -1.0)):
            field = np.full((3, 4, 5), value, dtype=np.float32)
            esdf = generate_sdf_from_yaw_stability(
                field,
                voxel_size_xy=0.2,
                yaw_weight=1.4,
                use_scipy=True,
            )
            self.assertEqual(np.unique(esdf).size, 1)
            self.assertGreater(sign * float(esdf.flat[0]), 0.0)

    def test_continuous_sampler_uses_xy_cell_centers_and_periodic_yaw(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (6, 5, 4),
        }
        cost_map = torch.empty(5, 6, 4)
        for row in range(5):
            for col in range(6):
                for yaw_bin in range(4):
                    cost_map[row, col, yaw_bin] = (
                        100 * row + 10 * col + yaw_bin
                    )

        # Centers of row=2 and columns 1/3. yaw=0 is exactly yaw bin 2.
        xy = torch.tensor([[[-0.85, -0.75], [-0.65, -0.75]]])
        yaw = torch.zeros(1, 2)
        sampled = _sample_cost_map_at_xy_yaw(
            xy, yaw, cost_map, map_info, "cpu"
        )
        self.assertTrue(
            torch.allclose(
                sampled, torch.tensor([[212.0, 232.0]]), atol=1e-4
            )
        )

        periodic = torch.zeros(5, 6, 4)
        periodic[..., 3] = 1.0
        epsilon = 1e-3
        probe_xy = torch.tensor([[[-0.85, -0.75]]])
        positive = _sample_cost_map_at_xy_yaw(
            probe_xy,
            torch.tensor([[np.pi - epsilon]]),
            periodic,
            map_info,
            "cpu",
        )
        negative = _sample_cost_map_at_xy_yaw(
            probe_xy,
            torch.tensor([[-np.pi + epsilon]]),
            periodic,
            map_info,
            "cpu",
        )
        self.assertLess(float((positive - negative).abs().max()), 0.01)

    def test_cache_contract_binds_semantics_and_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            map_file = directory / "map.p"
            with map_file.open("wb") as handle:
                pickle.dump(
                    {"tensor": np.zeros((2, 3, 4), dtype=np.float32)},
                    handle,
                )
            source_hash = stability_source_map_sha256(map_file)
            cache_file = directory / "stability_map.npz"
            np.savez_compressed(
                cache_file,
                yaw_stability=np.ones((2, 3, 4), dtype=np.float32),
                cost_map=np.ones((2, 3, 4), dtype=np.float32),
                yaw_bins=np.int32(4),
                voxel_size_xy=np.float32(0.1),
                yaw_weight=np.float32(1.4),
                format_version=np.int32(STABILITY_MAP_FORMAT_VERSION),
                semantic_version=np.asarray(STABILITY_MAP_SEMANTIC_VERSION),
                source_map_sha256=np.asarray(source_hash),
            )
            with np.load(cache_file) as cached:
                kwargs = dict(
                    map_shape=(2, 3),
                    resolution=0.1,
                    yaw_bins=4,
                    yaw_weight=1.4,
                    source_map_sha256=source_hash,
                )
                self.assertTrue(stability_cache_is_compatible(cached, **kwargs))
                self.assertFalse(
                    stability_cache_is_compatible(
                        cached,
                        **{**kwargs, "source_map_sha256": "0" * 64},
                    )
                )

    def test_endpoint_precheck_defaults_to_exact_represented_yaw(self):
        map_info = {
            "origin": (-1.0, -1.0, -np.pi),
            "resolution": 0.1,
            "size": (20, 20, 8),
        }
        cost_map = torch.ones(20, 20, 8)
        # Exact yaw=0 is bin 4 and is unsafe at the fixed start cell.  Nearby
        # yaw is safe, so an interval-max check would incorrectly admit this
        # context even though the 44D representation cannot change endpoint yaw.
        cost_map[10, 5, 4] = 0.05
        start = torch.tensor([-0.45, 0.05, 0.0])
        goal = torch.tensor([0.55, 0.05, 0.0])

        exact = endpoint_stability_feasibility(
            start,
            goal,
            cost_map,
            map_info,
            d_safe=0.15,
        )
        interval = endpoint_stability_feasibility(
            start,
            goal,
            cost_map,
            map_info,
            d_safe=0.15,
            start_yaw_tolerance_rad=0.35,
            goal_yaw_tolerance_rad=0.35,
        )
        self.assertFalse(exact["start_feasible"])
        self.assertTrue(interval["start_feasible"])
        self.assertEqual(exact["start_yaw_tolerance_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
