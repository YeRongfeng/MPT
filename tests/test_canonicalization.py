import math
import unittest

import torch

from bspline_utils import DifferentiableBSpline
from dit.Models import (
    PathDiffusionTransformer,
    PhysicalScaledEdgeResidualRepresentation,
)
from geometry.canonicalization import (
    canonicalize_map,
    canonicalize_pose,
    canonicalize_residual,
    canonicalize_trajectory,
    decanonicalize_pose,
    decanonicalize_residual,
    decanonicalize_trajectory,
    wrap_angle,
)


def _poses(dtype=torch.float64):
    start = torch.tensor([[1.25, -2.0, 2.9]], dtype=dtype)
    goal = torch.tensor([[4.25, 2.0, -2.8]], dtype=dtype)
    return start, goal


class CanonicalizationTests(unittest.TestCase):
    def test_pose_round_trip_start_origin_goal_positive_x_and_yaw_wrap(self):
        start, goal = _poses()
        poses = torch.tensor(
            [[[-2.0, 1.0, -3.5], [4.0, -3.0, 3.8]]], dtype=torch.float64
        )
        canonical = canonicalize_trajectory(poses, start, goal)
        restored = decanonicalize_trajectory(canonical, start, goal)
        self.assertTrue(
            torch.allclose(restored[..., :2], poses[..., :2], atol=1e-10)
        )
        self.assertTrue(
            torch.allclose(
                wrap_angle(restored[..., 2]),
                wrap_angle(poses[..., 2]),
                atol=1e-10,
            )
        )

        start_c = canonicalize_pose(start, start, goal)
        goal_c = canonicalize_pose(goal, start, goal)
        self.assertTrue(
            torch.allclose(start_c[:, :2], torch.zeros_like(start_c[:, :2]))
        )
        self.assertTrue(bool(torch.all(goal_c[:, 0] > 0)))
        self.assertTrue(
            torch.allclose(
                goal_c[:, 1], torch.zeros_like(goal_c[:, 1]), atol=1e-10
            )
        )
        self.assertTrue(
            bool(torch.all((start_c[:, 2] >= -math.pi) & (start_c[:, 2] < math.pi)))
        )

        restored_start = decanonicalize_pose(start_c, start, goal)
        self.assertTrue(torch.allclose(restored_start, start, atol=1e-10))

    def test_zero_sum_residual_rotation_and_round_trip(self):
        start, goal = _poses()
        generator = torch.Generator().manual_seed(4)
        residual = torch.randn(1, 25, 2, generator=generator, dtype=torch.float64)
        residual = residual - residual.mean(dim=1, keepdim=True)
        canonical = canonicalize_residual(residual, start, goal)
        self.assertLess(float(canonical.sum(dim=1).abs().max()), 1e-10)
        restored = decanonicalize_residual(canonical, start, goal)
        self.assertTrue(torch.allclose(restored, residual, atol=1e-10))

    def test_bspline_decode_commutes_with_rotation(self):
        start, goal = _poses(dtype=torch.float32)
        representation = PhysicalScaledEdgeResidualRepresentation()
        generator = torch.Generator().manual_seed(8)
        residual = representation.project_zero_sum(
            torch.randn(1, 25, 2, generator=generator)
        )
        start_c = canonicalize_pose(start, start, goal)
        goal_c = canonicalize_pose(goal, start, goal)
        residual_c = canonicalize_residual(residual, start, goal)

        global_control = representation.decode(
            residual, start[:, :2], goal[:, :2]
        )
        canonical_control = representation.decode(
            residual_c, start_c[:, :2], goal_c[:, :2]
        )
        expected_control = canonicalize_trajectory(global_control, start, goal)
        self.assertTrue(
            torch.allclose(canonical_control, expected_control, atol=2e-6)
        )

        bspline = DifferentiableBSpline(26, 100, 3)
        global_dense = bspline(global_control)
        canonical_dense = bspline(canonical_control)
        expected_dense = canonicalize_trajectory(global_dense, start, goal)
        self.assertTrue(
            torch.allclose(canonical_dense, expected_dense, atol=3e-6)
        )

    def test_map_spatial_and_normal_transform_and_scalar_elevation(self):
        raw = torch.zeros(4, 5, 5)
        raw[0, 3, 2] = 7.0
        raw[1, 3, 2] = 1.0
        raw[3] = 1.0
        start = torch.tensor([0.0, 0.0, 0.25])
        goal = torch.tensor([0.0, 1.0, -0.25])
        result = canonicalize_map(
            raw,
            start,
            goal,
            source_bounds=(-2.0, 2.0, -2.0, 2.0),
            output_bounds=(-2.0, 2.0, -2.0, 2.0),
            normal_channels=(1, 2, 3),
            mode="nearest",
        )
        self.assertEqual(float(result.values[0, 2, 3]), 7.0)
        self.assertTrue(
            torch.allclose(result.values[1, 2, 3], torch.tensor(0.0), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(result.values[2, 2, 3], torch.tensor(-1.0), atol=1e-6)
        )
        self.assertEqual(float(result.values[3, 2, 3]), 1.0)
        self.assertEqual(float(result.valid_mask[0, 2, 3]), 1.0)

    def test_same_source_has_same_canonical_value_under_rotated_task(self):
        generator = torch.Generator().manual_seed(13)
        source_c = torch.randn(1, 25, 2, generator=generator)
        source_c = source_c - source_c.mean(dim=1, keepdim=True)
        start_1 = torch.tensor([[0.0, 0.0, 0.2]])
        goal_1 = torch.tensor([[4.0, 0.0, -0.3]])
        alpha = 1.1
        rotation_task = torch.tensor([[0.0, 0.0, alpha]])
        translation = torch.tensor([[-1.2, 2.3]])

        def rotate(xy):
            c, s = math.cos(alpha), math.sin(alpha)
            matrix = torch.tensor([[c, -s], [s, c]])
            return xy @ matrix.T

        source_global_1 = source_c
        source_global_2 = rotate(source_global_1)
        start_2 = torch.cat(
            (
                rotate(start_1[:, :2]) + translation,
                rotation_task[:, 2:3] + start_1[:, 2:3],
            ),
            dim=1,
        )
        goal_2 = torch.cat(
            (
                rotate(goal_1[:, :2]) + translation,
                rotation_task[:, 2:3] + goal_1[:, 2:3],
            ),
            dim=1,
        )
        canonical_1 = canonicalize_residual(source_global_1, start_1, goal_1)
        canonical_2 = canonicalize_residual(source_global_2, start_2, goal_2)
        self.assertTrue(torch.allclose(canonical_1, canonical_2, atol=2e-6))

    def test_disabled_map_path_is_bitwise_identity(self):
        map_tensor = torch.randn(3, 11, 9)
        start = torch.tensor([1.0, -2.0, 0.0])
        goal = torch.tensor([2.0, 4.0, 0.0])
        result = canonicalize_map(
            map_tensor,
            start,
            goal,
            source_bounds=(-1.0, 1.0, -1.0, 1.0),
            enabled=False,
        )
        self.assertEqual(result.values.data_ptr(), map_tensor.data_ptr())
        self.assertTrue(torch.equal(result.values, map_tensor))
        self.assertTrue(torch.equal(result.valid_mask, torch.ones(1, 11, 9)))

    def test_disabled_preprocessing_preserves_model_output_exactly(self):
        torch.manual_seed(21)
        model = PathDiffusionTransformer(
            n_layers=1,
            n_heads=4,
            d_model=32,
            d_inner=64,
            dropout=0.0,
            coordinate_scale=10.0,
            map_channels=3,
        ).eval()
        map_tensor = torch.randn(1, 3, 100, 100)
        start_pose = torch.tensor([[-0.2, -0.1, 1.0, 0.0]])
        goal_pose = torch.tensor([[0.4, 0.5, 0.0, 1.0]])
        source = model.project_zero_sum(torch.randn(1, 22, 2))
        disabled = canonicalize_map(
            map_tensor,
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            source_bounds=(-10.0, 9.8, -10.0, 9.8),
            enabled=False,
        )
        with torch.no_grad():
            reference = model(
                map_tensor,
                source,
                torch.ones(1),
                torch.zeros(1),
                start_pose,
                goal_pose,
            )
            observed = model(
                disabled.values,
                source,
                torch.ones(1),
                torch.zeros(1),
                start_pose,
                goal_pose,
            )
        self.assertTrue(torch.equal(reference, observed))

if __name__ == "__main__":
    unittest.main()
