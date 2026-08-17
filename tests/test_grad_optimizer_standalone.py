import math
import inspect
import unittest

import torch

from bspline_utils import DifferentiableBSpline
from grad_optimizer import (
    _finite_difference_yaw,
    build_direct_cost_standalone_parser,
    run_direct_cost_standalone,
    run_formal_expert_standalone,
)


class GradOptimizerStandaloneTests(unittest.TestCase):
    def test_direct_entrypoint_uses_model_cost_backprop(self):
        source = inspect.getsource(run_direct_cost_standalone)
        self.assertIn("_direct_cost_forward(", source)
        self.assertIn('loss = output["task_cost"].mean()', source)
        self.assertIn("loss.backward()", source)
        self.assertIn("torch.optim.Adam(model.parameters()", source)
        self.assertNotIn("collect_privileged_distillation_round(", source)
        self.assertNotIn("optimize_control_points_multistep(", source)

    def test_direct_entrypoint_defaults_are_short_and_masked(self):
        args = build_direct_cost_standalone_parser().parse_args([])
        self.assertEqual(args.updates, 80)
        self.assertEqual(args.train_contexts, 1)
        self.assertEqual(args.sources_per_context, 1)
        self.assertEqual(args.pair_attempts, 8)
        self.assertEqual(args.early_stop_patience, 2)
        self.assertGreater(args.min_masked_fraction, 0.0)

    def test_direct_entrypoint_reuses_fixed_source(self):
        source = inspect.getsource(run_direct_cost_standalone)
        self.assertIn("fixed_train_source_seed = args.seed + 20_000", source)
        self.assertIn("source=fixed_train_source", source)
        self.assertIn("apply_privileged_path_correction(", source)
        self.assertIn('"training_uses_expert_target": False', source)

    def test_direct_entrypoint_rejects_impossible_endpoints(self):
        source = inspect.getsource(run_direct_cost_standalone)
        self.assertIn("endpoint_stability_feasibility(", source)
        self.assertIn("mask_start_goal_connected(", source)
        self.assertIn("require_endpoint_feasible=True", source)

    def test_standalone_uses_formal_stage2_expert_collector(self):
        # Retained as an explicitly callable historical comparison, but it is
        # no longer the file's direct-execution entrypoint.
        source = inspect.getsource(run_formal_expert_standalone)
        self.assertIn("collect_privileged_distillation_round(", source)
        self.assertIn("PrivilegedConstraintDistillationConfig()", source)
        self.assertNotIn("optimize_control_points_multistep(", source)

    def test_finite_difference_yaw_for_straight_and_vertical_paths(self):
        horizontal = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
        )
        vertical = torch.tensor(
            [[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]]
        )
        self.assertTrue(
            torch.allclose(
                _finite_difference_yaw(horizontal),
                torch.zeros(1, 3),
            )
        )
        self.assertTrue(
            torch.allclose(
                _finite_difference_yaw(vertical),
                torch.full((1, 3), math.pi / 2.0),
            )
        )

    def test_finite_difference_yaw_preserves_gradients(self):
        trajectory = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.2], [2.0, 0.0]]],
            requires_grad=True,
        )
        _finite_difference_yaw(trajectory).square().sum().backward()
        self.assertIsNotNone(trajectory.grad)
        self.assertTrue(bool(torch.isfinite(trajectory.grad).all()))

    def test_bspline_analytic_geometry_for_straight_path(self):
        layer = DifferentiableBSpline(
            num_control_points=6,
            num_output_points=31,
            degree=3,
        )
        control_points = torch.stack(
            [
                torch.linspace(0.0, 5.0, 6),
                torch.zeros(6),
            ],
            dim=1,
        ).unsqueeze(0)
        geometry = layer.evaluate_geometry(control_points)

        self.assertEqual(geometry["position"].shape, (1, 31, 2))
        self.assertEqual(geometry["first_derivative"].shape, (1, 31, 2))
        self.assertEqual(geometry["second_derivative"].shape, (1, 31, 2))
        self.assertTrue(
            torch.allclose(
                geometry["yaw"],
                torch.zeros(1, 31),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                geometry["curvature"],
                torch.zeros(1, 31),
                atol=1e-6,
            )
        )

    def test_bspline_analytic_geometry_preserves_gradients(self):
        layer = DifferentiableBSpline(
            num_control_points=6,
            num_output_points=31,
            degree=3,
        )
        control_points = torch.randn(2, 6, 2, requires_grad=True)
        geometry = layer.evaluate_geometry(control_points)
        (
            geometry["position"].square().mean()
            + geometry["curvature"].mean()
        ).backward()
        self.assertIsNotNone(control_points.grad)
        self.assertTrue(bool(torch.isfinite(control_points.grad).all()))


if __name__ == "__main__":
    unittest.main()
