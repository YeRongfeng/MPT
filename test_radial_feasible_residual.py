import unittest

import torch

from bspline_utils import DifferentiableBSpline
from dit.Models import PhysicalScaledEdgeResidualRepresentation


class RadialFeasibleResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.rep = PhysicalScaledEdgeResidualRepresentation()

    def test_arbitrary_outputs_decode_inside_box_and_preserve_endpoints(self):
        batch_size = 64
        start = torch.empty(batch_size, 2).uniform_(-0.9, 0.9)
        goal = torch.empty(batch_size, 2).uniform_(-0.9, 0.9)
        raw_v = torch.randn(batch_size, 25, 2) * 4.0
        raw_s = torch.randn(batch_size, 1)

        residual, diagnostics = self.rep.radial_feasible_residual(
            raw_v,
            raw_s,
            start,
            goal,
            return_diagnostics=True,
        )
        control_points = self.rep.decode(residual, start, goal)
        dense = DifferentiableBSpline(
            num_control_points=26,
            num_output_points=100,
            degree=3,
        )(control_points)

        self.assertLess(residual.sum(dim=1).abs().max().item(), 1e-5)
        self.assertLess((control_points[:, 0] - start).abs().max().item(), 1e-5)
        self.assertLess((control_points[:, -1] - goal).abs().max().item(), 1e-5)
        self.assertLessEqual(control_points.abs().max().item(), 1.0 + 1e-5)
        self.assertLessEqual(dense.abs().max().item(), 1.0 + 1e-5)
        self.assertLessEqual(diagnostics['x_norm'].max().item(), 1.0 + 1e-6)
        self.assertTrue(torch.isfinite(diagnostics['rho']).all())
        self.assertTrue(torch.isfinite(diagnostics['hard_boundary_margin']).all())
        self.assertEqual(diagnostics['active_control_point'].shape, (batch_size,))
        self.assertEqual(diagnostics['active_coordinate'].shape, (batch_size,))

    def test_legal_ground_truth_has_ball_coordinate_and_recovers_exactly(self):
        batch_size = 32
        control_points = torch.empty(batch_size, 26, 2).uniform_(-0.85, 0.85)
        start = control_points[:, 0].clone()
        goal = control_points[:, -1].clone()
        residual = self.rep.encode(control_points, start, goal)
        residual_norm = torch.linalg.vector_norm(
            residual.flatten(start_dim=1), dim=1
        )
        direction = residual / residual_norm.clamp_min(1e-6).view(-1, 1, 1)
        rho = self.rep.compute_max_feasible_radius(direction, start, goal)
        ball_coordinate = residual / rho.clamp_min(1e-6).view(-1, 1, 1)
        recovered = rho.view(-1, 1, 1) * ball_coordinate

        ball_norm = torch.linalg.vector_norm(
            ball_coordinate.flatten(start_dim=1), dim=1
        )
        self.assertLessEqual(ball_norm.max().item(), 1.0 + 1e-5)
        self.assertLess((recovered - residual).abs().max().item(), 1e-5)
        self.assertLess(
            (self.rep.decode(recovered, start, goal) - control_points)
            .abs()
            .max()
            .item(),
            1e-5,
        )

    def test_gradients_flow_through_direction_radius_and_slack(self):
        batch_size = 8
        start = torch.empty(batch_size, 2).uniform_(-0.8, 0.8)
        goal = torch.empty(batch_size, 2).uniform_(-0.8, 0.8)
        raw_v = torch.randn(batch_size, 25, 2, requires_grad=True)
        # Zero raw_s must remain trainable; softplus converts it to positive
        # slack with a non-zero derivative.
        raw_s = torch.zeros(batch_size, 1, requires_grad=True)

        residual = self.rep.radial_feasible_residual(
            raw_v,
            raw_s,
            start,
            goal,
        )
        weights = torch.linspace(
            0.5, 1.5, residual.numel(), dtype=residual.dtype
        ).reshape_as(residual)
        loss = (residual * weights).square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(raw_v.grad).all())
        self.assertTrue(torch.isfinite(raw_s.grad).all())
        self.assertGreater(raw_v.grad.abs().sum().item(), 0.0)
        self.assertGreater(raw_s.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
