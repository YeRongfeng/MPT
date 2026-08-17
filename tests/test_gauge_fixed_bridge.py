import math
import unittest

import numpy as np
import torch

from boundary_constrained_path import (
    GaugeFixedFirstOrderBridge,
    quintic_gauge,
)
from grad_optimizer import discrete_turning_curvature
from map_config import SAFETY_COST_CONFIG


def _poses(dtype=torch.float64):
    start = torch.tensor(
        [[-2.0, 1.5, 1.1], [0.3, -1.2, -2.4]], dtype=dtype
    )
    goal = torch.tensor(
        [[6.0, -0.5, -0.7], [4.8, 3.1, 2.2]], dtype=dtype
    )
    return start, goal


def _angle_error(left, right):
    return torch.abs(
        torch.atan2(torch.sin(left - right), torch.cos(left - right))
    )


def test_quintic_gauge_boundary_and_monotonicity():
    t = np.linspace(0.0, 1.0, 1001)
    slope = 0.73
    phi = quintic_gauge(t, slope)
    assert phi[0] == 0.0
    assert phi[-1] == 1.0
    assert np.all(np.diff(phi) > 0.0)
    derivative = np.gradient(phi, t, edge_order=2)
    assert abs(derivative[0] - slope) < 1e-4
    assert abs(derivative[-1] - slope) < 1e-4


def _assert_real_path_coordinate_fitter_regularizes_curvature_spike():
    """Exercise the fitter used by ``path_coordinates``, not the mask fitter."""
    samples = 100
    step = 0.17614245531889225
    parameter = np.linspace(0.0, 1.0, samples)
    coefficients = np.asarray(
        [0.7012304719983945, -1.2026014264993783,
         0.8309242235201438, 0.1708965484039555]
    )
    phases = np.asarray(
        [1.7020357762585556, 0.2925424432047601,
         2.0863593442082973, -1.952839741163541]
    )
    curvature_profile = sum(
        coefficients[index]
        * np.sin((index + 1) * np.pi * parameter + phases[index])
        for index in range(4)
    )
    curvature_profile *= 1.75 / np.max(np.abs(curvature_profile))
    yaw = np.empty(samples)
    yaw[0] = 0.12470859341100304
    yaw[1:] = yaw[0] + np.cumsum(
        0.5
        * (curvature_profile[:-1] + curvature_profile[1:])
        * step
    )
    position = np.zeros((samples, 2))
    midpoint_yaw = 0.5 * (yaw[:-1] + yaw[1:])
    position[1:, 0] = np.cumsum(np.cos(midpoint_yaw) * step)
    position[1:, 1] = np.cumsum(np.sin(midpoint_yaw) * step)
    trajectory = np.concatenate([position, yaw[:, None]], axis=1)

    raw_curvature = discrete_turning_curvature(
        torch.as_tensor(position, dtype=torch.float64).unsqueeze(0)
    ).amax()
    assert float(raw_curvature) < SAFETY_COST_CONFIG.curvature_limit

    bridge = GaugeFixedFirstOrderBridge(dense_points=200)
    state, diagnostics = bridge.fit_demonstration(trajectory)
    repeated_state, repeated_diagnostics = bridge.fit_demonstration(trajectory)
    assert torch.equal(state, repeated_state)
    assert diagnostics == repeated_diagnostics
    assert diagnostics.smoothness_weight > 0.0
    assert diagnostics.curvature_feasible
    assert (
        diagnostics.max_curvature
        <= SAFETY_COST_CONFIG.curvature_limit
        + SAFETY_COST_CONFIG.hard_constraint_epsilon
    )
    assert diagnostics.control_reconstruction_rmse < 0.01

    # Stage-1 consumes float32 path coordinates, so this regression checks the
    # exact tensor/dtype contract audited by ``fit_demonstration``.
    start = torch.as_tensor(trajectory[0:1], dtype=torch.float32)
    goal = torch.as_tensor(trajectory[-1:], dtype=torch.float32)
    geometry = bridge.evaluate(state.unsqueeze(0), start, goal)
    assert torch.allclose(
        geometry["position"][:, 0], start[:, :2], atol=1e-5, rtol=0.0
    )
    assert torch.allclose(
        geometry["position"][:, -1], goal[:, :2], atol=1e-5, rtol=0.0
    )
    bridge.assert_endpoint_yaw(
        state.unsqueeze(0), start, goal, tolerance=5e-5
    )
    assert abs(
        float(
            bridge.audit_curvature(
                state.unsqueeze(0), start, goal
            ).amax()
        )
        - diagnostics.max_curvature
    ) < 1e-6


def _assert_fit_curvature_audit_grid_catches_peak_missed_by_200_points():
    bridge = GaugeFixedFirstOrderBridge(dense_points=200)
    generator = torch.Generator().manual_seed(0)
    # This deterministic, target-scale state has a narrow peak between the
    # deployed 200 samples.  It guards the fitter against regressing to the
    # lower-density admission check.
    state = 0.05 * torch.randn(992, 22, 2, generator=generator)[-1:]
    start = torch.tensor([[-2.0, 0.0, 0.0]])
    goal = torch.tensor([[2.0, 0.0, 0.0]])
    deployed_max = float(
        bridge.evaluate(state, start, goal)["curvature"].amax()
    )
    audited_max = float(bridge.audit_curvature(state, start, goal).amax())
    assert deployed_max < SAFETY_COST_CONFIG.curvature_limit
    assert audited_max > SAFETY_COST_CONFIG.curvature_limit


class DemonstrationFitCurvatureTests(unittest.TestCase):
    def test_real_path_coordinate_fitter_regularizes_curvature_spike(self):
        _assert_real_path_coordinate_fitter_regularizes_curvature_spike()

    def test_fit_curvature_audit_catches_peak_missed_by_200_points(self):
        _assert_fit_curvature_audit_grid_catches_peak_missed_by_200_points()


def test_decode_has_exact_positions_and_analytic_yaw():
    bridge = GaugeFixedFirstOrderBridge(dense_points=200)
    start, goal = _poses()
    state = torch.randn(2, 22, 2, dtype=torch.float64)
    values = bridge.evaluate(state, start, goal)
    assert torch.allclose(
        values["position"][:, 0], start[:, :2], atol=1e-11, rtol=0.0
    )
    assert torch.allclose(
        values["position"][:, -1], goal[:, :2], atol=1e-11, rtol=0.0
    )
    assert float(
        _angle_error(values["yaw"][:, 0], start[:, 2]).max()
    ) < 1e-11
    assert float(
        _angle_error(values["yaw"][:, -1], goal[:, 2]).max()
    ) < 1e-11
    chord = torch.linalg.vector_norm(goal[:, :2] - start[:, :2], dim=1)
    assert torch.allclose(
        values["speed"][:, 0], chord, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        values["speed"][:, -1], chord, atol=1e-10, rtol=1e-10
    )


def test_decoder_is_affine_in_state():
    bridge = GaugeFixedFirstOrderBridge(dense_points=50)
    start, goal = _poses()
    first = torch.randn(2, 22, 2, dtype=torch.float64)
    second = torch.randn(2, 22, 2, dtype=torch.float64)
    weight = 0.37
    mixed = bridge.decode_control_points(
        weight * first + (1.0 - weight) * second, start, goal
    )
    expected = (
        weight * bridge.decode_control_points(first, start, goal)
        + (1.0 - weight)
        * bridge.decode_control_points(second, start, goal)
    )
    assert torch.allclose(mixed, expected, atol=1e-11, rtol=1e-11)


def test_state_control_round_trip():
    bridge = GaugeFixedFirstOrderBridge(dense_points=50)
    start, goal = _poses()
    state = torch.randn(2, 22, 2, dtype=torch.float64)
    condition = bridge._condition(
        start,
        goal,
        dtype=state.dtype,
        device=state.device,
    )
    canonical = bridge.canonical_control_points(
        state, condition[-2], condition[-1]
    )
    recovered = bridge.encode_canonical_controls(
        canonical, condition[-2], condition[-1]
    )
    assert torch.allclose(state, recovered, atol=1e-11, rtol=1e-11)


def test_se2_equivariance():
    bridge = GaugeFixedFirstOrderBridge(dense_points=50)
    start, goal = _poses()
    state = torch.randn(2, 22, 2, dtype=torch.float64)
    control = bridge.decode_control_points(state, start, goal)
    angle = 0.83
    translation = torch.tensor([1.7, -3.2], dtype=torch.float64)
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=torch.float64,
    )
    transformed_start = start.clone()
    transformed_goal = goal.clone()
    transformed_start[:, :2] = start[:, :2] @ rotation.T + translation
    transformed_goal[:, :2] = goal[:, :2] @ rotation.T + translation
    transformed_start[:, 2] += angle
    transformed_goal[:, 2] += angle
    transformed_control = bridge.decode_control_points(
        state, transformed_start, transformed_goal
    )
    expected = control @ rotation.T + translation
    assert torch.allclose(
        transformed_control, expected, atol=1e-11, rtol=1e-11
    )


def test_analytic_derivatives_match_finite_differences_away_from_boundary():
    bridge = GaugeFixedFirstOrderBridge(dense_points=1001)
    start, goal = _poses()
    generator = torch.Generator().manual_seed(17)
    state = 0.1 * torch.randn(
        2, 22, 2, dtype=torch.float64, generator=generator
    )
    values = bridge.evaluate(state, start, goal)
    step = 1.0 / 1000.0
    finite_first = (
        values["position"][:, 2:] - values["position"][:, :-2]
    ) / (2.0 * step)
    finite_second = (
        values["position"][:, 2:]
        - 2.0 * values["position"][:, 1:-1]
        + values["position"][:, :-2]
    ) / step**2
    first_reference = values["first_derivative"][:, 1:-1]
    second_reference = values["second_derivative"][:, 1:-1]
    first_relative = torch.linalg.vector_norm(
        finite_first - first_reference, dim=-1
    ) / torch.linalg.vector_norm(first_reference, dim=-1).clamp_min(1.0)
    second_relative = torch.linalg.vector_norm(
        finite_second - second_reference, dim=-1
    ) / torch.linalg.vector_norm(second_reference, dim=-1).clamp_min(1.0)
    # Central differences crossing a knot see the cubic spline's discontinuous
    # third derivative, so use a scale-aware bound instead of elementwise
    # equality at those samples.
    assert float(first_relative.max()) < 0.015
    assert float(second_relative.max()) < 0.05


if __name__ == "__main__":
    unittest.main()
