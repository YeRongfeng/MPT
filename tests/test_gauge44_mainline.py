import unittest

import torch

from dit.Models import PathDiffusionTransformer
from boundary_constrained_path import (
    GaugeFixedBridge,
    REPRESENTATION_SEMANTIC_VERSION,
)
from grad_optimizer import trajectory_validity_metrics
from map_config import MAP_CONFIG
from posterior_pipeline import DaggerReplayBuffer, normalize_poses


def _poses():
    start = torch.tensor([[0.0, 0.0, 0.4], [-2.0, 1.0, -0.7]])
    goal = torch.tensor([[5.0, 3.0, -0.2], [4.0, 6.0, 0.8]])
    return start, goal


def test_production_model_uses_unconstrained_44d_state():
    model = PathDiffusionTransformer(
        n_layers=1,
        n_heads=4,
        d_model=64,
        d_inner=128,
        dropout=0.0,
    ).eval()
    start_pose, goal_pose = _poses()
    start, goal = normalize_poses(
        start_pose, goal_pose, model.coordinate_scale, torch.device("cpu")
    )
    source = torch.randn(2, 22, 2)
    output = model(
        torch.randn(2, 4, 100, 100),
        source,
        torch.ones(2),
        torch.zeros(2),
        start,
        goal,
    )
    assert output.shape == (2, 22, 2)
    assert model.project_zero_sum(source).data_ptr() == source.data_ptr()
    geometry = model.evaluate_trajectory_state(output, start, goal)
    start_error = torch.atan2(
        torch.sin(geometry["yaw"][:, 0] - start_pose[:, 2]),
        torch.cos(geometry["yaw"][:, 0] - start_pose[:, 2]),
    ).abs()
    goal_error = torch.atan2(
        torch.sin(geometry["yaw"][:, -1] - goal_pose[:, 2]),
        torch.cos(geometry["yaw"][:, -1] - goal_pose[:, 2]),
    ).abs()
    assert float(torch.maximum(start_error, goal_error).max()) < 5e-4


def test_hard_validity_requires_analytic_geometry():
    bridge = GaugeFixedBridge()
    start_pose, goal_pose = _poses()
    state = torch.randn(2, 22, 2)
    geometry = bridge.evaluate(state, start_pose, goal_pose)
    stability = torch.ones(MAP_CONFIG.cost_map_size)
    try:
        trajectory_validity_metrics(
            geometry["position"],
            stability,
            MAP_CONFIG.cost_map_info(),
            start_pose=start_pose,
            goal_pose=goal_pose,
        )
    except ValueError as error:
        assert "analytic_yaw" in str(error)
    else:
        raise AssertionError("finite-difference hard-validity was accepted")

    result = trajectory_validity_metrics(
        geometry["position"],
        stability,
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=geometry["yaw"],
        analytic_curvature=geometry["curvature"],
        start_pose=start_pose,
        goal_pose=goal_pose,
    )
    assert bool(result["endpoint_yaw_ok"].all())


def test_dense_curvature_audit_controls_hard_validity():
    bridge = GaugeFixedBridge()
    start_pose, goal_pose = _poses()
    state = torch.zeros(2, 22, 2)
    geometry = bridge.evaluate(state, start_pose, goal_pose)
    dense_curvature = torch.zeros_like(geometry["curvature"])
    audit_curvature = torch.zeros(2, 1001)
    audit_curvature[0, 517] = 2.5
    result = trajectory_validity_metrics(
        geometry["position"],
        torch.ones(MAP_CONFIG.cost_map_size),
        MAP_CONFIG.cost_map_info(),
        analytic_yaw=geometry["yaw"],
        analytic_curvature=dense_curvature,
        analytic_curvature_audit=audit_curvature,
        start_pose=start_pose,
        goal_pose=goal_pose,
    )
    assert result["curvature_ok"].tolist() == [False, True]
    assert torch.allclose(result["dense_max_curvature"], torch.zeros(2))
    assert torch.allclose(result["max_curvature"], torch.tensor([2.5, 0.0]))
    assert result["curvature_audit_points"].tolist() == [1001, 1001]


def test_model_curvature_audit_uses_physical_units():
    model = PathDiffusionTransformer(
        n_layers=1,
        n_heads=4,
        d_model=64,
        d_inner=128,
        dropout=0.0,
    ).eval()
    start_pose, goal_pose = _poses()
    start, goal = normalize_poses(
        start_pose, goal_pose, model.coordinate_scale, torch.device("cpu")
    )
    state = torch.zeros(2, 22, 2)
    actual = model.audit_trajectory_state_curvature(state, start, goal)
    expected = model.trajectory_representation.audit_curvature(
        state,
        model._condition_pose_to_xy_yaw(start),
        model._condition_pose_to_xy_yaw(goal),
    ) / model.coordinate_scale
    assert actual.shape == (2, 1001)
    assert torch.allclose(actual, expected)


def test_replay_version_binds_gauge44_semantics():
    buffer = DaggerReplayBuffer(8)
    state = buffer.state_dict()
    assert state["version"] == 6
    assert (
        state["representation_semantic_version"]
        == REPRESENTATION_SEMANTIC_VERSION
    )
    old = dict(state)
    old["version"] = 5
    try:
        DaggerReplayBuffer.from_state_dict(old)
    except ValueError as error:
        assert "旧 25×2 replay" in str(error)
    else:
        raise AssertionError("legacy replay version was accepted")


class Gauge44MainlineUnittestBridge(unittest.TestCase):
    """Expose the function-style regression cases to unittest discovery."""

    def test_production_model(self):
        test_production_model_uses_unconstrained_44d_state()

    def test_hard_validity(self):
        test_hard_validity_requires_analytic_geometry()

    def test_dense_curvature_audit(self):
        test_dense_curvature_audit_controls_hard_validity()

    def test_model_curvature_audit(self):
        test_model_curvature_audit_uses_physical_units()

    def test_replay_version(self):
        test_replay_version_binds_gauge44_semantics()


if __name__ == "__main__":
    unittest.main()
