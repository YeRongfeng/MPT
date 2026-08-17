"""Phase-1 wiring tests for the token-aligned condition candidate.

These tests exercise only smoke, backward, full Path MeanFlow JVP, decoder
invariants and the raw-S/G wiring closure.  They do not train the model and
do not read any protected test split.
"""

import unittest
from unittest.mock import patch

import torch

from dit.Models import TokenAlignedSpatialPathMeanFlowTransformer
from posterior_pipeline import meanflow_transport_loss


def _small_model(device, **kwargs):
    model = TokenAlignedSpatialPathMeanFlowTransformer(
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_inner=256,
        dropout=0.0,
        **kwargs,
    ).to(device)
    model.use_gradient_checkpoint = False
    return model


def _conditions(batch, device):
    start = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0]], device=device, dtype=torch.float32
    ).expand(batch, -1).clone()
    goal = torch.tensor(
        [[0.5, 0.25, 0.0, 1.0]], device=device, dtype=torch.float32
    ).expand(batch, -1).clone()
    return start, goal


class TokenAlignedPhase1Tests(unittest.TestCase):
    def test_coordinate_encoding_shape_and_device(self):
        encoding = __import__(
            "dit.Models", fromlist=["CoordinateSincosEncoding"]
        ).CoordinateSincosEncoding(d_model=33, max_freq=8.0)
        coordinates = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.25]]], dtype=torch.float32
        )
        result = encoding(coordinates)
        self.assertEqual(tuple(result.shape), (1, 2, 33))
        self.assertTrue(torch.isfinite(result).all())

    def test_forward_backward_and_condition_module_removal(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_model(device).eval()
        batch = 2
        start, goal = _conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device, requires_grad=True)
        output, features = model(
            map_input,
            state,
            torch.ones(batch, device=device),
            torch.zeros(batch, device=device),
            start,
            goal,
            return_features=True,
        )
        self.assertEqual(tuple(output.shape), (batch, 22, 2))
        self.assertEqual(tuple(features["map_tokens"].shape[1:]), (144, 128))
        self.assertEqual(tuple(features["path_free_controls"].shape), (batch, 22, 2))
        self.assertEqual(
            tuple(features["canonical_map_centers"].shape), (batch, 144, 2)
        )
        self.assertFalse(hasattr(model, "pose_embedder"))
        self.assertFalse(hasattr(model, "cond_mlp"))
        keys = set(model.state_dict().keys())
        self.assertFalse(any("pose_embedder" in key for key in keys))
        self.assertFalse(any(key.startswith("cond_mlp.") for key in keys))
        self.assertTrue(any(key.startswith("time_condition_mlp.") for key in keys))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_endpoint_decode_preserves_boundary_invariants(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_model(device).eval()
        start, goal = _conditions(1, device)
        map_input = torch.randn(1, 4, 100, 100, device=device)
        source = torch.randn(1, 22, 2, device=device)
        state = model.sample_pmf_onestep(
            map_input, start, goal, source_noise=source, return_residual=True
        )
        geometry = model.evaluate_trajectory_state(state, start, goal)
        start_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, 0] - torch.zeros(1, device=device)),
            torch.cos(geometry["yaw"][:, 0] - torch.zeros(1, device=device)),
        ).abs()
        goal_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, -1] - torch.pi / 2),
            torch.cos(geometry["yaw"][:, -1] - torch.pi / 2),
        ).abs()
        self.assertLess(
            float(torch.maximum(start_yaw_error, goal_yaw_error).max()), 5e-4
        )

    def test_full_meanflow_jvp_is_finite(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_model(device).eval()
        batch = 2
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        start_pose = torch.tensor(
            [[-8.0, -6.0, 0.3], [2.0, -4.0, 1.2]],
            device=device,
            dtype=torch.float32,
        )
        goal_pose = torch.tensor(
            [[7.0, 3.0, -0.9], [-5.0, 8.0, 2.1]],
            device=device,
            dtype=torch.float32,
        )
        source = torch.randn(batch, 22, 2, device=device)
        target = torch.randn(batch, 22, 2, device=device)
        loss, terms = meanflow_transport_loss(
            model,
            map_input,
            start_pose,
            goal_pose,
            source,
            target,
            device,
            endpoint_probability=0.25,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms["flow"]))
        self.assertTrue(torch.isfinite(terms["endpoint"]))

    def test_raw_sg_wiring_closure_when_derived_inputs_frozen(self):
        """Raw S/G have no hidden learnable path in this architecture.

        Once every derived geometric intermediate is frozen, replacing raw
        S/G tensors must not change the forward output.  This is a wiring
        test, not a statement about semantic invariance.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_model(device).eval()
        batch = 1
        start_a, goal_a = _conditions(batch, device)
        start_b = torch.tensor(
            [[0.7, -0.6, 0.8, 0.6]], device=device, dtype=torch.float32
        )
        goal_b = torch.tensor(
            [[-0.4, 0.5, -0.2, 0.98]], device=device, dtype=torch.float32
        )
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        one = torch.ones(batch, device=device)
        zero = torch.zeros(batch, device=device)

        frozen_map_tokens = torch.randn(batch, 144, 128, device=device)
        frozen_map_centers = torch.randn(batch, 144, 2, device=device)
        frozen_controls = torch.randn(batch, 26, 2, device=device)
        frozen_distance = torch.tensor([0.6], device=device)
        frozen_theta = torch.tensor([0.4], device=device)
        frozen_start_direction = torch.tensor(
            [[0.9, 0.4]], device=device
        )
        frozen_goal_direction = torch.tensor(
            [[0.8, -0.6]], device=device
        )

        with patch.object(
            model,
            "_task_frame",
            return_value=(
                frozen_distance,
                frozen_theta,
                frozen_start_direction,
                frozen_goal_direction,
            ),
        ), patch.object(
            model, "_rotate_horizontal_normals", return_value=map_input
        ), patch.object(
            model,
            "_build_map_memory",
            return_value=(frozen_map_tokens, frozen_map_centers),
        ), patch.object(
            model.trajectory_representation,
            "canonical_control_points",
            return_value=frozen_controls,
        ):
            output_a = model(
                map_input, state, one, zero, start_a, goal_a
            )
            output_b = model(
                map_input, state, one, zero, start_b, goal_b
            )
        self.assertTrue(torch.equal(output_a, output_b))

    def test_task_cond_modes_and_fusions_smoke(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start, goal = _conditions(1, device)
        map_input = torch.randn(1, 4, 100, 100, device=device)
        state = torch.randn(1, 22, 2, device=device)
        one = torch.ones(1, device=device)
        zero = torch.zeros(1, device=device)
        combinations = [
            ("none", "additive"),
            ("scale", "additive"),
            ("direction", "additive"),
            ("direction_scale", "additive"),
            ("scale", "joint_mlp"),
            ("direction_scale", "joint_mlp"),
            ("scale", "additive_with_interaction"),
            ("direction_scale", "additive_with_interaction"),
        ]
        for task_cond_mode, condition_fusion in combinations:
            model = _small_model(
                device,
                task_cond_mode=task_cond_mode,
                condition_fusion=condition_fusion,
            ).eval()
            output = model(
                map_input, state, one, zero, start, goal
            )
            self.assertEqual(tuple(output.shape), (1, 22, 2))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(model.task_cond_mode, task_cond_mode)
            self.assertEqual(model.condition_fusion, condition_fusion)
        with self.assertRaises(ValueError):
            _small_model(
                device,
                task_cond_mode="none",
                condition_fusion="joint_mlp",
            )


if __name__ == "__main__":
    unittest.main()
