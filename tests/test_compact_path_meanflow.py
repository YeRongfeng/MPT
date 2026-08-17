import unittest

import torch

from dit.Models import (
    CompactPathMeanFlowTransformer,
    SpatialMapPathMeanFlowTransformer,
)


class CompactPathMeanFlowTest(unittest.TestCase):
    @staticmethod
    def _conditions(batch, device):
        start = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]], device=device
        ).expand(batch, -1).clone()
        goal = torch.tensor(
            [[0.5, 0.25, 0.0, 1.0]], device=device
        ).expand(batch, -1).clone()
        return start, goal

    def test_parameter_budget_and_gpu_forward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CompactPathMeanFlowTransformer(
            n_layers=4,
            n_heads=4,
            d_model=256,
            d_inner=768,
            dropout=0.0,
        ).to(device)
        params = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(params, 10_000_000)
        batch = 2
        start, goal = self._conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device, requires_grad=True)
        output = model(
            map_input,
            state,
            torch.ones(batch, device=device),
            torch.zeros(batch, device=device),
            start,
            goal,
        )
        self.assertEqual(tuple(output.shape), (batch, 22, 2))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_endpoint_decode_and_one_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CompactPathMeanFlowTransformer(dropout=0.0).to(device).eval()
        start, goal = self._conditions(1, device)
        map_input = torch.randn(1, 4, 100, 100, device=device)
        source = torch.randn(1, 22, 2, device=device)
        state = model.sample_pmf_onestep(
            map_input, start, goal, source_noise=source, return_residual=True
        )
        geometry = model.evaluate_trajectory_state(state, start, goal)
        self.assertEqual(tuple(geometry["control_points"].shape), (1, 26, 2))
        self.assertEqual(tuple(geometry["position"].shape[-2:]), (200, 2))
        start_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, 0] - torch.zeros(1, device=device)),
            torch.cos(geometry["yaw"][:, 0] - torch.zeros(1, device=device)),
        ).abs()
        goal_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, -1] - torch.pi / 2),
            torch.cos(geometry["yaw"][:, -1] - torch.pi / 2),
        ).abs()
        self.assertLess(float(torch.maximum(start_yaw_error, goal_yaw_error).max()), 5e-4)

    def test_spatial_map_conditioning_forward_and_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpatialMapPathMeanFlowTransformer(
            n_layers=2,
            n_heads=4,
            d_model=128,
            d_inner=256,
            dropout=0.0,
        ).to(device)
        self.assertFalse(hasattr(model, "guidance_encoder"))
        batch = 2
        start, goal = self._conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(
            batch, 22, 2, device=device, requires_grad=True
        )
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
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_spatial_map_layouts_and_tokenwise_head(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = 1
        start, goal = self._conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        expected_tokens = {"single_12": 144, "single_25": 625, "multi": 769}
        for layout, token_count in expected_tokens.items():
            model = SpatialMapPathMeanFlowTransformer(
                n_layers=2,
                n_heads=4,
                d_model=128,
                d_inner=256,
                dropout=0.0,
                map_layout=layout,
            ).to(device).eval()
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
            self.assertEqual(
                tuple(features["map_tokens"].shape[1:]),
                (token_count, 128),
            )
            self.assertEqual(model.OUTPUT_HEAD_TYPE, "tokenwise_22x2")
            self.assertEqual(model.main_pred[-1].out_features, 2)

        model = SpatialMapPathMeanFlowTransformer(
            n_layers=2,
            n_heads=4,
            d_model=128,
            d_inner=256,
            dropout=0.0,
            map_layout="single_12",
        ).to(device)
        first = model.dit_blocks[0].map_cross_attn
        second = model.dit_blocks[1].map_cross_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            self.assertIsNot(
                getattr(first, name).weight,
                getattr(second, name).weight,
            )


if __name__ == "__main__":
    unittest.main()
