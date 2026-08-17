"""Single-variable alignment_only_v2 wiring tests (no training)."""

import unittest

import torch

from dit.Models import (
    AlignmentOnlySpatialPathMeanFlowTransformer,
    AlignmentOnlyV2SpatialPathMeanFlowTransformer,
    SpatialMapPathMeanFlowTransformer,
)
from posterior_pipeline import meanflow_transport_loss

SMALL_ARGS = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 128,
    "d_inner": 256,
    "dropout": 0.0,
}
PRODUCTION_ARGS = {
    "n_layers": 6,
    "n_heads": 8,
    "d_model": 512,
    "d_inner": 1024,
    "dropout": 0.0,
}


def _param_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _conditions(batch, device):
    start = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0]], device=device, dtype=torch.float32
    ).expand(batch, -1).clone()
    goal = torch.tensor(
        [[0.5, 0.25, 0.0, 1.0]], device=device, dtype=torch.float32
    ).expand(batch, -1).clone()
    return start, goal


class AlignmentOnlyV2Tests(unittest.TestCase):
    def test_feature_centers_match_analytic_pool_centers(self):
        model = AlignmentOnlyV2SpatialPathMeanFlowTransformer(
            n_layers=2, n_heads=4, d_model=128, d_inner=256, dropout=0.0
        )
        centers = model.map_feature_centers_norm.reshape(12, 12, 2)
        j = torch.arange(12, dtype=torch.float32)
        expected = -1.0 + 0.02 * (8.0 * j + 3.5)
        self.assertTrue(torch.allclose(centers[0, :, 0], expected))
        self.assertTrue(torch.allclose(centers[:, 0, 1], expected))
        self.assertAlmostEqual(float(centers[0, 0, 0]), -0.93, places=5)
        self.assertAlmostEqual(float(centers[0, 0, 1]), -0.93, places=5)
        self.assertAlmostEqual(float(centers[-1, -1, 0]), 0.83, places=5)
        self.assertAlmostEqual(float(centers[-1, -1, 1]), 0.83, places=5)

    def test_v1_centers_remain_unchanged_linspace(self):
        model = AlignmentOnlySpatialPathMeanFlowTransformer(
            n_layers=2, n_heads=4, d_model=128, d_inner=256, dropout=0.0
        )
        centers = model.map_feature_centers_norm
        self.assertAlmostEqual(float(centers[0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(centers[0, 1]), -1.0, places=5)
        self.assertAlmostEqual(float(centers[-1, 0]), 0.98, places=5)
        self.assertAlmostEqual(float(centers[-1, 1]), 0.98, places=5)

    def test_parameter_count_unchanged_and_forward_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        baseline = SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        v1 = AlignmentOnlySpatialPathMeanFlowTransformer(**PRODUCTION_ARGS)
        v2 = AlignmentOnlyV2SpatialPathMeanFlowTransformer(**PRODUCTION_ARGS)
        self.assertEqual(_param_count(v2), _param_count(baseline))
        self.assertEqual(_param_count(v2), _param_count(v1))
        self.assertEqual(v2.ARCHITECTURE_NAME, "alignment_only_v2")

        model = AlignmentOnlyV2SpatialPathMeanFlowTransformer(
            **SMALL_ARGS
        ).to(device)
        model.use_gradient_checkpoint = False
        batch = 2
        start, goal = _conditions(batch, device)
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
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_full_meanflow_jvp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlignmentOnlyV2SpatialPathMeanFlowTransformer(
            **SMALL_ARGS
        ).to(device)
        model.use_gradient_checkpoint = False
        model.eval()
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


if __name__ == "__main__":
    unittest.main()
