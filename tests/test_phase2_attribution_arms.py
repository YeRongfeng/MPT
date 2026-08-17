"""Phase-2 attribution-arm wiring tests (no training).

Covers the two additional architectures required before the controlled
Stage-1 comparison:

* ``alignment_only``: single_12 condition path + canonical coordinate PE;
* ``token_aligned_capacity_matched``: full token-aligned geometry with
  half-width additive time/task heads (both conditions may modulate
  SA/CA/FFN).
"""

import unittest

import torch

from dit.Models import (
    AlignmentOnlySpatialPathMeanFlowTransformer,
    AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer,
    SpatialMapPathMeanFlowTransformer,
)
from posterior_pipeline import meanflow_transport_loss

PRODUCTION_ARGS = {
    "n_layers": 6,
    "n_heads": 8,
    "d_model": 512,
    "d_inner": 1024,
    "dropout": 0.0,
}
SMALL_ARGS = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 128,
    "d_inner": 256,
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


class Phase2AttributionArmTests(unittest.TestCase):
    def test_alignment_only_has_exact_baseline_parameter_count(self):
        baseline = SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        aligned = AlignmentOnlySpatialPathMeanFlowTransformer(
            **PRODUCTION_ARGS
        )
        self.assertEqual(_param_count(aligned), _param_count(baseline))
        self.assertTrue(hasattr(aligned, "pose_embedder"))
        self.assertTrue(hasattr(aligned, "cond_mlp"))
        self.assertEqual(aligned.ARCHITECTURE_NAME, "alignment_only")

    def test_alignment_only_forward_backward_and_full_meanflow_jvp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlignmentOnlySpatialPathMeanFlowTransformer(
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

        model.eval()
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

    def test_capacity_matched_parameter_count_is_within_8_percent_below(self):
        baseline = _param_count(
            SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        )
        model = AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer(
            **PRODUCTION_ARGS
        )
        params = _param_count(model)
        delta = (params - baseline) / baseline
        self.assertLessEqual(delta, 0.0)
        self.assertGreater(delta, -0.08)
        self.assertFalse(hasattr(model, "pose_embedder"))
        self.assertFalse(hasattr(model, "cond_mlp"))
        self.assertEqual(model.condition_fusion, "additive_half_width")
        self.assertEqual(
            model.ARCHITECTURE_NAME, "token_aligned_capacity_matched"
        )

    def test_capacity_matched_forward_and_full_meanflow_jvp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer(
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

        model.eval()
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

    def test_capacity_matched_task_cond_none_smoke(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer(
            **SMALL_ARGS,
            task_cond_mode="none",
        ).to(device).eval()
        start, goal = _conditions(1, device)
        output = model(
            torch.randn(1, 4, 100, 100, device=device),
            torch.randn(1, 22, 2, device=device),
            torch.ones(1, device=device),
            torch.zeros(1, device=device),
            start,
            goal,
        )
        self.assertEqual(tuple(output.shape), (1, 22, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
