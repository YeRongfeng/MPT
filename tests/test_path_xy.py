"""Wiring tests for the path_xy weak-XY hint (no training)."""

import unittest

import torch

from dit.Models import (
    PathXYSpatialPathMeanFlowTransformer,
    SpatialMapPathMeanFlowTransformer,
)
from posterior_pipeline import meanflow_transport_loss
from train_compact_stage1 import ARCHITECTURES, build_model, condition_semantics


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
SHARED_MODULES = (
    "map_fe_block1",
    "map_fe_block2",
    "map_fe_block3",
    "map_fe_block4",
    "map_position_enc",
    "path_patchify",
    "time_embedder",
    "pose_embedder",
    "cond_mlp",
    "layer_norm",
    "main_pred",
)


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


def _mean_l2(tensor):
    flat = tensor.reshape(-1, tensor.shape[-1])
    return float(torch.linalg.vector_norm(flat, dim=-1).mean())


class PathXYTests(unittest.TestCase):
    def test_factory_and_parameter_delta(self):
        device = torch.device("cpu")
        baseline = SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        model, _ = build_model("path_xy", device)
        extra = 3 * PRODUCTION_ARGS["d_model"]
        self.assertEqual(model.ARCHITECTURE_NAME, "path_xy")
        self.assertEqual(condition_semantics("path_xy"), "path_xy_linear_hint_v1")
        self.assertIn("path_xy", ARCHITECTURES)
        self.assertEqual(_param_count(model), _param_count(baseline) + extra)
        self.assertEqual(
            tuple(model.path_xy_proj.weight.shape),
            (PRODUCTION_ARGS["d_model"], 2),
        )
        self.assertAlmostEqual(model.path_xy_alpha, 0.1)
        self.assertFalse(hasattr(model, "coord_pe"))
        self.assertFalse(hasattr(model, "map_feature_centers_norm"))
        self.assertTrue(hasattr(model, "pose_embedder"))
        self.assertTrue(hasattr(model, "cond_mlp"))
        self.assertFalse(hasattr(model, "time_condition_mlp"))
        self.assertEqual(
            type(model.dit_blocks[0]).__name__,
            type(baseline.dit_blocks[0]).__name__,
        )
        for name in SHARED_MODULES:
            self.assertEqual(
                type(getattr(model, name)).__name__,
                type(getattr(baseline, name)).__name__,
                name,
            )

    def test_only_path_xy_injection_changes_forward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        baseline = SpatialMapPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        model = PathXYSpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        missing, unexpected = model.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertTrue(any("path_xy_proj" in name for name in missing))
        self.assertEqual(unexpected, [])
        nn_zeros = torch.nn.init.zeros_
        nn_zeros(model.path_xy_proj.weight)
        nn_zeros(model.path_xy_proj.bias)
        model.use_gradient_checkpoint = False
        baseline.use_gradient_checkpoint = False
        model.eval()
        baseline.eval()

        batch = 2
        start, goal = _conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        one = torch.ones(batch, device=device)
        zero = torch.zeros(batch, device=device)
        with torch.no_grad():
            base_out, base_feat = baseline(
                map_input, state, one, zero, start, goal, return_features=True
            )
            xy_out, xy_feat = model(
                map_input, state, one, zero, start, goal, return_features=True
            )
        self.assertTrue(torch.equal(base_out, xy_out))
        self.assertTrue(torch.equal(base_feat["map_tokens"], xy_feat["map_tokens"]))
        self.assertTrue(torch.allclose(xy_feat["path_xy_hint"], torch.zeros_like(
            xy_feat["path_xy_hint"]
        )))

        model.path_xy_proj.weight.data.normal_(std=0.05)
        with torch.no_grad():
            changed, changed_feat = model(
                map_input, state, one, zero, start, goal, return_features=True
            )
        self.assertFalse(torch.equal(base_out, changed))
        self.assertTrue(
            torch.equal(base_feat["map_tokens"], changed_feat["map_tokens"])
        )
        self.assertGreater(
            float((changed_feat["path_xy_hint"]).abs().max()), 0.0
        )

    def test_forward_backward_and_meanflow_jvp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PathXYSpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        model.use_gradient_checkpoint = False
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
        self.assertEqual(tuple(features["path_free_xy"].shape), (batch, 22, 2))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(state.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.path_xy_proj.weight.grad.abs().sum()), 0.0
        )

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
        jvp_loss, terms = meanflow_transport_loss(
            model,
            map_input,
            start_pose,
            goal_pose,
            source,
            target,
            device,
            endpoint_probability=0.25,
        )
        self.assertTrue(torch.isfinite(jvp_loss))
        self.assertTrue(torch.isfinite(terms["flow"]))
        self.assertTrue(torch.isfinite(terms["endpoint"]))

    def test_xy_hint_does_not_dominate_path_mlp_at_init(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(20260816)
        model = PathXYSpatialPathMeanFlowTransformer(**PRODUCTION_ARGS).to(device)
        model.eval()
        batch = 4
        start, goal = _conditions(batch, device)
        start = start.clone()
        goal = goal.clone()
        start[:, :2] = torch.tensor([[-6.0, -4.0]], device=device)
        goal[:, :2] = torch.tensor([[5.0, 3.0]], device=device)
        state = torch.randn(batch, 22, 2, device=device)
        with torch.no_grad():
            components = model.path_token_components(state, start, goal)
        norms = {
            "path_mlp": _mean_l2(components["path_mlp"]),
            "path_index_pe": _mean_l2(components["path_index_pe"]),
            "path_xy_raw": _mean_l2(components["path_xy_raw"]),
            "path_xy_hint": _mean_l2(components["path_xy_hint"]),
            "path_tokens": _mean_l2(components["path_tokens"]),
        }
        self.assertLess(norms["path_xy_hint"], norms["path_mlp"])
        self.assertLess(norms["path_xy_hint"] / max(norms["path_mlp"], 1e-6), 0.5)
        self.assertAlmostEqual(
            norms["path_xy_hint"],
            model.path_xy_alpha * norms["path_xy_raw"],
            places=5,
        )
        print("path_xy init component norms:", norms)


if __name__ == "__main__":
    unittest.main()
