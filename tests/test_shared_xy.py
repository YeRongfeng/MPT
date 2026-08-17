"""Wiring tests for the shared_xy weak XY language (no training)."""

import unittest

import torch

from dit.Models import (
    AlignmentOnlyV2SpatialPathMeanFlowTransformer,
    PathXYSpatialPathMeanFlowTransformer,
    SharedXYSpatialPathMeanFlowTransformer,
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


class SharedXYTests(unittest.TestCase):
    def test_factory_parameter_count_and_shared_projection(self):
        device = torch.device("cpu")
        baseline = SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        path_xy = PathXYSpatialPathMeanFlowTransformer(**PRODUCTION_ARGS)
        model, _ = build_model("shared_xy", device)
        extra = 3 * PRODUCTION_ARGS["d_model"]
        self.assertEqual(model.ARCHITECTURE_NAME, "shared_xy")
        self.assertEqual(
            condition_semantics("shared_xy"), "shared_xy_linear_hint_v1"
        )
        self.assertIn("shared_xy", ARCHITECTURES)
        self.assertEqual(_param_count(model), _param_count(baseline) + extra)
        self.assertEqual(_param_count(model), _param_count(path_xy))
        self.assertFalse(hasattr(model, "coord_pe"))
        self.assertTrue(hasattr(model, "pose_embedder"))
        self.assertTrue(hasattr(model, "cond_mlp"))
        self.assertFalse(hasattr(model, "time_condition_mlp"))
        self.assertIs(model.path_xy_proj, model.path_xy_proj)
        self.assertEqual(
            type(model.dit_blocks[0]).__name__,
            type(baseline.dit_blocks[0]).__name__,
        )

    def test_feature_centers_match_corrected_pool_centers(self):
        model = SharedXYSpatialPathMeanFlowTransformer(**SMALL_ARGS)
        reference = AlignmentOnlyV2SpatialPathMeanFlowTransformer(**SMALL_ARGS)
        self.assertTrue(
            torch.allclose(
                model.map_feature_centers_norm,
                reference.map_feature_centers_norm,
            )
        )
        centers = model.map_feature_centers_norm.reshape(12, 12, 2)
        self.assertAlmostEqual(float(centers[0, 0, 0]), -0.93, places=5)
        self.assertAlmostEqual(float(centers[-1, -1, 0]), 0.83, places=5)

    def test_only_shared_map_xy_differs_from_path_xy(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        path_xy = PathXYSpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        model = SharedXYSpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        missing, unexpected = model.load_state_dict(
            path_xy.state_dict(), strict=False
        )
        self.assertTrue(
            any("map_feature_centers_norm" in name for name in missing)
        )
        self.assertEqual(unexpected, [])
        path_xy.use_gradient_checkpoint = False
        model.use_gradient_checkpoint = False
        path_xy.eval()
        model.eval()

        batch = 2
        start, goal = _conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        one = torch.ones(batch, device=device)
        zero = torch.zeros(batch, device=device)
        with torch.no_grad():
            path_out, path_feat = path_xy(
                map_input, state, one, zero, start, goal, return_features=True
            )
            shared_out, shared_feat = model(
                map_input, state, one, zero, start, goal, return_features=True
            )
        self.assertTrue(torch.equal(path_feat["path_xy_hint"], shared_feat["path_xy_hint"]))
        self.assertFalse(torch.equal(path_feat["map_tokens"], shared_feat["map_tokens"]))
        reconstructed = path_feat["map_tokens"] + shared_feat["map_xy_hint"]
        self.assertTrue(torch.allclose(shared_feat["map_tokens"], reconstructed))
        # AdaLN-Zero keeps CA gated off at init, so force a unit CA gate
        # to confirm the map XY hint can change the generator output.
        hidden = model.dit_blocks[0].adaLN_modulation[-1].bias.numel() // 9
        for block in list(path_xy.dit_blocks) + list(model.dit_blocks):
            block.adaLN_modulation[-1].bias.data[5 * hidden:6 * hidden] = 1.0
        with torch.no_grad():
            path_out, _ = path_xy(
                map_input, state, one, zero, start, goal, return_features=True
            )
            shared_out, _ = model(
                map_input, state, one, zero, start, goal, return_features=True
            )
        self.assertFalse(torch.equal(path_out, shared_out))

        torch.nn.init.zeros_(model.path_xy_proj.weight)
        torch.nn.init.zeros_(model.path_xy_proj.bias)
        baseline = SpatialMapPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        baseline.load_state_dict(path_xy.state_dict(), strict=False)
        baseline.use_gradient_checkpoint = False
        baseline.eval()
        with torch.no_grad():
            zero_out, zero_feat = model(
                map_input, state, one, zero, start, goal, return_features=True
            )
            base_out, base_feat = baseline(
                map_input, state, one, zero, start, goal, return_features=True
            )
        self.assertTrue(torch.equal(zero_out, base_out))
        self.assertTrue(torch.equal(zero_feat["map_tokens"], base_feat["map_tokens"]))

    def test_forward_backward_and_meanflow_jvp(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SharedXYSpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
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
        self.assertEqual(tuple(features["map_free_xy"].shape), (batch, 144, 2))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertGreater(float(model.path_xy_proj.weight.grad.abs().sum()), 0.0)

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

    def test_xy_hint_does_not_dominate_at_init(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(20260816)
        model = SharedXYSpatialPathMeanFlowTransformer(**PRODUCTION_ARGS).to(device)
        model.eval()
        batch = 4
        start, goal = _conditions(batch, device)
        start = start.clone()
        goal = goal.clone()
        start[:, :2] = torch.tensor([[-6.0, -4.0]], device=device)
        goal[:, :2] = torch.tensor([[5.0, 3.0]], device=device)
        state = torch.randn(batch, 22, 2, device=device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        with torch.no_grad():
            path_c = model.path_token_components(state, start, goal)
            map_c = model.map_token_components(map_input, start, goal)
        path_mlp = _mean_l2(path_c["path_mlp"])
        path_hint = _mean_l2(path_c["path_xy_hint"])
        map_base = _mean_l2(map_c["map_base"])
        map_hint = _mean_l2(map_c["map_xy_hint"])
        self.assertLess(path_hint, path_mlp)
        self.assertLess(path_hint / max(path_mlp, 1e-6), 0.5)
        self.assertLess(map_hint / max(map_base, 1e-6), 0.5)
        print(
            "shared_xy init norms:",
            {
                "path_mlp": path_mlp,
                "path_xy_hint": path_hint,
                "map_base": map_base,
                "map_xy_hint": map_hint,
            },
        )


if __name__ == "__main__":
    unittest.main()
