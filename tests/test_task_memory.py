"""Wiring tests for the task_memory patch+SA encoder (no training)."""

import unittest

import torch

from dit.Models import (
    SpatialMapPathMeanFlowTransformer,
    TaskMemorySpatialPathMeanFlowTransformer,
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


class TaskMemoryTests(unittest.TestCase):
    def test_factory_and_token_shapes(self):
        device = torch.device("cpu")
        baseline = SpatialMapPathMeanFlowTransformer(**PRODUCTION_ARGS)
        model, _ = build_model("task_memory", device)
        self.assertEqual(model.ARCHITECTURE_NAME, "task_memory")
        self.assertEqual(
            condition_semantics("task_memory"),
            "task_memory_patch_sg_sa_v1",
        )
        self.assertIn("task_memory", ARCHITECTURES)
        self.assertFalse(hasattr(model, "cond_mlp"))
        self.assertFalse(hasattr(model, "map_fe_block1"))
        self.assertTrue(hasattr(model, "pose_embedder"))
        self.assertTrue(hasattr(model, "map_patch_embed"))
        self.assertEqual(model.map_patch_embed.kernel_size, (10, 10))
        self.assertEqual(len(model.task_sa), 2)
        self.assertEqual(
            type(model.dit_blocks[0]).__name__,
            type(baseline.dit_blocks[0]).__name__,
        )
        print(
            "task_memory params:",
            _param_count(model),
            "baseline params:",
            _param_count(baseline),
            "delta:",
            _param_count(model) - _param_count(baseline),
        )

    def test_time_does_not_enter_task_memory(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaskMemorySpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        model.use_gradient_checkpoint = False
        model.eval()
        batch = 2
        start, goal = _conditions(batch, device)
        map_input = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        with torch.no_grad():
            _, feat_a = model(
                map_input,
                state,
                torch.ones(batch, device=device) * 0.2,
                torch.zeros(batch, device=device),
                start,
                goal,
                return_features=True,
            )
            _, feat_b = model(
                map_input,
                state,
                torch.ones(batch, device=device) * 0.9,
                torch.ones(batch, device=device) * 0.3,
                start,
                goal,
                return_features=True,
            )
        self.assertEqual(tuple(feat_a["task_memory"].shape), (batch, 102, 128))
        self.assertTrue(torch.equal(feat_a["task_memory"], feat_b["task_memory"]))
        self.assertFalse(torch.equal(feat_a["condition"], feat_b["condition"]))

    def test_start_goal_and_map_change_task_memory(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaskMemorySpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
        model.use_gradient_checkpoint = False
        model.eval()
        batch = 1
        start, goal = _conditions(batch, device)
        map_a = torch.randn(batch, 4, 100, 100, device=device)
        map_b = torch.randn(batch, 4, 100, 100, device=device)
        state = torch.randn(batch, 22, 2, device=device)
        one = torch.ones(batch, device=device)
        zero = torch.zeros(batch, device=device)
        start_b = start.clone()
        start_b[:, :2] = torch.tensor([[0.3, -0.2]], device=device)
        with torch.no_grad():
            _, feat_a = model(
                map_a, state, one, zero, start, goal, return_features=True
            )
            _, feat_sg = model(
                map_a, state, one, zero, start_b, goal, return_features=True
            )
            _, feat_map = model(
                map_b, state, one, zero, start, goal, return_features=True
            )
        self.assertFalse(
            torch.equal(feat_a["task_memory"], feat_sg["task_memory"])
        )
        self.assertFalse(
            torch.equal(feat_a["task_memory"], feat_map["task_memory"])
        )

    def test_forward_backward_jvp_and_decoder(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaskMemorySpatialPathMeanFlowTransformer(**SMALL_ARGS).to(device)
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
        hidden = model.dit_blocks[0].adaLN_modulation[-1].bias.numel() // 9
        for block in model.dit_blocks:
            block.adaLN_modulation[-1].bias.data[5 * hidden:6 * hidden] = 1.0
        model.zero_grad(set_to_none=True)
        state.grad = None
        output = model(
            map_input,
            state,
            torch.ones(batch, device=device),
            torch.zeros(batch, device=device),
            start,
            goal,
        )
        output.square().mean().backward()
        self.assertGreater(
            float(model.map_patch_embed.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.pose_embedder[0].weight.grad.abs().sum()), 0.0
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

        decoded = model.sample_pmf_onestep(
            map_input[:1],
            start[:1],
            goal[:1],
            source_noise=source[:1],
            return_residual=True,
        )
        geometry = model.evaluate_trajectory_state(decoded, start[:1], goal[:1])
        start_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, 0]),
            torch.cos(geometry["yaw"][:, 0]),
        ).abs()
        goal_yaw_error = torch.atan2(
            torch.sin(geometry["yaw"][:, -1] - torch.pi / 2),
            torch.cos(geometry["yaw"][:, -1] - torch.pi / 2),
        ).abs()
        self.assertLess(
            float(torch.maximum(start_yaw_error, goal_yaw_error).max()), 5e-4
        )


if __name__ == "__main__":
    unittest.main()
