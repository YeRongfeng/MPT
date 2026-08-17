import unittest
from types import SimpleNamespace

import torch

from experiment_direct_cost_mgda_vs_fixed_sgd import (
    _apply_gradient_tuple,
    _apply_mgda_gradients,
    _build_train_manifest,
)
from posterior_pipeline import (
    DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS,
    _direct_cost_config_from_args,
    _direct_cost_optimizer_name,
    _make_direct_cost_optimizer,
    _require_direct_cost_resume_protocol,
)
from train_flow import build_parser


class TestPairedPlainSGD(unittest.TestCase):
    def test_formal_mgda_uses_plain_sgd(self):
        model = torch.nn.Linear(2, 1)
        args = SimpleNamespace(stage2_use_mgda=True, stage2_lr=1e-5)
        optimizer = _make_direct_cost_optimizer(model, args)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(_direct_cost_optimizer_name(args), "SGD")
        self.assertEqual(optimizer.defaults["momentum"], 0.0)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)

    def test_formal_fixed_arm_retains_adam(self):
        model = torch.nn.Linear(2, 1)
        args = SimpleNamespace(stage2_use_mgda=False, stage2_lr=1e-5)
        optimizer = _make_direct_cost_optimizer(model, args)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(_direct_cost_optimizer_name(args), "Adam")

    def test_resume_rejects_legacy_mgda_adam_optimizer(self):
        args = build_parser().parse_args(["--workflow", "stage2"])
        checkpoint = {
            "stage2_training_protocol_semantics": (
                DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS
            ),
            "stage2_optimizer": "Adam",
            "direct_cost_config": _direct_cost_config_from_args(args),
        }
        with self.assertRaisesRegex(ValueError, "optimizer mismatch"):
            _require_direct_cost_resume_protocol(checkpoint, args)

    def test_resume_accepts_fixed_adam_optimizer(self):
        args = build_parser().parse_args(
            ["--workflow", "stage2", "--no-stage2_use_mgda"]
        )
        checkpoint = {
            "stage2_training_protocol_semantics": (
                DIRECT_COST_TRAINING_PROTOCOL_SEMANTICS
            ),
            "stage2_optimizer": "Adam",
            "direct_cost_config": _direct_cost_config_from_args(args),
        }
        _require_direct_cost_resume_protocol(checkpoint, args)

    def test_manual_fixed_update_is_plain_sgd(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        gradient = torch.tensor([0.5, -0.25])
        _apply_gradient_tuple([parameter], [gradient], 0.1)
        self.assertTrue(torch.allclose(parameter, torch.tensor([0.95, -1.975])))

    def test_manual_mgda_update_is_weighted_gradient_sum(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        gradients = (
            (torch.tensor([1.0, 0.0]),),
            (torch.tensor([0.0, 2.0]),),
            (torch.tensor([-1.0, 1.0]),),
        )
        alpha = [0.2, 0.3, 0.5]
        _apply_mgda_gradients([parameter], gradients, alpha, 0.1)
        expected_gradient = torch.tensor([-0.3, 1.1])
        self.assertTrue(torch.allclose(parameter, torch.tensor([1.0, 2.0]) - 0.1 * expected_gradient))

    def test_training_manifest_is_deterministic(self):
        first = _build_train_manifest(
            12,
            epochs=2,
            batch_size=3,
            order_seed=17,
            source_seed=29,
        )
        second = _build_train_manifest(
            12,
            epochs=2,
            batch_size=3,
            order_seed=17,
            source_seed=29,
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(len(epoch) for epoch in first), 8)
        self.assertEqual(
            sorted(index for batch in first[0] for index in batch["indices"]),
            list(range(12)),
        )


if __name__ == "__main__":
    unittest.main()
