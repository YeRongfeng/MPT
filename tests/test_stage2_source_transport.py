import unittest
from unittest.mock import patch

import torch

from stage2_critic import PathAlignedCandidateCritic, physical_path_grid
from stage2_source_transport_training import _evaluate
from train_flow import build_parser, validate_args


class Stage2SourceTransportTests(unittest.TestCase):
    def test_formal_stage2_direct_cost_defaults(self):
        args = build_parser().parse_args(
            [
                "--workflow",
                "stage2",
                "--prior_checkpoint",
                "data/path_meanflow/stage1_best.pth",
            ]
        )
        validate_args(args)
        self.assertEqual(args.stage2_epochs, 3)
        self.assertEqual(args.stage2_sources_per_context, 4)
        self.assertIsNone(args.stage2_max_updates)
        self.assertEqual(args.stage2_eval_every_updates, 100)
        self.assertEqual(args.stage2_log_every_updates, 100)
        self.assertEqual(args.stage2_p_mask, 1.0)
        self.assertEqual(args.stage2_split_seed, 20260802)
        self.assertEqual(args.stage2_train_environments, 24)
        self.assertEqual(args.stage2_validation_sources, 16)
        self.assertEqual(args.stage2_validation_contexts, 16)
        self.assertEqual(args.stage2_validation_environments, 16)
        self.assertEqual(args.stage2_max_regression_rate, 0.05)
        self.assertFalse(hasattr(args, "stage2_method"))

    def test_validation_restores_the_callers_model_mode(self):
        model = torch.nn.Linear(2, 2)

        def fake_eval_mode(current_model, *_args):
            self.assertFalse(current_model.training)
            return {"ok": True}

        with patch(
            "stage2_source_transport_training._evaluate_in_eval_mode",
            side_effect=fake_eval_mode,
        ):
            model.train()
            self.assertEqual(_evaluate(model, None, None, None, None), {"ok": True})
            self.assertTrue(model.training)

            model.eval()
            self.assertEqual(_evaluate(model, None, None, None, None), {"ok": True})
            self.assertFalse(model.training)

    def test_critic_has_two_candidate_aligned_outputs(self):
        critic = PathAlignedCandidateCritic(hidden_dim=32).eval()
        with torch.no_grad():
            risk, safe_logit = critic(
                torch.randn(2, 4, 100, 100),
                torch.randn(2, 4),
                torch.randn(2, 4),
                torch.randn(2, 3, 200, 2),
                torch.rand(2, 3, 200),
            )
        self.assertEqual(tuple(risk.shape), (2, 3))
        self.assertEqual(tuple(safe_logit.shape), (2, 3))
        self.assertTrue(bool((risk >= 0).all()))

    def test_path_grid_uses_x_as_width_and_y_as_height(self):
        grid = physical_path_grid(torch.tensor([[[0.0, 0.0]]]))
        self.assertEqual(tuple(grid.shape), (1, 1, 2))
        self.assertTrue(bool(torch.isfinite(grid).all()))


if __name__ == "__main__":
    unittest.main()
