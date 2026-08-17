import unittest

import torch

from posterior_pipeline import (
    frozen_stage1_environment_split,
    stage1_endpoint_curvature_auxiliary,
)


class Stage1TrainingProtocolTests(unittest.TestCase):
    def test_split_is_disjoint_and_deterministic(self):
        environments = [f"env{index:06d}" for index in range(100)]
        first = frozen_stage1_environment_split(
            environments,
            seed=20260802,
            train_count=80,
            validation_count=20,
        )
        second = frozen_stage1_environment_split(
            reversed(environments),
            seed=20260802,
            train_count=80,
            validation_count=20,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 80)
        self.assertEqual(len(first["validation"]), 20)
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(first["unused"])

    def test_split_rejects_oversubscription(self):
        with self.assertRaisesRegex(ValueError, "101/100"):
            frozen_stage1_environment_split(
                [f"env{index:06d}" for index in range(100)],
                seed=20260802,
                train_count=81,
                validation_count=20,
            )

    def test_curvature_auxiliary_penalizes_only_relative_excess(self):
        class DummyModel:
            def audit_trajectory_state_curvature(
                self, state, start_condition, goal_condition
            ):
                curvature = torch.zeros(
                    state.shape[0], 1001, dtype=state.dtype
                )
                curvature[:, 500] = 2.1 * (
                    1.0 + torch.relu(state[:, 0, 0])
                )
                return curvature

        state = torch.zeros(2, 22, 2, requires_grad=True)
        with torch.no_grad():
            state[1, 0, 0] = 1.0
        condition = torch.zeros(2, 4)
        loss, pass_rate, curvature = stage1_endpoint_curvature_auxiliary(
            DummyModel(),
            state,
            condition,
            condition,
            tail_ratio=0.01,
        )
        self.assertEqual(curvature.shape, (2, 1001))
        self.assertAlmostEqual(float(pass_rate), 0.5)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))
        self.assertGreater(float(state.grad[1, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
