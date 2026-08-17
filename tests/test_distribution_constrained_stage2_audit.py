import unittest

import torch

from audit_distribution_constrained_stage2 import (
    _accept_candidate,
    _zero_if_none,
    closed_form_constraint_update,
)


class DistributionConstrainedStage2AuditTest(unittest.TestCase):
    def test_closed_form_update_hits_linearized_constraint(self):
        g_s = (torch.tensor([1.0, 0.0]),)
        g_f = (torch.tensor([1.0, 0.0]),)

        lambda_f, combined, increment = closed_form_constraint_update(
            g_s,
            g_f,
            h_f=1.0,
            eta=0.5,
            eps=1e-12,
        )

        self.assertAlmostEqual(lambda_f, 1.0, places=6)
        self.assertTrue(torch.equal(combined[0], torch.tensor([2.0, 0.0])))
        self.assertAlmostEqual(increment, -1.0, places=6)

    def test_lambda_is_recomputed_for_backtracked_eta(self):
        g_s = (torch.tensor([1.0, 0.0]),)
        g_f = (torch.tensor([1.0, 0.0]),)

        lambda_large, _, _ = closed_form_constraint_update(
            g_s, g_f, h_f=1.0, eta=0.5, eps=1e-12
        )
        lambda_small, _, _ = closed_form_constraint_update(
            g_s, g_f, h_f=1.0, eta=0.25, eps=1e-12
        )

        self.assertAlmostEqual(lambda_large, 1.0, places=6)
        self.assertAlmostEqual(lambda_small, 3.0, places=6)

    def test_unused_gradients_are_zero(self):
        parameter_a = torch.nn.Parameter(torch.ones(2))
        parameter_b = torch.nn.Parameter(torch.ones(2))
        gradients = _zero_if_none(
            (None, torch.tensor([2.0, 3.0])),
            (parameter_a, parameter_b),
        )

        self.assertTrue(torch.equal(gradients[0], torch.zeros(2)))
        self.assertTrue(torch.equal(gradients[1], torch.tensor([2.0, 3.0])))

    def test_infeasible_candidate_must_reduce_forbidden_cost(self):
        accepted, reason = _accept_candidate(
            forbidden_before=2.0,
            forbidden_after=1.5,
            tau_f=1.0,
            stability_before=1.0,
            stability_after=1.01,
            constraint_tolerance=1e-5,
            stability_rise_tolerance=0.02,
            forbidden_decrease_tolerance=0.0,
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "constraint_repaired")

    def test_feasible_candidate_cannot_cross_constraint(self):
        accepted, reason = _accept_candidate(
            forbidden_before=0.5,
            forbidden_after=1.1,
            tau_f=1.0,
            stability_before=1.0,
            stability_after=0.9,
            constraint_tolerance=1e-5,
            stability_rise_tolerance=0.02,
            forbidden_decrease_tolerance=0.0,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "feasible_constraint_regression")


if __name__ == "__main__":
    unittest.main()
