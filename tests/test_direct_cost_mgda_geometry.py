import unittest

import torch

from audit_direct_cost_mgda_geometry import solve_mgda_active_set


class TestMGDAActiveSet(unittest.TestCase):
    def test_identity_has_uniform_interior_solution(self):
        solution = solve_mgda_active_set(torch.eye(3, dtype=torch.float64))
        self.assertEqual(solution["label"], "interior")
        self.assertTrue(torch.allclose(solution["alpha"], torch.full((3,), 1.0 / 3.0, dtype=torch.float64), atol=1e-10))

    def test_conflicting_edge_beats_the_third_vertex(self):
        # g_F=(1,0), g_S=(-1,0), g_K=(0,1).  The F-S midpoint is zero.
        gram = torch.tensor(
            [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        solution = solve_mgda_active_set(gram)
        self.assertEqual(solution["label"], "F-S")
        self.assertTrue(torch.allclose(solution["alpha"], torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64), atol=1e-10))
        self.assertAlmostEqual(solution["objective"], 0.0, places=12)

    def test_solution_is_feasible_and_not_worse_than_dense_simplex_scan(self):
        vectors = torch.tensor(
            [[1.0, 0.2], [-0.3, 1.1], [0.7, -0.8]],
            dtype=torch.float64,
        )
        gram = vectors @ vectors.T
        solution = solve_mgda_active_set(gram)
        alpha = solution["alpha"]
        self.assertGreaterEqual(float(alpha.min()), -1e-10)
        self.assertAlmostEqual(float(alpha.sum()), 1.0, places=10)

        best_grid = float("inf")
        for index in range(1001):
            alpha_f = index / 1000.0
            for second in range(1001 - index):
                alpha_s = second / 1000.0
                candidate = torch.tensor(
                    [alpha_f, alpha_s, 1.0 - alpha_f - alpha_s],
                    dtype=torch.float64,
                )
                best_grid = min(best_grid, float(candidate @ gram @ candidate))
        self.assertLessEqual(solution["objective"], best_grid + 2e-5)


if __name__ == "__main__":
    unittest.main()
