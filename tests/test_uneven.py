"""Unit tests for the Original Uneven baseline adapter."""

import unittest

import numpy as np

from baselines.uneven import _parse_path


class UnevenAdapterTests(unittest.TestCase):
    def test_parse_path_ignores_ros_logs(self):
        stdout = """
[ INFO] map: SO(2) --> RXS2 done.
PATH -1.0 2.0 0.3
PATH 0.0 2.5 0.4
"""
        path = _parse_path(stdout)
        self.assertIsInstance(path, np.ndarray)
        self.assertEqual(path.shape, (2, 3))
        np.testing.assert_allclose(path[0], [-1.0, 2.0, 0.3])

    def test_parse_path_requires_two_points(self):
        self.assertIsNone(_parse_path("PATH 1.0 2.0 0.0\n"))


if __name__ == "__main__":
    unittest.main()
