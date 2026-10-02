import unittest

import numpy as np

from .degradation import fit_knee_candidate


class KneeTests(unittest.TestCase):
    def test_insufficient_observations(self):
        result, curve = fit_knee_candidate(np.arange(50), np.ones(50))
        self.assertEqual(result["status"], "insufficient data")
        self.assertIsNone(curve)

    def test_accelerated_decline(self):
        cycles = np.arange(1, 1001, dtype=float)
        capacity = 1.1 - cycles * 0.00002 - np.maximum(0, cycles - 500) * 0.00015
        result, _ = fit_knee_candidate(cycles, capacity)
        self.assertEqual(result["status"], "candidate")
        self.assertAlmostEqual(result["knee_cycle"], 500, delta=15)


if __name__ == "__main__":
    unittest.main()
