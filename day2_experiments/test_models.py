import unittest

import numpy as np

from day2_experiments.features import early_features, make_target
from day2_experiments.evaluation import metrics


class FeatureSafetyTests(unittest.TestCase):
    def summary(self):
        return {
            "cycle": np.arange(1, 8),
            "QDischarge": np.arange(1, 8, dtype=float),
            "IR": np.full(7, 0.02),
            "Tavg": np.full(7, 30.0),
            "Tmax": np.full(7, 35.0),
            "chargetime": np.full(7, 10.0),
        }

    def test_target_boundaries(self):
        self.assertEqual(make_target(499), 1)
        self.assertEqual(make_target(1001), 0)
        for value in [500, 1000, 700, np.nan, np.inf, -1, 0]:
            self.assertTrue(np.isnan(make_target(value)))

    def test_future_values_do_not_change_features(self):
        summary = self.summary()
        before, complete = early_features(summary)
        self.assertTrue(complete)
        for key in summary:
            if key != "cycle":
                summary[key][5:] = 999999.0
        after, _ = early_features(summary)
        self.assertEqual(before, after)
        self.assertEqual(before["qd_mean_2_5"], 3.5)
        self.assertEqual(before["qd_delta_5_2"], 3.0)

    def test_cycle_numbers_not_array_positions(self):
        summary = self.summary()
        shuffled = {key: value[::-1] for key, value in summary.items()}
        self.assertEqual(early_features(summary), early_features(shuffled))
        summary["cycle"][2] = 2
        self.assertFalse(early_features(summary)[1])

    def test_missing_resistance_and_undefined_batch_recall(self):
        summary = self.summary()
        summary["IR"][:] = 0
        features, complete = early_features(summary)
        self.assertTrue(complete)
        self.assertTrue(np.isnan(features["ir_mean_2_5"]))
        scores = metrics([0, 0], [0, 1])
        self.assertTrue(np.isnan(scores["recall_short"]))
        self.assertTrue(np.isnan(scores["balanced_accuracy"]))
        self.assertEqual(scores["fp"], 1)


if __name__ == "__main__":
    unittest.main()
