import unittest

import numpy as np

from aoa.gates import empirical_threshold


class EmpiricalThresholdTests(unittest.TestCase):
    def test_tied_scores_cannot_accept_a_partial_tie(self):
        for correct in ([1, 0], [0, 1]):
            with self.subTest(correct=correct):
                self.assertTrue(np.isinf(empirical_threshold([1, 1], correct, 0.8)))

    def test_loosest_feasible_whole_tie_block(self):
        score = np.array([3, 3, 2, 2, 1])
        correct = np.array([1, 0, 1, 1, 0])
        threshold = empirical_threshold(score, correct, 0.7)
        self.assertEqual(threshold, 2)
        self.assertGreaterEqual(correct[score >= threshold].mean(), 0.7)


if __name__ == "__main__":
    unittest.main()
