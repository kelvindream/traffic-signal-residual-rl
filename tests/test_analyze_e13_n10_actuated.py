import unittest

import numpy as np

from tools.analyze_e13_n10_actuated import (
    claim_eligible,
    hierarchical_bootstrap_ci,
    holm_adjust,
    oriented_effect,
    seed_bootstrap_ci,
)


class AnalyzeE13Tests(unittest.TestCase):
    def test_effect_orientation(self):
        self.assertEqual(oriented_effect(2.0, "throughput"), 2.0)
        self.assertEqual(oriented_effect(-2.0, "throughput"), -2.0)
        self.assertEqual(oriented_effect(2.0, "unfinished"), -2.0)
        self.assertEqual(oriented_effect(-2.0, "total_waiting_increment_s"), 2.0)

    def test_holm_known_values(self):
        adjusted = holm_adjust([0.001, 0.01, 0.04, 0.20])
        np.testing.assert_allclose(adjusted, [0.004, 0.03, 0.08, 0.20])

    def test_seed_bootstrap_is_deterministic(self):
        values = np.arange(10, dtype=float)
        self.assertEqual(
            seed_bootstrap_ci(values, n_resamples=2000, rng_seed=7),
            seed_bootstrap_ci(values, n_resamples=2000, rng_seed=7),
        )

    def test_hierarchical_bootstrap_shape_and_direction(self):
        matrix = np.ones((10, 100), dtype=float) * 3.0
        interval = hierarchical_bootstrap_ci(matrix, n_resamples=1000, rng_seed=9)
        np.testing.assert_allclose(interval, [3.0, 3.0])

    def test_claim_requires_all_three_conditions(self):
        self.assertTrue(claim_eligible(0.01, [0.1, 1.0], [0.2, 0.9]))
        self.assertFalse(claim_eligible(0.06, [0.1, 1.0], [0.2, 0.9]))
        self.assertFalse(claim_eligible(0.01, [-0.1, 1.0], [0.2, 0.9]))
        self.assertFalse(claim_eligible(0.01, [0.1, 1.0], [-0.2, 0.9]))


if __name__ == "__main__":
    unittest.main()
