import unittest

import numpy as np

from demo import borrower_top1_agreement
from near_tie import cases, ordered_pairs, rule_metrics, select_tau, selective_rate


class NearTieTests(unittest.TestCase):
    def test_90_ordered_pairs_match_unordered_agreement(self):
        fits = np.random.default_rng(9).normal(size=(10, 4, 3))
        frame = cases(fits, [[0], [1], [2]])
        self.assertEqual(len(ordered_pairs(10)), 90)
        self.assertEqual(len(frame['repeat']), 360)
        self.assertAlmostEqual(float(frame['repeat'].mean()),
                               borrower_top1_agreement(list(fits), [[0], [1], [2]]))

    def test_source_fit_alone_sets_gap_and_zero_is_excluded(self):
        fits = np.array([[[5., 0.]], [[1., .99]]])
        frame = cases(fits, [[0], [1]])
        self.assertAlmostEqual(frame['relative_gap'][frame['source_fit'] == 0][0], 1.)
        self.assertAlmostEqual(frame['relative_gap'][frame['source_fit'] == 1][0], .01)
        zero = cases(np.array([[[0., 0.]], [[1., 0.]]]), [[0], [1]])
        self.assertEqual(int(zero['valid'].sum()), 0)

    def test_threshold_is_smallest_grid_value_to_meet_target(self):
        frame = {'relative_gap': np.array([.1, .5, .95]),
                 'repeat': np.array([False, False, True]),
                 'valid': np.array([True, True, True])}
        tau, selection, grid = select_tau([frame])
        self.assertEqual(tau, .51)
        self.assertIn('smallest', selection)
        self.assertAlmostEqual(next(row for row in grid if row['tau'] == .5)['single_reason_repeat'], .5)

    def test_controls_are_exact_expectations(self):
        frame = {
            'valid': np.array([True, True, True, True]),
            'relative_gap': np.array([.9, .9, .1, .1]),
            'repeat': np.array([True, False, True, False]),
            'contain_two': np.array([True, True, True, False]),
        }
        score = rule_metrics(frame, .5, 3)
        self.assertAlmostEqual(score['coverage'], .5)
        self.assertAlmostEqual(score['baseline_repeat'], .5)
        self.assertAlmostEqual(score['single_reason_repeat'], .5)
        self.assertAlmostEqual(score['containment'], .5)
        self.assertAlmostEqual(score['random_subset_containment'], .625)
        self.assertAlmostEqual(score['random_second_containment'], .75)

    def test_selective_rate_uses_largest_gaps(self):
        frame = {'valid': np.array([True] * 4), 'relative_gap': np.array([.1, .9, .8, .2]),
                 'repeat': np.array([False, True, True, False]), 'borrower': np.arange(4),
                 'source_fit': np.zeros(4), 'replicate_fit': np.ones(4)}
        value, count = selective_rate(frame, .5)
        self.assertEqual(count, 2)
        self.assertEqual(value, 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
