import csv
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import demo
from demo import (ROOT, borrower_top1_agreement, borrower_top1_summary, first_reason_rows,
                  fit_one, group_cancellation, group_magnitudes, groups_from_training, kth_margin,
                  linear_contributions, load_attributions, matched_random_partitions, mean_stability,
                  pack_groupings, rbo, unpack_groupings)

class Contracts(unittest.TestCase):
    def test_top_group_agreement_within_borrowers(self):
        # For borrower 0 the second group wins twice. For borrower 1 the
        # first group always wins, including an exact tie in the first fit.
        fits = [
            np.array([[5, -4.8, -1], [-2, -1, 3]]),
            np.array([[5, -3.5, -1], [-2, -1, 2.5]]),
            np.array([[5, -4.5, -1], [-2, -1, 1]]),
        ]
        self.assertAlmostEqual(borrower_top1_agreement(fits, [[0, 1], [2]]), (1 / 3 + 1) / 2)

    def test_opposing_contributions_cancel(self):
        net, magnitude, fraction = group_cancellation([5.0, -4.8])
        self.assertAlmostEqual(net, 0.2)
        self.assertAlmostEqual(magnitude, 9.8)
        self.assertAlmostEqual(fraction, 48 / 49)

    def test_positive_contributions_have_zero_cancellation(self):
        self.assertEqual(group_cancellation([2.0, 3.0]), (5.0, 5.0, 0.0))

    def test_all_zero_contributions_have_undefined_cancellation(self):
        self.assertEqual(group_cancellation([0.0, 0.0]), (0.0, 0.0, None))

    def test_identical_rbo(self):
        for n in [1,3,5,19]:
            self.assertAlmostEqual(rbo(list(range(n)),list(range(n))),1)
    def test_hand_computed_rbo(self):
        self.assertAlmostEqual(rbo(list('abc'),list('acb')),0.93625)
    def test_rbo_rejects_unequal_or_empty(self):
        for a,b in [([],[]),([1],[1,2]),([1,1],[1,1])]:
            with self.assertRaises(ValueError):rbo(a,b)
    def test_complete_linkage_within_group(self):
        rng=np.random.default_rng(8);a=rng.normal(size=1000)
        X=pd.DataFrame({'a':a,'b':a+.02*rng.normal(size=1000),'c':rng.normal(size=1000)})
        groups,corr=groups_from_training(X)
        self.assertIn([0,1],groups)
        for g in groups:self.assertGreaterEqual(np.abs(corr[np.ix_(g,g)]).min(),.7)
    def test_quantile_off_by_one_regression(self):
        self.assertEqual(kth_margin(np.arange(20),.2),16)
    def test_small_sample_margin_is_infinite(self):
        self.assertTrue(np.isinf(kth_margin([.1,.2],.1)))
    def test_negative_shift_clipped(self):
        self.assertEqual(kth_margin([-2,-1,0,1],.5),0)
    def test_owen_not_feature_sum(self):
        # Enumerate all orders independently for a unanimity game.
        import itertools
        values=[]
        for order in itertools.permutations('abc'):
            present=set();credits={x:0 for x in 'abc'}
            for player in order:
                before=int(len(present)==3);present.add(player)
                credits[player]=int(len(present)==3)-before
            values.append(credits['a']+credits['b'])
        grouped=[]
        for order in itertools.permutations(['ab','c']):
            present=set();credits={x:0 for x in ['ab','c']}
            for player in order:
                before=int(len(present)==3);present.update(player)
                credits[player]=int(len(present)==3)-before
            grouped.append(credits['ab'])
        self.assertAlmostEqual(np.mean(values),2/3)
        self.assertAlmostEqual(np.mean(grouped),1/2)
    def test_linear_shap_additivity(self):
        beta=np.array([3.,2.]);x=np.array([2.,5.]);base=np.array([1.,2.])
        phi=beta*(x-base)
        np.testing.assert_allclose(phi,[3,6])
        self.assertAlmostEqual(float(beta@base+phi.sum()),float(beta@x))

    def test_absolute_value_is_after_the_signed_sum(self):
        magnitudes = group_magnitudes(np.array([[5.0, -4.0, 0.2]]), [[0, 1], [2]])
        np.testing.assert_allclose(magnitudes, [[1.0, 0.2]])

    def test_hand_contribution_uses_the_training_mean(self):
        values = np.array([2.0, 5.0])
        training_mean = np.array([1.0, 2.0])
        scale = np.array([2.0, 4.0])
        beta = np.array([3.0, 2.0])
        np.testing.assert_allclose(linear_contributions(values, training_mean, scale, beta), [1.5, 1.5])

    def test_random_partitions_keep_sizes_and_the_historical_stream(self):
        sizes = [6, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1]
        partitions = matched_random_partitions(sizes, 19, 30, 101)
        self.assertEqual(len(partitions), 30)
        rng = np.random.default_rng(101)
        cuts = np.cumsum(sizes)[:-1]
        for partition in partitions:
            self.assertEqual(sorted(len(group) for group in partition), sorted(sizes))
            self.assertEqual(sorted(index for group in partition for index in group), list(range(19)))
            expected = [part.tolist() for part in np.split(rng.permutation(19), cuts)]
            self.assertEqual(partition, expected)

    def test_ten_fits_cover_45_unordered_and_90_ordered_pairs(self):
        fits = [np.random.default_rng(0).normal(size=(4, 3)) for _ in range(10)]
        groups = [[0], [1], [2]]
        winners = [group_magnitudes(phi, groups).argmax(axis=1) for phi in fits]
        unordered = list(combinations(winners, 2))
        ordered = [(i, j) for i in range(10) for j in range(10) if i != j]
        self.assertEqual(len(unordered), 45)
        self.assertEqual(len(ordered), 90)
        published = borrower_top1_agreement(fits, groups)
        self.assertAlmostEqual(published, float(np.mean([np.mean(a == b) for a, b in unordered])))
        self.assertAlmostEqual(published, float(np.mean([np.mean(winners[i] == winners[j]) for i, j in ordered])))

    def test_zero_row_is_not_a_first_reason_and_a_tie_keeps_partition_order(self):
        detail = first_reason_rows(np.zeros((1, 3)), [[0, 1], [2]])
        self.assertEqual(detail['winner'][0], -1)
        self.assertTrue(detail['undefined'][0])
        tied = first_reason_rows(np.array([[-2.0, -1.0, 3.0]]), [[0, 1], [2]])
        self.assertEqual(tied['winner'][0], 0)
        self.assertTrue(tied['tied'][0])
        self.assertAlmostEqual(tied['winner_net'][0], -3.0)
        fits = [np.array([[0.0, 0.0], [5.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 4.0]])]
        summary = borrower_top1_summary(fits, [[0], [1]])
        self.assertEqual(summary['undefined_first_reason_rows'], 2)
        self.assertAlmostEqual(summary['agreement'], 0.5)
        self.assertAlmostEqual(summary['agreement_defined_only'], 0.0)
        self.assertAlmostEqual(summary['top_group_lowers_risk_share'], 0.0)

    def test_packed_groupings_round_trip(self):
        groupings = [[[0, 1, 2], [3]], [[4], [5, 6]]]
        self.assertEqual(unpack_groupings(*pack_groupings(groupings)), groupings)

    def test_readme_tables_match_the_committed_csv(self):
        committed = ROOT / 'experiments' / 'repeated-splits' / 'per_split.csv'
        with committed.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        repeated = (ROOT / 'experiments' / 'repeated-splits' / 'README.md').read_text(encoding='utf-8')
        root = (ROOT / 'README.md').read_text(encoding='utf-8')
        for row in rows:
            delta_rbo = f"{float(row['delta_rbo']):+.4f}".replace('-', '\u2212')
            delta_top1 = f"{float(row['delta_top1_pp']):+.3f}".replace('-', '\u2212')
            role = ' (pilot)' if row['split_role'] == 'pilot' else ''
            rendered = (
                f"| {row['split_seed']}{role} | {float(row['correlation_rbo']):.4f} | "
                f"{float(row['random_mean_rbo']):.4f} | {delta_rbo} | "
                f"{float(row['correlation_top1']):.3%} | {float(row['random_mean_top1']):.3%} | "
                f"{delta_top1} | {float(row['median_bill_cancellation']):.1%} |"
            )
            self.assertIn(rendered, repeated)
        self.assertIn('78.5%', root)
        self.assertIn('80.6%', root)
        self.assertIn('85.6%', root)
        self.assertIn('90.7%', root)

    @unittest.skipUnless((ROOT / 'data' / 'default_of_credit_card_clients.xls').exists(),
                         'workbook not downloaded')
    def test_split_background_and_saved_attributions_use_training_rows(self):
        seen = {}
        real_groups = groups_from_training

        def record_groups(X, rho=0.7):
            seen['rows'] = len(X)
            seen['index'] = X.index.to_numpy().copy()
            return real_groups(X, rho)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'pilot'
            with mock.patch.object(demo, 'groups_from_training', record_groups):
                result = demo.run(bootstraps=2, random_partitions=2, borrower_audit=True, output_dir=output)
            with np.load(output / 'splits.npz') as splits:
                train, test, audit = splits['train'].copy(), splits['test'].copy(), splits['audit'].copy()
            self.assertEqual(len(set(train) & set(test)), 0)
            self.assertTrue(set(audit).issubset(set(test)))
            self.assertEqual(len(audit), 500)
            self.assertEqual(seen['rows'], 24000)
            self.assertTrue(set(seen['index']).issubset(set(train)))
            self.assertEqual(len(set(seen['index']) & set(test)), 0)
            frame = pd.read_excel(ROOT / 'data' / 'default_of_credit_card_clients.xls', header=1)
            y = frame['default payment next month'].astype(int)
            X = frame.drop(columns=['ID', 'SEX', 'MARRIAGE', 'AGE', 'EDUCATION', 'default payment next month'])
            fitted = fit_one(X, y, train, audit, 0)
            self.assertTrue(set(fitted['sampled']).issubset(set(train)))
            training_background = fitted['prep'].transform(X.iloc[train]).mean(axis=0)
            np.testing.assert_allclose(fitted['background'], training_background)
            test_background = fitted['prep'].transform(X.iloc[test]).mean(axis=0)
            self.assertFalse(np.allclose(fitted['background'], test_background))
            stored, stored_ids, groups, partitions = load_attributions(output / 'attributions.npz')
            np.testing.assert_array_equal(stored_ids, audit)
            self.assertEqual([[X.columns[i] for i in group] for group in groups], result['groups'])
            self.assertEqual(len(partitions), 2)
            self.assertAlmostEqual(mean_stability(list(stored), groups), result['group_RBO'])
            self.assertAlmostEqual(mean_stability(list(stored), partitions[0]), result['matched_random_scores'][0])


if __name__=='__main__':unittest.main(verbosity=2)
