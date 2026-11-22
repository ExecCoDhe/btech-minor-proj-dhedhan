"""Predefined one-or-two-reason analysis for the COMET-XAI Lite audit."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from demo import first_reason_rows, load_attributions

TAU_GRID = tuple(i / 100 for i in range(100))
FRESH_SEEDS = (20261032, 20261033, 20261034, 20261035, 20261036)
COVERAGES = (0.25, 0.50, 0.75, 1.0)


def ordered_pairs(n_fits):
    return [(a, b) for a in range(n_fits) for b in range(n_fits) if a != b]


def cases(attributions, groups):
    """One row per borrower and ordered fit pair; only source-fit data set the gap."""
    details = [first_reason_rows(phi, groups) for phi in attributions]
    output = {key: [] for key in ('borrower', 'source_fit', 'replicate_fit',
                                  'relative_gap', 'absolute_gap', 'valid',
                                  'repeat', 'contain_two')}
    for source, replicate in ordered_pairs(len(details)):
        left, right = details[source], details[replicate]
        largest = left['largest']
        relative = np.full(len(largest), np.nan)
        positive = largest > 0
        relative[positive] = ((largest[positive] - left['second_largest'][positive])
                              / largest[positive])
        output['borrower'].append(np.arange(len(largest)))
        output['source_fit'].append(np.full(len(largest), source))
        output['replicate_fit'].append(np.full(len(largest), replicate))
        output['relative_gap'].append(relative)
        output['absolute_gap'].append(largest - left['second_largest'])
        output['valid'].append(~left['undefined'] & ~right['undefined'])
        output['repeat'].append(left['winner'] == right['winner'])
        output['contain_two'].append((right['winner'] == left['winner'])
                                     | (right['winner'] == left['second']))
    return {key: np.concatenate(value) for key, value in output.items()}


def rate(values):
    return float(np.mean(values)) if len(values) else None


def auc(score, outcome, valid):
    valid_score, valid_outcome = score[valid], outcome[valid]
    if not len(valid_score) or len(np.unique(valid_outcome)) < 2:
        return None
    return float(roc_auc_score(valid_outcome, valid_score))


def selective_rate(frame, coverage):
    index = np.flatnonzero(frame['valid'])
    if not len(index):
        return None, 0
    n = len(index) if coverage == 1 else int(round(coverage * len(index)))
    # Deterministic tie-breaking matters only at a coverage boundary.
    order = np.lexsort((frame['replicate_fit'][index], frame['source_fit'][index],
                        frame['borrower'][index], -frame['relative_gap'][index]))
    chosen = index[order[:n]]
    return rate(frame['repeat'][chosen]), int(n)


def rule_metrics(frame, tau, n_groups):
    """Evaluate the frozen rule and exact matched controls."""
    valid = frame['valid']
    n = int(valid.sum())
    if not n or n_groups < 2:
        raise ValueError('Need defined cases and at least two groups')
    one = valid & (frame['relative_gap'] >= tau)
    coverage = int(one.sum()) / n
    baseline = rate(frame['repeat'][valid])
    single = rate(frame['repeat'][one])
    containment = rate(np.where(one, frame['repeat'], frame['contain_two'])[valid])
    # Expected containment if the same fraction of cases receives two reasons,
    # but those cases are chosen independently of their source-fit gap.
    random_subset = coverage * baseline + (1 - coverage) * rate(frame['contain_two'][valid])
    random_second = baseline + (1 - baseline) / (n_groups - 1)
    return {
        'n_cases': n, 'n_excluded_undefined': int((~valid).sum()), 'tau': tau,
        'n_groups': n_groups, 'coverage': coverage, 'two_reason_rate': 1 - coverage,
        'single_reason_repeat': single, 'baseline_repeat': baseline,
        'single_minus_baseline': single - baseline,
        'containment': containment, 'random_subset_containment': random_subset,
        'containment_minus_random_subset': containment - random_subset,
        'random_second_containment': random_second,
        'containment_minus_random_second': containment - random_second,
    }


def select_tau(frames):
    relative = np.concatenate([frame['relative_gap'] for frame in frames])
    repeat = np.concatenate([frame['repeat'] for frame in frames])
    valid = np.concatenate([frame['valid'] for frame in frames])
    rows, chosen = [], None
    for tau in TAU_GRID:
        selected = valid & (relative >= tau)
        row = {'tau': tau, 'n_selected': int(selected.sum()),
               'coverage': int(selected.sum()) / int(valid.sum()),
               'single_reason_repeat': rate(repeat[selected])}
        rows.append(row)
        if chosen is None and row['single_reason_repeat'] >= .90:
            chosen = tau
    if chosen is None:
        chosen, selection = float(np.median(relative[valid])), (
            'pooled median of the relative gap because no grid value reached 90 percent')
    else:
        selection = 'smallest grid value whose pooled single-reason repeat rate is at least 90 percent'
    return chosen, selection, rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def split_dirs(path):
    return sorted((item for item in Path(path).iterdir() if item.is_dir() and item.name.isdigit()),
                  key=lambda item: int(item.name))


def read_split(path):
    attributions, _, groups, partitions = load_attributions(path / 'attributions.npz')
    return groups, cases(attributions, groups), [cases(attributions, partition) for partition in partitions]


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def write_protocol(path, protocol, grid):
    chosen = next(row for row in grid if row['tau'] == protocol['tau'])
    text = f"""# Protocol for the one-or-two reason rule

This protocol was written and committed before the fresh splits were run.
Seeds 20261027–20261031 were exploratory only. The fresh evaluation uses
20261032–20261036.

## Definitions

Group score is the absolute signed sum. For each source fit, `m1` and `m2` are
the largest and second-largest scores; ties keep partition order. The relative
gap is `(m1 - m2) / m1`; an all-zero row is excluded and counted. Each case is
one of 90 ordered pairs from ten fits. The source fit alone decides what the
borrower sees. The second fit is the replicate.

## Rule frozen from exploratory data

Show one reason when its relative gap is at least {protocol['tau']:.2f};
otherwise show the leading two groups. This threshold was selected as the
{protocol['tau_selection_rule']}. On exploratory cases it showed one reason
in {chosen['coverage']:.1%} of cases, with {chosen['single_reason_repeat']:.1%}
repeat agreement.

## Fresh measures and controls

For each fresh split report: one-reason coverage, its repeat rate `R1`,
two-reason rate, containment of the replicate first reason, and all-case
repeat rate `B`. The primary same-coverage comparison is `R1 - B`, because a
random subset at the same coverage has expected repeat rate `B`.

Also report containment against (1) a random subset receiving two reasons and
(2) the leading reason plus a uniformly random other group. Report selective
repeat rates at 25, 50, 75 and 100 percent coverage for correlation groups
and the mean of the 30 size-matched random partitions.

The rule is a meaningful gain only if `R1 - B` is at least 5 percentage points
at coverage at least 50 percent on at least 4 of the 5 fresh splits. Top-two
containment alone is not an improvement claim. Fresh splits reuse one source
population and can share borrowers, so results are descriptive.

The audit seed remains 20; bootstrap seeds remain 0–9; random partitions
remain the 30 draws from seed 101.
"""
    path.write_text(text, encoding='utf-8')


def explore(splits_dir, output):
    bins, auc_rows, selective, frames = [], [], [], []
    for directory in split_dirs(splits_dir):
        groups, correlation, randoms = read_split(directory)
        frames.append(correlation)
        for name, frame in [('correlation', correlation), ('random_mean', None)]:
            if frame is None:
                gap_auc = mean_or_none(auc(item['relative_gap'], item['repeat'], item['valid']) for item in randoms)
                absolute_auc = mean_or_none(auc(item['absolute_gap'], item['repeat'], item['valid']) for item in randoms)
            else:
                gap_auc = auc(frame['relative_gap'], frame['repeat'], frame['valid'])
                absolute_auc = auc(frame['absolute_gap'], frame['repeat'], frame['valid'])
            auc_rows.append({'analysis': 'exploratory', 'split_seed': int(directory.name), 'method': name,
                             'relative_gap_auc': gap_auc, 'absolute_gap_auc': absolute_auc})
        for coverage in COVERAGES:
            correlation_rate, n = selective_rate(correlation, coverage)
            random_results = [selective_rate(frame, coverage) for frame in randoms]
            selective.extend([
                {'analysis': 'exploratory', 'split_seed': int(directory.name), 'method': 'correlation',
                 'coverage': coverage, 'n_selected': n, 'repeat_rate': correlation_rate},
                {'analysis': 'exploratory', 'split_seed': int(directory.name), 'method': 'random_mean',
                 'coverage': coverage, 'n_selected': mean_or_none(item[1] for item in random_results),
                 'repeat_rate': mean_or_none(item[0] for item in random_results)},
            ])
        for low in np.arange(0, 1, .1):
            high = round(low + .1, 1)
            for name, frames_to_average in [('correlation', [correlation]), ('random_mean', randoms)]:
                per_partition = []
                for frame in frames_to_average:
                    select = frame['valid'] & (frame['relative_gap'] >= low)
                    select &= frame['relative_gap'] <= high if high == 1 else frame['relative_gap'] < high
                    per_partition.append((int(select.sum()), rate(frame['repeat'][select])))
                bins.append({'analysis': 'exploratory', 'split_seed': int(directory.name), 'method': name,
                             'bin_low': low, 'bin_high': high,
                             'n_cases': mean_or_none(row[0] for row in per_partition),
                             'repeat_rate': mean_or_none(row[1] for row in per_partition)})
    tau, selection, grid = select_tau(frames)
    for row in grid:
        row.update(analysis='exploratory', selected=row['tau'] == tau)
    output = Path(output)
    write_csv(output / 'gap_bins.csv', bins)
    write_csv(output / 'gap_auc.csv', auc_rows)
    write_csv(output / 'selective_coverage.csv', selective)
    write_csv(output / 'tau_grid.csv', grid)
    protocol = {
        'status': 'frozen before fresh runs', 'tau': tau, 'tau_selection_rule': selection,
        'fresh_split_seeds': list(FRESH_SEEDS), 'reliability_target': .90,
        'meaningful_gain': 'R1 - B >= 5 percentage points at >=50% coverage on >=4 of 5 fresh splits',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    write_protocol(output.parent / 'PROTOCOL.md', protocol, grid)
    print(f'Selected tau {tau:.2f}: {selection}')


def evaluate(splits_dir, protocol_path, output):
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    expected = protocol['fresh_split_seeds']
    directories = split_dirs(splits_dir)
    if [int(directory.name) for directory in directories] != expected:
        raise SystemExit(f'Expected fresh seeds {expected}')
    rows, selective = [], []
    for directory in directories:
        groups, correlation, randoms = read_split(directory)
        metric = rule_metrics(correlation, protocol['tau'], len(groups))
        metric.update(analysis='confirmatory', split_seed=int(directory.name),
                      gain_on_this_split=(metric['single_minus_baseline'] >= .05 and metric['coverage'] >= .50))
        rows.append(metric)
        for coverage in COVERAGES:
            correlation_rate, n = selective_rate(correlation, coverage)
            random_results = [selective_rate(frame, coverage) for frame in randoms]
            selective.extend([
                {'analysis': 'confirmatory', 'split_seed': int(directory.name), 'method': 'correlation',
                 'coverage': coverage, 'n_selected': n, 'repeat_rate': correlation_rate},
                {'analysis': 'confirmatory', 'split_seed': int(directory.name), 'method': 'random_mean',
                 'coverage': coverage, 'n_selected': mean_or_none(item[1] for item in random_results),
                 'repeat_rate': mean_or_none(item[0] for item in random_results)},
            ])
    output = Path(output)
    write_csv(output / 'rule_metrics.csv', rows)
    write_csv(output / 'selective_coverage.csv', selective)
    verdict = {'tau': protocol['tau'], 'splits_meeting_gain_rule': int(sum(row['gain_on_this_split'] for row in rows)),
               'meaningful_gain': int(sum(row['gain_on_this_split'] for row in rows)) >= 4,
               'rule': protocol['meaningful_gain']}
    (output / 'verdict.json').write_text(json.dumps(verdict, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('explore', 'evaluate'):
        item = sub.add_parser(command)
        item.add_argument('--splits-dir', type=Path, required=True)
        item.add_argument('--output', type=Path, required=True)
        if command == 'evaluate':
            item.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'explore':
        explore(args.splits_dir, args.output)
    else:
        evaluate(args.splits_dir, args.protocol, args.output)
