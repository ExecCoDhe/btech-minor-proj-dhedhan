"""Check a newly produced COMET-XAI Lite run against committed result contracts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from demo import PILOT_SPLIT_SEED

TOLERANCE = 1e-9
ORIGINAL_SEEDS = (20261027, 20261028, 20261029, 20261030, 20261031)
FRESH_SEEDS = (20261032, 20261033, 20261034, 20261035, 20261036)


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def close(name, actual, expected):
    difference = abs(float(actual) - float(expected))
    if difference > TOLERANCE:
        raise SystemExit(f'{name}: {actual} != {expected}; difference {difference} exceeds {TOLERANCE}')


def compare_csv(actual_path, expected_path, label):
    actual, expected = rows(actual_path), rows(expected_path)
    if len(actual) != len(expected):
        raise SystemExit(f'{label}: row count {len(actual)} != {len(expected)}')
    for number, (got, want) in enumerate(zip(actual, expected, strict=True), start=1):
        if not set(want).issubset(got):
            raise SystemExit(f'{label}: row {number} is missing archived columns')
        for field in want:
            if field in {'split_seed', 'split_role'}:
                if got[field] != want[field]:
                    raise SystemExit(f'{label}: row {number} {field} differs')
            else:
                close(f'{label}: row {number} {field}', got[field], want[field])


def validate_fresh(path):
    fresh = rows(path)
    seeds = tuple(int(row['split_seed']) for row in fresh)
    if seeds != FRESH_SEEDS:
        raise SystemExit(f'fresh results have seeds {seeds}, not {FRESH_SEEDS}')
    for row in fresh:
        for field in ('correlation_rbo', 'random_mean_rbo', 'correlation_top1',
                      'random_mean_top1', 'median_bill_cancellation'):
            value = float(row[field])
            if not 0 <= value <= 1:
                raise SystemExit(f'fresh {row["split_seed"]} has invalid {field}={value}')


def main(args):
    pilot = json.loads((args.pilot_dir / 'pilot.json').read_text(encoding='utf-8'))
    expected = json.loads(Path('results/pilot.json').read_text(encoding='utf-8'))
    if pilot['split_seed'] != PILOT_SPLIT_SEED:
        raise SystemExit('pilot split seed changed')
    if pilot['data_sha256'] != expected['data_sha256'] or pilot['groups'] != expected['groups']:
        raise SystemExit('pilot data hash or learned grouping changed')
    for field in ('mean_AP', 'mean_ROC_AUC', 'mean_Brier', 'group_RBO',
                  'matched_random_mean_RBO', 'group_minus_matched_random_RBO'):
        close(f'pilot {field}', pilot[field], expected[field])
    audit = json.loads((args.pilot_dir / 'borrower_audit.json').read_text(encoding='utf-8'))
    if audit['undefined_first_reason_rows'] != 0 or audit['tied_top_first_reason_rows'] != 0:
        raise SystemExit('the committed pilot contract expects no undefined or tied first reason')
    if audit['fit_pairs'] != 45 or audit['ordered_fit_pairs'] != 90:
        raise SystemExit('incorrect number of fit pairs')
    compare_csv(args.original_csv, 'experiments/repeated-splits/per_split.csv', 'original five splits')
    validate_fresh(args.fresh_csv)
    rule = rows(args.near_tie_dir / 'rule_metrics.csv')
    if tuple(int(row['split_seed']) for row in rule) != FRESH_SEEDS:
        raise SystemExit('near-tie output has the wrong fresh seeds')
    if not all(float(row['coverage']) >= .50 and float(row['single_minus_baseline']) >= .05
               for row in rule):
        raise SystemExit('fresh near-tie output does not satisfy its recorded result contract')
    print('Reproduction checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--original-csv', type=Path, required=True)
    parser.add_argument('--fresh-csv', type=Path, required=True)
    parser.add_argument('--near-tie-dir', type=Path, required=True)
    main(parser.parse_args())
