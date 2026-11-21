"""Repeat the existing COMET-XAI Lite audit on five predefined train/test splits."""
import argparse
import csv
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import mkdtemp

from demo import ROOT, PILOT_SPLIT_SEED, run


SPLIT_SEEDS = (PILOT_SPLIT_SEED, 20261028, 20261029, 20261030, 20261031)


def repeat_splits(output_dir=None):
    if output_dir is None:
        runs = ROOT / 'runs'
        runs.mkdir(exist_ok=True)
        output_dir = Path(mkdtemp(prefix='repeated-splits-', dir=runs))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    print(f'Saving split audits to {output_dir.resolve()}', flush=True)

    rows = []
    for seed in SPLIT_SEEDS:
        split_dir = output_dir / str(seed)
        with (output_dir / f'{seed}.log').open('w') as log, redirect_stdout(log):
            pilot = run(bootstraps=10, random_partitions=30, split_seed=seed,
                        borrower_audit=True, output_dir=split_dir)
        audit = json.loads((split_dir / 'borrower_audit.json').read_text())
        top1 = audit['top1_agreement_correlation_groups']
        random_top1 = audit['top1_agreement_size_matched_random_mean']
        rows.append({
            'split_seed': seed,
            'split_role': 'pilot' if seed == PILOT_SPLIT_SEED else 'new',
            'correlation_rbo': pilot['group_RBO'],
            'random_mean_rbo': pilot['matched_random_mean_RBO'],
            'delta_rbo': pilot['group_minus_matched_random_RBO'],
            'correlation_top1': top1,
            'random_mean_top1': random_top1,
            'delta_top1_pp': 100 * (top1 - random_top1),
            'median_bill_cancellation': audit['median_bill_cancellation_fraction'],
        })
        print(f'{seed}  ranking overlap {rows[-1]["delta_rbo"]:+.4f}'
              f'  first-reason agreement {rows[-1]["delta_top1_pp"]:+.2f} percentage points',
              flush=True)

    path = output_dir / 'per_split.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved per-split results to {path.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='New directory for the five audits and results table')
    repeat_splits(parser.parse_args().output_dir)
