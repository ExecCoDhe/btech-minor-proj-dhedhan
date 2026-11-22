"""Trace one workbook row through the split, refit 0, the bill table and Figure 2.

The row is not chosen for its result. It is the borrower selected by the existing
Figure 2 rule: closest to the median cancellation in refit 0, then the lowest
row index. The published figure names that row as 2812.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from demo import (PILOT_SPLIT_SEED, ROOT, fit_one, first_reason_rows, groups_from_training,
                  linear_contributions, load_frame)
from report_figures import median_borrower

TRACE_ROW = 2812
MONTHS = {
    'BILL_AMT1': 'September', 'BILL_AMT2': 'August', 'BILL_AMT3': 'July',
    'BILL_AMT4': 'June', 'BILL_AMT5': 'May', 'BILL_AMT6': 'April',
}
PUBLISHED_FIGURE = {
    'September': 0.1182, 'July': -0.0739, 'net': 0.0388, 'magnitude': 0.2220, 'percent': '82.5%',
}


def signed(value, digits=4):
    return f'{value:+.{digits}f}'.replace('-', '\u2212')


def trace(bill_csv, output):
    frame, X, y = load_frame()
    failures = []

    def check(name, passed, detail):
        if not passed:
            failures.append(f'{name}: {detail}')
        return 'pass' if passed else 'fail'

    ids = frame['ID'].to_numpy()
    id_check = check('UCI ID mapping', np.array_equal(ids, np.arange(1, 30001)),
                     f'row {TRACE_ROW} has ID {ids[TRACE_ROW]}')
    row_id = int(ids[TRACE_ROW])
    train_ids, test_ids = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=PILOT_SPLIT_SEED, stratify=y)
    in_test = int(TRACE_ROW) in set(test_ids)
    in_train = int(TRACE_ROW) in set(train_ids)
    membership = check('split membership', in_test and not in_train,
                       f'train={in_train}, test={in_test}')
    audit_ids = np.random.default_rng(20).choice(test_ids, 500, replace=False)
    in_audit = int(TRACE_ROW) in set(audit_ids)
    audit_check = check('audit membership', in_audit, f'in audit={in_audit}')
    audit_position = int(np.flatnonzero(audit_ids == TRACE_ROW)[0]) if in_audit else None

    train = X.iloc[train_ids]
    groups, _ = groups_from_training(train)
    fitted = fit_one(X, y, train_ids, audit_ids, 0)
    if not set(fitted['sampled']).issubset(set(train_ids)):
        failures.append('bootstrap sample left the training rows')
    training_mean = train.mean().to_numpy()
    scale = fitted['prep'].named_steps['standardscaler'].scale_
    beta = fitted['linear'].coef_[0]
    raw = X.iloc[TRACE_ROW].to_numpy(dtype=float)
    hand = linear_contributions(raw, training_mean, scale, beta)
    model_phi = fitted['phi'][audit_position]
    formula_gap = float(np.max(np.abs(hand - model_phi)))
    formula_check = check('hand formula', formula_gap < 1e-8, f'max gap {formula_gap:.3e}')
    background_gap = float(np.max(np.abs(
        fitted['background'] - fitted['prep'].transform(train).mean(axis=0))))
    check('training background', background_gap < 1e-12, f'max gap {background_gap:.3e}')
    check('additivity', fitted['residual'] < 1e-8, f'residual {fitted["residual"]:.3e}')
    missing = int(X.isna().sum().sum())
    check('no missing model inputs', missing == 0, f'missing cells {missing}')

    reasons = first_reason_rows(fitted['phi'], groups)
    winner = int(reasons['winner'][audit_position])
    winner_net = float(reasons['winner_net'][audit_position])
    group_names = [[X.columns[i] for i in group] for group in groups]
    group_rows = []
    for index, group in enumerate(groups):
        net = float(fitted['phi'][audit_position, group].sum())
        group_rows.append((index, group_names[index], net, abs(net), index == winner))

    bills = pd.read_csv(bill_csv)
    chosen = median_borrower(bills)
    chosen_index = int(chosen['audit_row_index'])
    check('figure rule selects 2812', chosen_index == TRACE_ROW and int(chosen['refit']) == 0,
          f'selected row {chosen_index}, refit {int(chosen["refit"])}')
    saved = bills[(bills['refit'] == 0) & (bills['audit_row_index'] == TRACE_ROW)]
    check('one saved bill row', len(saved) == 1, f'rows found {len(saved)}')
    saved = saved.iloc[0]
    month_gaps = [abs(float(saved[month]) - float(model_phi[X.columns.get_loc(month)])) for month in MONTHS]
    check('saved bill contributions', max(month_gaps) < 1e-8, f'max gap {max(month_gaps):.3e}')
    net_gap = abs(float(saved['net']) - float(model_phi[[X.columns.get_loc(m) for m in MONTHS]].sum()))
    check('saved bill net', net_gap < 1e-8, f'gap {net_gap:.3e}')

    published_checks = {
        'September': float(saved['BILL_AMT1']),
        'July': float(saved['BILL_AMT3']),
        'net': float(saved['net']),
        'magnitude': float(saved['magnitude']),
    }
    for key, value in published_checks.items():
        rendered = f'{value:.4f}'
        expected = f'{PUBLISHED_FIGURE[key]:.4f}'
        check(f'figure value {key}', rendered == expected, f'{rendered} vs {expected}')
    check('figure percent', f'{float(saved["fraction"]):.1%}' == PUBLISHED_FIGURE['percent'],
          f'{float(saved["fraction"]):.1%}')

    direction = 'lowered' if winner_net < 0 else 'raised'
    lines = [
        '# Borrower trace: workbook row 2812',
        '',
        'This row is the borrower selected by the existing Figure 2 rule on the pilot split: '
        'among refit 0, the defined cancellation closest to that fit\'s median, breaking ties '
        'by the lowest row index. It was not chosen because the explanation looks favorable.',
        '',
        'The first reason below is the group with the largest absolute signed contribution. '
        'A negative contribution lowers predicted risk relative to the average training borrower. '
        'The audit fits no decision threshold, so this is not a reason for denial.',
        '',
        '## Checks',
        '',
        f'- UCI `ID` equals the zero-based row index plus one: {id_check}. Row 2812 has ID {row_id}.',
        f'- The row is in the seed-{PILOT_SPLIT_SEED} test set and not in the training set: {membership}.',
        f'- It is one of the 500 audit borrowers drawn with seed 20: {audit_check}.',
        f'- Refit 0 contributions match `beta_j / sd_j * (x_j - training mean_j)`: {formula_check} '
        f'(largest absolute gap {formula_gap:.3e}).',
        f'- The background uses the full training set after this fit\'s scaler. '
        f'Largest gap against that mean: {background_gap:.3e}.',
        f'- Contributions plus the baseline reconstruct the model log-odds. '
        f'Residual {fitted["residual"]:.3e}.',
        f'- Model inputs have {missing} missing cells, so the median imputer does not move this row.',
        f'- Bootstrap refit 0 resamples training rows only '
        f'({len(set(fitted["sampled"]))} distinct rows, all inside the training set).',
        '',
        '## This borrower in refit 0',
        '',
        f'- Outcome `default payment next month`: {int(y.iloc[TRACE_ROW])}.',
        f'- Position within the audit sample: {audit_position}.',
        f'- First reason: group {winner} ({", ".join(group_names[winner])}), '
        f'signed contribution {signed(winner_net)}. It {direction} predicted risk.',
        f'- Exact tie for the largest magnitude: {bool(reasons["tied"][audit_position])}.',
        '',
        '| Group, in partition order | Signed sum | Absolute sum | First reason |',
        '| --- | ---: | ---: | --- |',
    ]
    for index, names, net, magnitude, is_first in group_rows:
        lines.append(f'| {index}. {", ".join(names)} | {signed(net)} | {magnitude:.4f} | '
                     f'{"yes" if is_first else ""} |')
    lines.extend([
        '',
        '## Six monthly bills',
        '',
        'The saved `bill_cancellation.csv` row for refit 0 matches the recomputed contributions. '
        f'Cancellation is {float(saved["fraction"]):.1%}: total magnitude {float(saved["magnitude"]):.4f}, '
        f'signed total {signed(float(saved["net"]))}.',
        '',
        '| Month | Feature | Raw amount | Contribution to log-odds |',
        '| --- | --- | ---: | ---: |',
    ])
    for feature, month in MONTHS.items():
        column = X.columns.get_loc(feature)
        lines.append(f'| {month} | {feature} | {raw[column]:.0f} | {signed(float(model_phi[column]))} |')
    lines.extend([
        '',
        'These rounded contributions are the ones printed on Figure 2: September +0.1182, '
        'July −0.0739, combined +0.0388, total magnitude 0.2220, and 82.5% cancelled.',
        '',
        '## Result',
        '',
        'All checks passed.' if not failures else 'Checks failed:\n\n' + '\n'.join(f'- {item}' for item in failures),
        '',
    ])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    if failures:
        raise SystemExit('Borrower trace failed:\n' + '\n'.join(failures))
    print(f'Borrower trace passed and was written to {output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bill-csv', type=Path, required=True,
                        help='bill_cancellation.csv from a pilot audit run')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'experiments' / 'borrower-trace' / 'row-2812.md')
    args = parser.parse_args()
    trace(args.bill_csv, args.output)
