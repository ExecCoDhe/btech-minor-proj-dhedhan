"""Render report figures from machine-readable COMET-XAI Lite audit results."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory


INK = '#182E40'
MUTED = '#475E6A'
TEAL = '#006D70'
SLATE = '#748498'
RUST = '#AA5128'
GRID = '#D6DFE2'
STYLE = {'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
         'axes.spines.top': False, 'axes.spines.right': False,
         'axes.spines.bottom': False, 'axes.spines.left': False,
         'xtick.major.size': 0, 'ytick.major.size': 0,
         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK}


def rule(fig, x0, x1, y0, y1):
    fig.add_artist(Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                          color=GRID, linewidth=0.8))


def page(number, topic, title, subtitle):
    fig = Figure(figsize=(9.4, 5.8), facecolor='white')
    fig.text(0.055, 0.937, f'FIGURE {number}   /   {topic}', size=10, weight='bold', color=TEAL)
    fig.text(0.945, 0.937, 'COMET-XAI Lite', size=10, ha='right', color=MUTED)
    fig.text(0.055, 0.840, title, size=22, family='DejaVu Serif', color=INK)
    fig.text(0.055, 0.775, subtitle, size=11, color=MUTED)
    rule(fig, 0.055, 0.945, 0.735, 0.735)
    rule(fig, 0.055, 0.945, 0.156, 0.156)
    fig.text(0.055, 0.035, 'Source  ·  I-Cheng Yeh, Default of Credit Card Clients (UCI 350).',
             size=10, color=MUTED)
    return fig


def signed(value, digits):
    return f'{value:+.{digits}f}'.replace('-', '−')


def agreement_figure(pilot, audit):
    scores = (pilot['group_RBO'], pilot['matched_random_mean_RBO'])
    rates = (audit['top1_agreement_correlation_groups'], audit['top1_agreement_size_matched_random_mean'])
    title = ('Correlation groups gave a steadier overall ranking'
             if scores[0] > scores[1] else 'How consistent are explanations across refits?')
    fig = page(1, 'EXPLANATION STABILITY', title,
               'The model was refit on resampled data. Higher values mean closer agreement across fits.')
    rule(fig, 0.50, 0.50, 0.19, 0.70)
    panels = [
        (0.055, 'Overall ranking', 'Rank overlap score (0 to 1)', scores, '.3f',
         f'{signed(scores[0] - scores[1], 3)} in ranking overlap'),
        (0.535, 'Each borrower’s first reason', 'Share of comparisons with the same group first', rates, '.1%',
         f'{signed(100 * (rates[0] - rates[1]), 1)} percentage points'),
    ]
    for left, heading, unit, values, label_format, difference in panels:
        fig.text(left, 0.669, heading, size=14, weight='bold', color=INK)
        fig.text(left, 0.617, unit, size=10, color=MUTED)
        ax = fig.add_axes([left + 0.055, 0.327, 0.348, 0.232])
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=0.65)
        ax.set_ylim(0, 1.07)
        ax.set_yticks([0, 0.5, 1], ['0', '0.5', '1'] if label_format == '.3f' else ['0%', '50%', '100%'])
        ax.bar([0, 1], values, width=0.53, color=[TEAL, SLATE], zorder=2)
        ax.set_xlim(-0.65, 1.65)
        ax.set_xticks([0, 1], ['Correlation groups', 'Random groups'], size=10)
        for x, value in enumerate(values):
            ax.annotate(format(value, label_format), (x, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', size=13, weight='bold', color=INK)
        fig.text(left + 0.055, 0.214, difference, color=TEAL if values[0] >= values[1] else INK,
                 size=13, weight='bold')
    fig.text(0.055, 0.111, f'{audit["audit_borrowers"]:,} held-out borrowers · {audit["bootstrap_fits"]} refits'
             f' · {audit["fit_pairs"]} fit pairs · One train/test split', size=10, color=INK)
    fig.text(0.055, 0.073, f'Random bars average {audit["random_partitions"]} partitions with the same group sizes.'
             ' The rank overlap score gives more weight to the top ranks.', size=9, color=MUTED)
    return fig


def median_borrower(bills):
    """Find a borrower closest to the median defined fraction in fixed refit 0."""
    eligible = bills.loc[bills['refit'].eq(0)].dropna(subset=['fraction'])
    if eligible.empty:
        raise ValueError('Figure 2 requires a defined bill cancellation fraction in refit 0')
    ordered = eligible['fraction'].sort_values()
    # With an even count, both central values are equally close to the median.
    middle = ordered.iloc[[(len(ordered) - 1) // 2, len(ordered) // 2]]
    candidates = eligible.loc[eligible['fraction'].isin(middle)]
    return candidates.sort_values('audit_row_index').iloc[0]


def cancellation_figure(bills, audit):
    row = median_borrower(bills)
    fig = page(2, 'BILL HISTORY', 'Why six monthly contributions can add up to little',
               'Relative to the average training borrower, rightward bars raised predicted default risk; leftward bars lowered it.')
    ax = fig.add_axes([0.17, 0.272, 0.485, 0.415])
    labels = ['April', 'May', 'June', 'July', 'August', 'September', 'Combined']
    values = [row[f'BILL_AMT{i}'] for i in range(6, 0, -1)] + [row['net']]
    positions = [0, 1, 2, 3, 4, 5, 6.45]
    colors = [TEAL if value >= 0 else RUST for value in values[:-1]] + [INK]
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.axhline(5.75, color=GRID, linewidth=0.8)
    ax.barh(positions, values, height=0.55, color=colors, zorder=2)
    ax.set_yticks(positions, labels, size=11)
    ax.get_yticklabels()[-1].set_weight('bold')
    ax.set_ylim(7.05, -0.62)
    extent = max(abs(value) for value in values)
    ax.set_xlim(min(0, *values) - 0.48 * extent, max(0, *values) + 0.48 * extent)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.set_xlabel('Contribution to predicted default (log-odds)', fontsize=10, labelpad=6)
    for y, value in zip(positions, values):
        right = value >= 0
        ax.annotate(signed(value, 4), (value, y), xytext=(4 if right else -4, 0),
                    textcoords='offset points', va='center', ha='left' if right else 'right',
                    fontsize=9.5, weight='bold' if y == positions[-1] else 'normal', color=INK)

    rule(fig, 0.688, 0.688, 0.23, 0.69)
    fig.text(0.717, 0.625, f'{row["fraction"]:.1%}', size=29, weight='bold', color=TEAL)
    fig.text(0.72, 0.574, 'of the total magnitude\ncancelled out', va='top', size=12, weight='bold', color=INK)
    fig.text(0.72, 0.429, f'{row["magnitude"]:.4f}  total magnitude\n{signed(row["net"], 4)}  signed total',
             linespacing=1.7, size=10.5, color=MUTED)
    fig.text(0.72, 0.265, f'Across all borrower–fit pairs,\nmedian cancellation was '
             f'{audit["median_bill_cancellation_fraction"]:.1%}.', linespacing=1.5, size=9.5, color=MUTED)
    fig.text(0.055, 0.110, f'April–September 2005 · Audit row {int(row["audit_row_index"]):,} · Fixed refit 0',
             size=10, color=INK)
    fig.text(0.055, 0.073, 'Selected by cancellation closest to the median for this fit. '
             'Ties go to the lowest zero-based workbook row index.', size=10, color=MUTED)
    return fig


def save_figures(directory, pilot, audit, bills):
    directory.mkdir()
    with rc_context(STYLE):
        for name, fig in [('figure-1-agreement', agreement_figure(pilot, audit)),
                          ('figure-2-cancellation', cancellation_figure(bills, audit))]:
            fig.savefig(directory / f'{name}.png', dpi=300)
            fig.savefig(directory / f'{name}.pdf', metadata={'CreationDate': None})
    print(f'Saved PNG and PDF figures to {directory.resolve()}')


def save_split_figure(directory, rows):
    with rc_context(STYLE):
        fig = page(3, 'TRAIN/TEST SPLITS', 'Does the grouping advantage repeat?',
                   'Each dot compares correlation groups with size-matched random groups on one split. Zero is a tie.')
        fig.text(0.055, 0.665, 'Split seed', weight='bold', size=12, color=INK)
        panels = [
            ('delta_rbo', 0.255, 'Overall ranking', 'Difference in rank overlap', 4),
            ('delta_top1_pp', 0.638, 'First reason per borrower', 'Difference in percentage points', 3),
        ]
        for field, left, title, unit, precision in panels:
            fig.text(left, 0.665, title, size=12, weight='bold', color=INK)
            fig.text(left, 0.619, unit, size=10, color=MUTED)
            ax = fig.add_axes([left, 0.262, 0.225, 0.323])
            values = [row[field] for row in rows]
            ax.axvline(0, color=MUTED, linewidth=1)
            ax.grid(axis='x', linewidth=0.55, color=GRID)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_yticks([])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            row_coords = blended_transform_factory(fig.transFigure, ax.transData)
            for i, (row, value) in enumerate(zip(rows, values, strict=True)):
                color = TEAL if value >= 0 else RUST
                ax.plot([0, value], [i, i], color=color, linewidth=1.6)
                ax.scatter(value, i, color=color, s=34, zorder=3)
                fig.text(left + 0.312, i, signed(value, precision), ha='right', va='center',
                         size=10, color=INK, transform=row_coords)
                if field == 'delta_rbo':
                    label = str(row['split_seed']) + ('  (pilot)' if row['split_role'] == 'pilot' else '')
                    fig.text(0.055, i, label, size=10, va='center', color=INK, transform=row_coords)
            ax.margins(x=0.12)
        fig.text(0.055, 0.113, 'Positive values favor correlation groups. Negative values favor random groups.',
                 size=10.5, color=INK)
        fig.text(0.055, 0.073, 'Each split uses 500 held-out borrowers, 10 model fits and 30 matched random partitions.',
                 size=10, color=MUTED)
        fig.savefig(directory / 'figure-3-split-stability.png', dpi=300)
        fig.savefig(directory / 'figure-3-split-stability.pdf', metadata={'CreationDate': None})


def _csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _save(directory, name, fig):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{name}.png', dpi=300)
    fig.savefig(directory / f'{name}.pdf', metadata={'CreationDate': None})


def gap_diagnostic_figure(gap_auc_path, gap_bin_path):
    aucs = pd.read_csv(gap_auc_path)
    bins = pd.read_csv(gap_bin_path)
    corr = bins[bins['method'].eq('correlation')].groupby('bin_low', as_index=False)['repeat_rate'].mean()
    random = bins[bins['method'].eq('random_mean')].groupby('bin_low', as_index=False)['repeat_rate'].mean()
    fig = page(4, 'NEAR-TIE DIAGNOSTIC', 'A clear leading reason was more repeatable',
               'Exploratory only: the first fit’s gap is used to predict whether its leading group repeats in the second fit.')
    ax = fig.add_axes([0.11, .27, .52, .39])
    ax.plot(corr['bin_low'], corr['repeat_rate'], marker='o', color=TEAL, label='Correlation groups')
    ax.plot(random['bin_low'], random['repeat_rate'], marker='o', color=SLATE, label='Random groups (mean)')
    ax.set_ylim(.4, 1.03)
    ax.set_xlabel('Relative gap between first and second groups')
    ax.set_ylabel('Repeat rate of the first reason')
    ax.grid(axis='y', color=GRID)
    ax.legend(frameon=False, loc='lower right')
    corr_auc = aucs[aucs['method'].eq('correlation')]['relative_gap_auc'].mean()
    random_auc = aucs[aucs['method'].eq('random_mean')]['relative_gap_auc'].mean()
    fig.text(.69, .60, f'{corr_auc:.3f}', size=24, weight='bold', color=TEAL)
    fig.text(.69, .54, 'mean AUC for correlation groups', size=11, color=INK)
    fig.text(.69, .43, f'{random_auc:.3f}', size=24, weight='bold', color=SLATE)
    fig.text(.69, .37, 'mean AUC for random groups', size=11, color=INK)
    fig.text(.055, .09, 'AUC treats the source-fit gap as a score for whether the first reason will repeat. '
             'Bins pool five exploratory train/test splits.', size=10, color=MUTED)
    return fig


def near_tie_rule_figure(metrics_path):
    rows = pd.read_csv(metrics_path)
    fig = page(5, 'ONE OR TWO REASONS', 'The frozen rule’s reliability and coverage',
               'Fresh splits only. One reason is shown when the source-fit relative gap is at least 0.36.')
    labels = [str(seed) for seed in rows['split_seed']]
    x = list(range(len(rows)))
    ax = fig.add_axes([.11, .29, .37, .37])
    ax.bar(x, rows['coverage'], color=TEAL)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels, rotation=30, ha='right')
    ax.set_ylabel('One-reason coverage')
    ax.grid(axis='y', color=GRID)
    ax2 = fig.add_axes([.57, .29, .37, .37])
    ax2.plot(x, rows['single_reason_repeat'], marker='o', color=TEAL, label='One-reason repeat')
    ax2.plot(x, rows['baseline_repeat'], marker='o', color=SLATE, label='All-case baseline')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x, labels, rotation=30, ha='right')
    ax2.set_ylabel('Repeat rate')
    ax2.grid(axis='y', color=GRID)
    ax2.legend(frameon=False, fontsize=9, loc='lower left')
    fig.text(.055, .105, 'Primary: one-reason repeat minus the all-case baseline.',
             size=10, color=MUTED)
    fig.text(.055, .075, 'Top-two containment is separate because showing two reasons makes matching easier.',
             size=10, color=MUTED)
    return fig


def _split_figure(rows, number, topic, title):
    fig = page(number, topic, title, 'Positive values favor correlation groups. Every dot is one predefined split.')
    axes = [(fig.add_axes([.13, .29, .31, .37]), 'delta_rbo', 'Δ rank overlap'),
            (fig.add_axes([.62, .29, .31, .37]), 'delta_top1_pp', 'Δ first reason (pp)')]
    x = list(range(len(rows)))
    labels = [str(row['split_seed']) for row in rows]
    for ax, key, ylabel in axes:
        values = [float(row[key]) for row in rows]
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.scatter(x, values, color=[TEAL if value >= 0 else RUST for value in values], s=44)
        ax.set_xticks(x, labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color=GRID)
    fig.text(.055, .105, 'Correlation and random groups use the same fitted models and held-out borrowers.',
             size=10, color=MUTED)
    fig.text(.055, .075, 'Random partitions also reuse the feature-level contributions and exact learned group sizes.',
             size=10, color=MUTED)
    return fig


def render_from_results(pilot_dir, original_csv, fresh_csv, exploratory_dir, fresh_near_tie_dir, output_dir):
    """Regenerate all report figures from committed inputs, not live model objects."""
    pilot_dir, output_dir = Path(pilot_dir), Path(output_dir)
    pilot = json.loads((pilot_dir / 'pilot.json').read_text(encoding='utf-8'))
    audit = json.loads((pilot_dir / 'borrower_audit.json').read_text(encoding='utf-8'))
    bills = pd.read_csv(pilot_dir / 'bill_cancellation.csv')
    original, fresh = _csv_rows(original_csv), _csv_rows(fresh_csv)
    with rc_context(STYLE):
        for name, fig in [
            ('figure-1-agreement', agreement_figure(pilot, audit)),
            ('figure-2-cancellation', cancellation_figure(bills, audit)),
            ('figure-3-split-stability', _split_figure(original, 3, 'TRAIN/TEST SPLITS',
                                                        'Does the grouping advantage repeat?')),
            ('figure-4-gap-diagnostic', gap_diagnostic_figure(Path(exploratory_dir) / 'gap_auc.csv',
                                                               Path(exploratory_dir) / 'gap_bins.csv')),
            ('figure-5-near-tie-rule', near_tie_rule_figure(Path(fresh_near_tie_dir) / 'rule_metrics.csv')),
            ('figure-6-fresh-split-stability', _split_figure(fresh, 6, 'FRESH TRAIN/TEST SPLITS',
                                                              'Does the original stability pattern repeat?')),
        ]:
            _save(output_dir, name, fig)
    print(f'Regenerated figures from machine-readable results in {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--original-csv', type=Path, required=True)
    parser.add_argument('--fresh-csv', type=Path, required=True)
    parser.add_argument('--exploratory-dir', type=Path, required=True)
    parser.add_argument('--fresh-near-tie-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    render_from_results(args.pilot_dir, args.original_csv, args.fresh_csv, args.exploratory_dir,
                        args.fresh_near_tie_dir, args.output_dir)
