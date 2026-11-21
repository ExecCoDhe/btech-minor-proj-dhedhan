"""Render report figures from COMET-XAI Lite audit results."""
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
             size=9, color=MUTED)
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
               'Rightward bars raised this borrower’s predicted risk of default. Leftward bars lowered it.')
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
             'Ties go to the lowest audit row index.', size=9, color=MUTED)
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
                 size=9, color=MUTED)
        fig.savefig(directory / 'figure-3-split-stability.png', dpi=300)
        fig.savefig(directory / 'figure-3-split-stability.pdf', metadata={'CreationDate': None})
