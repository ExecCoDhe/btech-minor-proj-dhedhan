# COMET-XAI Lite: preliminary executable baseline

Run the pilot from the repository root. The `new-repo/` directory contains the
files from `Shreya732-debug/new-repo` at commit
`32a86d4ceac71cac54a683676edb053f0f1c9e5d`. See the
[research package index](new-repo/comet_deliverables/00_START_HERE.md).
One clone includes the pilot and research materials:

```bash
git clone https://github.com/prk0024/btech-minor-proj-dhedhan.git
```

## Reproduce the three borrower-level results

From the repository root, run this command with [uv](https://docs.astral.sh/uv/).
It sets up Python 3.12 with the pinned analysis packages, verifies the UCI workbook
(downloading it on first use), fits the models, and prints all three percentages.

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python demo.py --borrower-audit
```

If you already installed `requirements.txt` in a Python 3.12 environment,
`python demo.py --borrower-audit` runs the same analysis.

| Measure on the fixed pilot split | Expected result |
| --- | ---: |
| Same top group for a borrower across two refits, using correlation groups | 78.5% |
| Same top group for a borrower across two refits, averaged over 30 size-matched random partitions | 80.6% |
| Median share of the six monthly bill contributions cancelled when added together | 85.6% |

For each of 500 held-out borrowers, the audit identifies the group with the
largest absolute **signed sum** of contributions to the predicted log-odds.
An exact tie goes to the first group in partition order. It compares the same
borrower across all 45 pairs of the 10 bootstrap fits, giving 22,500 comparisons
per partition. Each random partition keeps the correlation groups' sizes and
stays fixed across those fits.

For cancellation, each borrower and fit contributes one row. Using the six
signed contributions for `BILL_AMT1` through `BILL_AMT6`, the cancelled share
is `1 - abs(sum(contributions)) / sum(abs(contributions))`. The reported
value is the median across all 5,000 rows pooled together. A row with zero
total magnitude has an undefined fraction and is excluded from the median.

The command prints the path to a new `runs/pilot-*/` directory.
`borrower_audit.json` contains the full-precision results, counts and all 30
random-partition scores. `bill_cancellation.csv` contains the six signed
contributions, net, total magnitude and cancelled fraction for each borrower
and fit. Its `audit_row_index` matches the zero-based indices in `splits.npz`.
These descriptive results cover one dataset and one train/test split.

## Report figures

### Figure 1. Stability of the overall ranking and each borrower's first reason

![Correlation groups achieved 0.924 overall rank overlap versus 0.863 for random groups. The same reason ranked first for an individual in 78.5% versus 80.6% of comparisons.](figures/figure-1-agreement.png)

On this split, grouping related features improved the stability of the overall
ranking by 0.061 on a 0–1 scale. The score is rank-biased overlap with
`p = 0.85`, which gives more weight to the first few positions. For an
individual borrower, the same reason ranked first 2.1 percentage points less
often with correlation groups than with the size-matched random groups.
[Vector PDF](figures/figure-1-agreement.pdf)

### Figure 2. Six monthly bill contributions for one borrower

![Diverging bars for six monthly bill contributions. September raised the predicted log-odds by 0.1182 and July lowered them by 0.0739. The six months together contributed +0.0388.](figures/figure-2-cancellation.png)

Positive bars raised the model's predicted log-odds of default relative to its
training background. Negative bars lowered them. For this borrower, the six
absolute contributions summed to 0.2220 log-odds units, and **82.5% cancelled**
when their signs were included. The median across all 5,000 borrower–fit pairs
was 85.6%.
[Vector PDF](figures/figure-2-cancellation.pdf)

The example uses bootstrap refit 0. Among borrowers with a defined cancellation
fraction, choose the one closest to that fit's median, breaking ties by the
lowest zero-based audit row index. This selects row 2812.

To regenerate both figures and the underlying audit in one run:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python demo.py --figures
```

Each run writes a 300 dpi PNG and a vector PDF of each figure in its own
`runs/pilot-*/figures/` directory.

## Archived pilot and population-wide rankings

The recorded UCI logistic-regression pilot comes from `ExecCoDhe/Research-paper-code`
at commit `88aef5b15a62311054c3dd92018778a735be26c0`. Its original files are in
`results/`. The saved ranking stability scores are **0.9240** for the correlation
groups and **0.8631** for the mean of 30 size-matched random partitions.
`results/pilot.json` records the workbook SHA-256 and package versions,
`results/metrics.csv` contains the per-fit metrics, and `results/splits.npz`
contains the training, test and audit row IDs.

Verify the archived results against the source commit from the repository root:

```bash
git diff --exit-code 88aef5b15a62311054c3dd92018778a735be26c0 -- results/
```

Python 3.12. Install `requirements.txt` in a virtual environment, then run:

```bash
python download_data.py
python -m unittest -v test_demo.py
python demo.py --bootstraps 10 --random-partitions 30
```

Each successful run prints its new directory under `runs/`, containing
`pilot.json`, `metrics.csv` and `splits.npz`. To choose a name, pass
`--output-dir runs/my-pilot`. The directory must be new. The pilot uses split
seed 20261027, audit seed 20, bootstrap seeds 0 through 9 for the default ten
fits, and seed 101 for random partitions.

Outputs contain data/split counts, group membership, metrics, versions and limits.
`kth_margin` and its tests audit a quantile error in the supplied manual; they are
not used by the pilot and do not constitute a recourse implementation.

Tested in the current cloud environment with versions in `requirements.txt`.
Laptop dependency installation and execution in a personal Colab runtime remain
to be verified. Read data/README.md for attribution, licence and source.

No claim of causal recourse, temporal robustness, legal compliance, general
superiority or publication readiness is made. Preserve preliminary labels.
