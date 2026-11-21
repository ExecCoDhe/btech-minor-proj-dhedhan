# Five-split explanation stability experiment

## Protocol

Use five stratified 80/20 train/test splits on the UCI Default of Credit Card
Clients data. Seed **20261027** reproduces the published pilot. The four new
seeds are **20261028, 20261029, 20261030 and 20261031**, chosen consecutively
before running them. Include every split and give the original pilot its own
label.

Change only the train/test split seed in the root pilot. On each split, learn
groups on the 24,000 training records using complete linkage and an absolute
Spearman correlation threshold of 0.7. Keep those groups fixed across the ten
bootstrap logistic fits (seeds 0–9, C = 1). Select 500 of the 6,000 held-out
borrowers with seed 20. Use the same borrowers and 45 refit pairs for both
grouping methods. The control is the mean over 30 random partitions with
exactly the learned group sizes, generated with seed 101.

For each split, report both methods' overall rank-biased overlap (0–1,
p = 0.85) and borrower-level agreement on the first-ranked group. Calculate
each difference as correlation grouping minus the matched random mean. Express
the first-reason difference in percentage points and take its direction before
rounding. Also report the median cancellation fraction for the six-month bill
group over the 5,000 borrower–fit pairs.

Report the range and direction of the differences, identifying any reversal
relative to the pilot. All five splits reuse the same source population and
can share borrowers. Interpret the differences descriptively.

## Reproduction

From the repository root, run the following command.

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python repeat_splits.py
```

The printed run directory contains `per_split.csv` and a directory for each
seed with its train/test/audit row indices, model metrics, groups and
borrower-level calculations. Fractions in the CSV are on a 0–1 scale except
for `delta_top1_pp`, which is in percentage points.
