# Audit of the published calculation

The checks below were run after the code was changed to count zero-magnitude rows, exact ties and the sign of the first reason, and to save the attributions used for later analysis. The random-number stream was not changed. The comparison is against the reproduction recorded before those edits.

## What was checked

One borrower, workbook row 2812, was followed from the UCI sheet through the seed-20261027 split, bootstrap refit 0, the group sums, `bill_cancellation.csv` and the rounded Figure 2 values. The trace is in [row-2812.md](../borrower-trace/row-2812.md). The row is the one the existing Figure 2 rule selects. Its UCI `ID` is 2813. It is in the test set and in the audit sample, and not in the training set. Its contributions match `beta_j / sd_j * (x_j - training mean_j)` to 1.4e-17. They add back to the model log-odds. The six bill contributions match the saved table and the numbers printed on Figure 2.

On that same refit, this borrower's first reason is `LIMIT_BAL`, with signed contribution −0.0578. It lowered predicted risk. The monthly-bill group is not the first reason. The first reason means the largest absolute contribution, so a group that lowers predicted risk can come first. Nothing in the audit turns that into a reason for denial. There is no decision threshold.

The same held-out borrowers are explained by every refit. Both grouping methods use those attributions. The random partitions have exactly the learned group sizes and cover every feature once. Ten fits produce 45 unordered pairs and 90 ordered pairs, and the two averages agree. The signed contributions are added before the absolute value is taken.

## What the re-run changed

Nothing in the published scores. On all five splits the re-run matched the pre-edit reproduction: rank overlap and first-reason agreement to the printed digit, and the median cancellation fraction to the same 1e-13-level difference from the committed CSV already recorded in the reproduction note. There were no undefined first-reason rows and no exact ties for first place, in any of the 5,000 borrower–fit rows on any split. The agreement computed after dropping undefined rows is therefore identical to the published agreement. No corrected table is needed, and the conclusion does not change.

The new count is the share of borrower–fit rows whose first reason lowered predicted risk:

| Split seed | Undefined rows | Exact ties | First reason lowered predicted risk |
| --- | ---: | ---: | ---: |
| 20261027 | 0 of 5,000 | 0 | 49.4% (2,470 rows) |
| 20261028 | 0 of 5,000 | 0 | 50.1% (2,503 rows) |
| 20261029 | 0 of 5,000 | 0 | 51.4% (2,568 rows) |
| 20261030 | 0 of 5,000 | 0 | 54.1% (2,705 rows) |
| 20261031 | 0 of 5,000 | 0 | 53.2% (2,658 rows) |

The shares for seeds 20261028 through 20261031 are 0.5006, 0.5136, 0.5410 and 0.5316. Across every predefined split, the leading group lowers predicted risk about as often as it raises it.
