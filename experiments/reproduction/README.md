# Reproduction before any analysis edit

Recorded on 22 November 2026, on branch `submission-readiness`, before the audit changes in later commits. The commands were run from the repository root. No analysis code had been edited. `runs/` is gitignored, so the numbers are copied here.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11, build 10.0.26300 |
| uv | 0.11.25 |
| Python | 3.12.10 |
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| scipy | 1.17.0 |
| scikit-learn | 1.8.0 |
| xlrd | 2.0.1 |
| matplotlib | 3.10.8 |

The workbook was absent at the start. `demo.py --figures` downloaded it through `download_data.py` and checked SHA-256 `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

## Commands

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest -v test_demo.py
uv run --no-project --python 3.12 --with-requirements requirements.txt python demo.py --figures --output-dir runs/repro-pilot
uv run --no-project --python 3.12 --with-requirements requirements.txt python repeat_splits.py --output-dir runs/repro-original-five
```

## Tests

13 tests, all passing, in 0.007 seconds. The archived `results/test_log.txt` reports 9 tests. That log predates the borrower-level cancellation tests. It is not a failure of this run.

## Pilot compared with `results/pilot.json`

Absolute differences were 0 for the row counts, the prevalence, the ten per-fit AP / ROC-AUC / Brier values and iteration counts, both rank-overlap scores, the 30 random-partition scores, the grouping, and the data hash. Python was 3.12.10 in both files. Runtime differed, as expected.

`results/run_log.txt` does not come from that same run. It records Python 3.12.14, mean Brier `0.14765037952357002` against `0.14765037952356969` in `pilot.json`, and a largest additivity error of `2.22e-15` against `1.55e-15`. The rank-overlap scores match. The archive was left unchanged. This reproduction matches `pilot.json`, not the cloud log.

Printed borrower-level results, unchanged from the root README: 78.5% correlation-group first-reason agreement, 80.6% for size-matched random groups, and 85.6% median six-month bill cancellation.

## Five splits compared with `experiments/repeated-splits/per_split.csv`

Tolerance was an absolute difference of 1e-9. Rank overlap, first-reason agreement, and both differences matched to the printed digit (difference 0). Only the median cancellation fraction differed, and only past the 12th decimal place:

| Split seed | Committed median | Reproduced median | Absolute difference |
| --- | ---: | ---: | ---: |
| 20261027 | 0.855603330447942 | 0.8556033304477719 | 1.7e-13 |
| 20261028 | 0.9072332834341273 | 0.9072332834340764 | 5.1e-14 |
| 20261029 | 0.8888916351926588 | 0.8888916351926701 | 1.1e-14 |
| 20261030 | 0.8824401719123234 | 0.8824401719122275 | 9.6e-14 |
| 20261031 | 0.8828328869875044 | 0.8828328869875033 | 1.1e-15 |

The largest difference is 1.7e-13, inside the tolerance. Rounded to one decimal percentage, every split is unchanged: 85.6%, 90.7%, 88.9%, 88.2%, 88.3%.

Printed differences, which match the committed table:

| Split seed | Rank-overlap difference | First-reason difference (percentage points) |
| --- | ---: | ---: |
| 20261027 | +0.0609 | −2.062 |
| 20261028 | +0.0479 | +0.005 |
| 20261029 | +0.0209 | −2.789 |
| 20261030 | +0.0064 | +0.311 |
| 20261031 | −0.0018 | −0.457 |

## Figures

Figure 1 and Figure 3 vector PDFs were byte-identical to `figures/`. The Figure 2 PDF differed by one byte (34,065 against 34,066). All three PNGs differed in file size. Matplotlib's PNG encoder is not stable across machines, so the PDF match is the meaningful check. No figure text was inspected as a numerical discrepancy, because the underlying scores matched.

## Conclusion

The published rounded findings reproduced. No analysis code was changed in order to obtain them.
