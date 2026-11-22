# Fresh five-split replication

These five split seeds were frozen in
[the near-tie protocol](../near-tie/PROTOCOL.md) before this directory was
created. They use the same UCI source population, model family, audit size,
bootstrap seeds and random-control procedure as the original splits. They are
fresh relative to the threshold selection, but they are not independent cohorts.

| Seed | Correlation RBO | Random RBO | Difference | Correlation first reason | Random first reason | Difference (pp) | Bill cancellation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261032 | 0.8895 | 0.8812 | +0.0083 | 79.378% | 79.519% | −0.141 | 91.4% |
| 20261033 | 0.8745 | 0.8753 | −0.0008 | 75.684% | 78.132% | −2.447 | 89.6% |
| 20261034 | 0.8761 | 0.8786 | −0.0025 | 77.013% | 77.543% | −0.530 | 84.5% |
| 20261035 | 0.8861 | 0.8754 | +0.0107 | 78.711% | 81.409% | −2.698 | 84.6% |
| 20261036 | 0.8788 | 0.8566 | +0.0222 | 76.520% | 74.567% | +1.953 | 85.8% |

Correlation grouping had higher overall RBO on three of these five splits. It
had higher first-reason agreement on one. These results therefore reinforce the
need to keep ranking stability and individual-reason stability separate.

`per_split.csv` holds full precision. Each seed directory holds its split IDs,
model metrics, attributions, borrower audit, and bill-cancellation table. The
fresh one-or-two-reason evaluation is in
[../near-tie/fresh/rule_metrics.csv](../near-tie/fresh/rule_metrics.csv).
