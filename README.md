# COMET-XAI Lite: preliminary executable baseline

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
