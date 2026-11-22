# Clean-checkout verification

Verified on 22 November 2026 from a new `--no-local` clone of branch
`submission-readiness` at commit `42ad3b5`. The clone began without the
ignored UCI workbook or any `runs/` directory.

Commands were run from that clone:

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest -v test_demo.py test_near_tie.py
uv run --no-project --python 3.12 --with-requirements requirements.txt python demo.py --borrower-audit --output-dir runs/pilot
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest -v test_demo.py test_near_tie.py
uv run --no-project --python 3.12 --with-requirements requirements.txt python repeat_splits.py --seeds original --output-dir runs/original-five
uv run --no-project --python 3.12 --with-requirements requirements.txt python near_tie.py explore --splits-dir runs/original-five --output runs/near-tie/exploratory
uv run --no-project --python 3.12 --with-requirements requirements.txt python repeat_splits.py --seeds fresh --output-dir runs/fresh-five
uv run --no-project --python 3.12 --with-requirements requirements.txt python near_tie.py evaluate --splits-dir runs/fresh-five --protocol runs/near-tie/protocol.json --output runs/near-tie/fresh
uv run --no-project --python 3.12 --with-requirements requirements.txt python check_reproduction.py --pilot-dir runs/pilot --original-csv runs/original-five/per_split.csv --fresh-csv runs/fresh-five/per_split.csv --near-tie-dir runs/near-tie/fresh
```

The first test invocation passed 25 tests and skipped the data-gated
integration test, as expected before `demo.py` downloaded the verified
workbook. The second invocation passed all 26 tests in 5.130 seconds. The
pilot printed the archived 78.5%, 80.6%, and 85.6% values. Both five-split
runs completed, the frozen threshold was again 0.36, the fresh verdict again
met the criterion on all five splits, and `check_reproduction.py` passed.

`git status --short` was empty after the workflow because the downloaded
workbook and run outputs are ignored.
