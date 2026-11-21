"""COMET-XAI Lite: preliminary attribution-stability audit, NOT a research result.

Run: python demo.py --bootstraps 10 --random-partitions 30
Uses official UCI 350. No recourse, no LIME, no tree SHAP, no claimed novelty.
Logistic interventional SHAP is exact in log-odds: beta_j * (z_j - E[z_j]).
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from itertools import combinations
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PILOT_SPLIT_SEED = 20261027


def groups_from_training(X, rho=0.7):
    if not 0 <= rho <= 1:
        raise ValueError('rho must be in [0, 1]')
    if X.shape[1] < 2:
        return [[0]], np.eye(1)
    if (X.nunique() < 2).any():
        raise ValueError('Remove constant columns using training data first')
    corr = X.corr(method='spearman').to_numpy()
    if not np.isfinite(corr).all():
        raise ValueError('Undefined correlation')
    dist = np.clip(1 - np.abs(corr), 0, 1)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=True), method='complete'),
                      t=1-rho, criterion='distance')
    groups = [np.flatnonzero(labels == k).tolist() for k in np.unique(labels)]
    return groups, corr


def rbo(a, b, p=0.85):
    if not 0 < p < 1 or not len(a) or len(a) != len(b):
        raise ValueError('Need nonempty equal-length rankings and 0<p<1')
    if len(set(a)) != len(a) or set(a) != set(b):
        raise ValueError('Need permutations of the same unique universe')
    overlaps = [len(set(a[:d]) & set(b[:d])) / d for d in range(1, len(a)+1)]
    return float((1-p)*sum(v*p**i for i,v in enumerate(overlaps)) + overlaps[-1]*p**len(a))


def group_cancellation(contributions) -> tuple[float, float, float | None]:
    """Return net, total magnitude and fraction cancelled for one group on one row."""
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty vector of finite feature contributions")
    net = float(values.sum())
    magnitude = float(np.abs(values).sum())
    fraction = None if magnitude == 0 else 1 - abs(net) / magnitude
    return net, magnitude, fraction


def group_magnitudes(phi, groups):
    """Absolute net log-odds contribution for each borrower and group."""
    return np.column_stack([np.abs(phi[:, group].sum(axis=1)) for group in groups])


def mean_stability(attributions, groups):
    scores = [group_magnitudes(phi, groups).mean(axis=0) for phi in attributions]
    rankings = [np.argsort(-s, kind='stable').tolist() for s in scores]
    vals = [rbo(rankings[i], rankings[j]) for i in range(len(rankings))
            for j in range(i+1, len(rankings))]
    return float(np.mean(vals))


def borrower_top1_agreement(attributions, groups):
    """Mean within-borrower agreement over unordered pairs of refits."""
    winners = [group_magnitudes(phi, groups).argmax(axis=1) for phi in attributions]
    return float(np.mean([np.mean(a == b) for a, b in combinations(winners, 2)]))


def bill_cancellation_table(attributions, audit_ids, features, groups):
    months = [f'BILL_AMT{i}' for i in range(1, 7)]
    indices = [features.index(month) for month in months]
    if not any(set(group) == set(indices) for group in groups):
        raise ValueError('Expected the six bill months to form one training-derived group')
    rows = []
    for seed, phi in enumerate(attributions):
        for borrower, contributions in zip(audit_ids, phi[:, indices], strict=True):
            net, magnitude, fraction = group_cancellation(contributions)
            rows.append((seed, borrower, *contributions, net, magnitude, fraction))
    return pd.DataFrame(rows, columns=['refit', 'audit_row_index', *months, 'net', 'magnitude', 'fraction'])


def kth_margin(shifts, alpha):
    # Exact finite-sample order statistic, NOT a drift coverage guarantee.
    a = np.maximum(0, np.asarray(shifts, dtype=float))
    if not len(a) or not np.isfinite(a).all() or not 0 < alpha < 1:
        raise ValueError('Need finite nonempty shifts and 0<alpha<1')
    k = int(np.ceil((len(a)+1)*(1-alpha)))
    return float('inf') if k > len(a) else float(np.sort(a)[k-1])


def run(bootstraps=10, random_partitions=30, *, output_dir=None, borrower_audit=False, figures=False,
        split_seed=PILOT_SPLIT_SEED):
    if bootstraps < 2 or random_partitions < 1:
        raise ValueError('Need >=2 bootstrap fits and >=1 random partition')
    start = time.perf_counter()
    path = ROOT/'data/default_of_credit_card_clients.xls'
    borrower_audit = borrower_audit or figures
    if borrower_audit:
        from download_data import ensure_workbook
        ensure_workbook(path)
    frame = pd.read_excel(path, header=1)
    assert len(frame) == 30000, 'Unexpected UCI row count'
    frame.columns = frame.columns.astype(str)
    y = frame['default payment next month'].astype(int)
    # Historical snapshot: explaining predictions only, never editing these histories.
    X = frame.drop(columns=['ID','SEX','MARRIAGE','AGE','EDUCATION','default payment next month'])
    assert X.shape[1] == 19
    ids = np.arange(len(X))
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=split_seed, stratify=y)
    assert not set(train_ids) & set(test_ids)
    train, test = X.iloc[train_ids], X.iloc[test_ids]
    groups, corr = groups_from_training(train)
    for g in groups:
        assert np.abs(corr[np.ix_(g,g)]).min() >= .7-1e-12
    # Audit subset selected without outcomes; same applicants for every model.
    audit_ids = np.random.default_rng(20).choice(test_ids, 500, replace=False)
    audit = X.iloc[audit_ids]
    phis, metrics, residuals = [], [], []
    for seed in range(bootstraps):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(train_ids, len(train_ids), replace=True)
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        model.fit(X.iloc[sampled], y.iloc[sampled])
        prep, linear = model[:-1], model[-1]
        # Common raw training background, transformed with each model's own preprocessor.
        background = prep.transform(train).mean(axis=0)
        z = prep.transform(audit)
        phi = (z-background)*linear.coef_[0]
        baseline = float(linear.intercept_[0]+background @ linear.coef_[0])
        residuals.append(float(np.max(np.abs(baseline+phi.sum(axis=1)-model.decision_function(audit)))))
        phis.append(phi)
        probs = model.predict_proba(test)[:, 1]
        metrics.append({'seed':seed,'AP':average_precision_score(y.iloc[test_ids],probs),
                        'ROC_AUC':roc_auc_score(y.iloc[test_ids],probs),
                        'Brier':brier_score_loss(y.iloc[test_ids],probs),
                        'iterations':int(linear.n_iter_[0])})
    assert max(residuals) < 1e-8
    rng = np.random.default_rng(101)
    sizes = [len(g) for g in groups]
    random_scores = []
    random_top1 = []
    for _ in range(random_partitions):
        perm = rng.permutation(X.shape[1])
        random_groups = [a.tolist() for a in np.split(perm,np.cumsum(sizes)[:-1])]
        random_scores.append(mean_stability(phis, random_groups))
        if borrower_audit:
            random_top1.append(borrower_top1_agreement(phis, random_groups))
    grouped = mean_stability(phis, groups)
    feature_stability = mean_stability(phis, [[i] for i in range(X.shape[1])])
    result = {'status':'preliminary single-split real-data logistic baseline; not full method validation',
              'dataset':'UCI Default of Credit Card Clients, ID 350', 'n_rows':len(frame),
              'split_seed':split_seed,
              'train_rows':len(train_ids),'test_rows':len(test_ids),'audit_rows':len(audit),
              'default_prevalence':float(y.mean()),'features':X.columns.tolist(),
              'groups':[[X.columns[i] for i in g] for g in groups],
              'bootstrap_fits':bootstraps,'bootstrap_pairs':bootstraps*(bootstraps-1)//2,
              'random_partitions':random_partitions,'predictive_metrics_per_fit':metrics,
              'mean_AP':float(np.mean([m['AP'] for m in metrics])),
              'mean_ROC_AUC':float(np.mean([m['ROC_AUC'] for m in metrics])),
              'mean_Brier':float(np.mean([m['Brier'] for m in metrics])),
              'feature_RBO':feature_stability,'group_RBO':grouped,
              'matched_random_mean_RBO':float(np.mean(random_scores)),
              'matched_random_sd_RBO':float(np.std(random_scores,ddof=1)),
              'matched_random_scores':random_scores,
              'group_minus_matched_random_RBO':grouped-float(np.mean(random_scores)),
              'max_logit_additivity_error':max(residuals),
              'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'python':platform.python_version(),
              'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','xlrd']},
              'elapsed_seconds':time.perf_counter()-start,
              'limitations':['one fixed split','logistic model only','no LIME or tree SHAP',
                            'bootstrap pairs dependent; no inferential p-values',
                            'no temporal, causal, recourse, or publication claim',
                            'grouping fixed; group discovery uncertainty not measured',
                            'random partition score spread is not a confidence interval']}
    if output_dir is None:
        runs = ROOT / 'runs'
        runs.mkdir(exist_ok=True)
        out = Path(mkdtemp(prefix='pilot-', dir=runs))
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(out/'splits.npz',train=train_ids,test=test_ids,audit=audit_ids)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    if borrower_audit:
        bills = bill_cancellation_table(phis, audit_ids, result['features'], groups)
        defined = bills['fraction'].dropna()
        median_fraction = float(defined.median()) if len(defined) else None
        top1 = borrower_top1_agreement(phis, groups)
        control = float(np.mean(random_top1))
        summary = {
            'audit_borrowers': len(audit_ids),
            'bootstrap_fits': bootstraps,
            'fit_pairs': result['bootstrap_pairs'],
            'random_partitions': random_partitions,
            'top1_agreement_correlation_groups': top1,
            'top1_agreement_size_matched_random_mean': control,
            'top1_agreement_by_random_partition': random_top1,
            'bill_borrower_refit_rows': len(bills),
            'undefined_bill_fractions': len(bills) - len(defined),
            'median_bill_cancellation_fraction': median_fraction,
        }
        bills.to_csv(out / 'bill_cancellation.csv', index=False)
        (out / 'borrower_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(f'Borrower first-place agreement (correlated groups): {top1:.1%}')
        print(f'Borrower first-place agreement (matched random groups): {control:.1%}')
        median_label = 'undefined' if median_fraction is None else f'{median_fraction:.1%}'
        print(f'Median six-month bill cancellation: {median_label}')
        if figures:
            from report_figures import save_figures
            save_figures(out / 'figures', result, summary, bills)
    (out/'pilot.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['matched_random_scores','predictive_metrics_per_fit','features']},indent=2))
    print(f"Saved pilot results to {out.resolve()}")
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bootstraps',type=int,default=10)
    parser.add_argument('--random-partitions',type=int,default=30)
    parser.add_argument('--borrower-audit', action='store_true',
                        help='Verify or download the UCI workbook and report borrower-level results')
    parser.add_argument('--figures', action='store_true',
                        help='Run the borrower audit and export both report figures as PNG and PDF')
    parser.add_argument('--output-dir', type=Path, help='New directory for this run (default: runs/pilot-<unique suffix>)')
    args=parser.parse_args()
    run(args.bootstraps, args.random_partitions, output_dir=args.output_dir,
        borrower_audit=args.borrower_audit, figures=args.figures)
