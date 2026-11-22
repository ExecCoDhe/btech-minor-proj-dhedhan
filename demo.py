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
OUTCOME = 'default payment next month'
DROPPED_COLUMNS = ['ID', 'SEX', 'MARRIAGE', 'AGE', 'EDUCATION', OUTCOME]


def load_frame(path=None):
    """Load the UCI workbook and the 19 history features used by the pilot."""
    path = ROOT / 'data' / 'default_of_credit_card_clients.xls' if path is None else Path(path)
    frame = pd.read_excel(path, header=1)
    if len(frame) != 30000:
        raise ValueError('Unexpected UCI row count')
    frame.columns = frame.columns.astype(str)
    y = frame[OUTCOME].astype(int)
    X = frame.drop(columns=DROPPED_COLUMNS)
    if X.shape[1] != 19:
        raise ValueError('Unexpected feature count')
    return frame, X, y


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
    """Absolute value of the signed sum for each borrower and group.

    Contributions are added with their signs first. The absolute value is taken
    afterwards, so opposing features inside a group reduce that group's score.
    """
    return np.column_stack([np.abs(phi[:, group].sum(axis=1)) for group in groups])


def linear_contributions(values, training_mean, scale, beta):
    """Return beta_j / sd_j * (x_j - training mean_j) for one borrower.

    The bootstrap scaler's center cancels when the background is the mean of
    that same scaling applied to the full training set.
    """
    values = np.asarray(values, dtype=float)
    training_mean = np.asarray(training_mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if values.shape != beta.shape or training_mean.shape != beta.shape or scale.shape != beta.shape:
        raise ValueError('The borrower, training mean, scale and coefficients must have the same length')
    if not np.isfinite(values).all() or np.any(scale == 0):
        raise ValueError('Need a finite borrower row and non-zero scales')
    return beta / scale * (values - training_mean)


def matched_random_partitions(sizes, n_features, count, seed):
    """Random partitions with exactly the learned group sizes, in seed order."""
    sizes = [int(size) for size in sizes]
    if count < 1 or n_features < 1 or not sizes or any(size < 1 for size in sizes) or sum(sizes) != n_features:
        raise ValueError('Need a positive count and positive group sizes that sum to the feature count')
    rng = np.random.default_rng(seed)
    cuts = np.cumsum(sizes)[:-1]
    return [[part.tolist() for part in np.split(rng.permutation(n_features), cuts)]
            for _ in range(count)]


def pack_groupings(groupings):
    """Pack a list of partitions into three integer arrays, with no pickle."""
    flat, lengths, counts = [], [], []
    for grouping in groupings:
        counts.append(len(grouping))
        for group in grouping:
            lengths.append(len(group))
            flat.extend(int(index) for index in group)
    return (np.asarray(flat, dtype=np.int16),
            np.asarray(lengths, dtype=np.int16),
            np.asarray(counts, dtype=np.int16))


def unpack_groupings(flat, lengths, counts):
    groupings, index, group_index = [], 0, 0
    for count in counts:
        grouping = []
        for _ in range(int(count)):
            length = int(lengths[group_index])
            grouping.append(flat[index:index + length].astype(int).tolist())
            index += length
            group_index += 1
        groupings.append(grouping)
    if index != len(flat) or group_index != len(lengths):
        raise ValueError('Packed groupings do not line up')
    return groupings


def save_attributions(path, attributions, audit_ids, groups, partitions):
    group_flat, group_lengths, group_counts = pack_groupings([groups])
    part_flat, part_lengths, part_counts = pack_groupings(partitions)
    np.savez_compressed(
        path,
        attributions=np.stack([np.asarray(phi, dtype=float) for phi in attributions]),
        audit_ids=np.asarray(audit_ids, dtype=np.int64),
        group_flat=group_flat, group_lengths=group_lengths, group_counts=group_counts,
        partition_flat=part_flat, partition_lengths=part_lengths, partition_counts=part_counts,
    )


def load_attributions(path):
    with np.load(path) as archive:
        groups = unpack_groupings(archive['group_flat'], archive['group_lengths'], archive['group_counts'])[0]
        partitions = unpack_groupings(archive['partition_flat'], archive['partition_lengths'], archive['partition_counts'])
        return archive['attributions'].copy(), archive['audit_ids'].copy(), groups, partitions


def first_reason_rows(phi, groups):
    """Identify each borrower's first reason, without treating a zero row as one.

    The first reason is the group with the largest absolute signed sum. An exact
    tie goes to the earliest group in partition order. A borrower whose every
    group has magnitude zero has no first reason: the winner is -1, and the
    signed net is undefined. A negative signed net means the first reason lowered
    predicted risk. It is not a reason for denial, because the audit applies no
    decision threshold.
    """
    magnitudes = group_magnitudes(phi, groups)
    nets = np.column_stack([phi[:, group].sum(axis=1) for group in groups])
    order = np.argsort(-magnitudes, axis=1, kind='stable')
    top = order[:, 0]
    second = order[:, 1] if order.shape[1] > 1 else order[:, 0]
    row = np.arange(len(magnitudes))
    largest = magnitudes[row, top]
    undefined = largest == 0
    at_maximum = magnitudes == largest[:, None]
    winner = top.copy()
    winner[undefined] = -1
    winner_net = nets[row, top].astype(float)
    winner_net[undefined] = np.nan
    return {'winner': winner, 'second': second, 'largest': largest,
            'second_largest': magnitudes[row, second],
            'tied': (at_maximum.sum(axis=1) > 1) & ~undefined,
            'undefined': undefined, 'winner_net': winner_net}


def borrower_top1_summary(attributions, groups):
    """Published agreement, plus explicit counts for ties, zeros and sign.

    `agreement` keeps the historical rule, including the argmax behaviour on an
    all-zero row, so previously published scores stay comparable. Rows with no
    first reason are removed only from `agreement_defined_only`.
    """
    details = [first_reason_rows(phi, groups) for phi in attributions]
    defined_rates = []
    for left, right in combinations(details, 2):
        keep = ~left['undefined'] & ~right['undefined']
        if keep.any():
            defined_rates.append(float(np.mean(left['winner'][keep] == right['winner'][keep])))
    undefined_rows = int(sum(int(detail['undefined'].sum()) for detail in details))
    tied_rows = int(sum(int(detail['tied'].sum()) for detail in details))
    defined_rows = int(sum(int((~detail['undefined']).sum()) for detail in details))
    lowers = int(sum(int(np.sum(detail['winner_net'] < 0)) for detail in details))
    return {
        'agreement': borrower_top1_agreement(attributions, groups),
        'agreement_defined_only': float(np.mean(defined_rates)) if defined_rates else None,
        'undefined_first_reason_rows': undefined_rows,
        'tied_top_first_reason_rows': tied_rows,
        'defined_first_reason_rows': defined_rows,
        'top_group_lowers_risk_rows': lowers,
        'top_group_lowers_risk_share': (lowers / defined_rows) if defined_rows else None,
        'unordered_fit_pairs': len(attributions) * (len(attributions) - 1) // 2,
        'ordered_fit_pairs': len(attributions) * (len(attributions) - 1),
    }


def fit_one(X, y, train_ids, audit_ids, seed):
    """Fit one bootstrap logistic model and explain the audit rows.

    Resampling, the imputer and the scaler use the training rows only. The
    background is the full training set transformed by this fit's preprocessor.
    """
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(train_ids), len(train_ids), replace=True)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    model.fit(X.iloc[sampled], y.iloc[sampled])
    prep, linear = model[:-1], model[-1]
    background = prep.transform(X.iloc[train_ids]).mean(axis=0)
    audit = X.iloc[audit_ids]
    phi = (prep.transform(audit) - background) * linear.coef_[0]
    baseline = float(linear.intercept_[0] + background @ linear.coef_[0])
    residual = float(np.max(np.abs(baseline + phi.sum(axis=1) - model.decision_function(audit))))
    return {'sampled': np.asarray(sampled), 'model': model, 'background': background,
            'phi': phi, 'residual': residual, 'linear': linear, 'prep': prep}


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
    # Historical snapshot: explaining predictions only, never editing these histories.
    frame, X, y = load_frame(path)
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
        fitted = fit_one(X, y, train_ids, audit_ids, seed)
        residuals.append(fitted['residual'])
        phis.append(fitted['phi'])
        linear = fitted['linear']
        probs = fitted['model'].predict_proba(test)[:, 1]
        metrics.append({'seed':seed,'AP':average_precision_score(y.iloc[test_ids],probs),
                        'ROC_AUC':roc_auc_score(y.iloc[test_ids],probs),
                        'Brier':brier_score_loss(y.iloc[test_ids],probs),
                        'iterations':int(linear.n_iter_[0])})
    assert max(residuals) < 1e-8
    partitions = matched_random_partitions([len(g) for g in groups], X.shape[1], random_partitions, 101)
    random_scores = [mean_stability(phis, partition) for partition in partitions]
    random_top1 = [borrower_top1_agreement(phis, partition) for partition in partitions] if borrower_audit else []
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
              'versions':{m:importlib.metadata.version(m) for m in
                          ['numpy','pandas','scipy','scikit-learn','xlrd','matplotlib']},
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
    save_attributions(out/'attributions.npz', phis, audit_ids, groups, partitions)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    if borrower_audit:
        reasons = borrower_top1_summary(phis, groups)
        top1 = reasons['agreement']
        control = float(np.mean(random_top1))
        month_index = [result['features'].index(f'BILL_AMT{i}') for i in range(1, 7)]
        bill_group_found = any(set(group) == set(month_index) for group in groups)
        if bill_group_found:
            bills = bill_cancellation_table(phis, audit_ids, result['features'], groups)
            defined = bills['fraction'].dropna()
            median_fraction = float(defined.median()) if len(defined) else None
            undefined_bills = len(bills) - len(defined)
            bill_rows = len(bills)
            bills.to_csv(out / 'bill_cancellation.csv', index=False)
        else:
            bills = None
            defined = None
            median_fraction = None
            undefined_bills = None
            bill_rows = 0
        summary = {
            'audit_borrowers': len(audit_ids),
            'bootstrap_fits': bootstraps,
            'fit_pairs': result['bootstrap_pairs'],
            'ordered_fit_pairs': reasons['ordered_fit_pairs'],
            'random_partitions': random_partitions,
            'top1_agreement_correlation_groups': top1,
            'top1_agreement_defined_only': reasons['agreement_defined_only'],
            'top1_agreement_size_matched_random_mean': control,
            'top1_agreement_by_random_partition': random_top1,
            'undefined_first_reason_rows': reasons['undefined_first_reason_rows'],
            'tied_top_first_reason_rows': reasons['tied_top_first_reason_rows'],
            'defined_first_reason_rows': reasons['defined_first_reason_rows'],
            'top_group_lowers_risk_rows': reasons['top_group_lowers_risk_rows'],
            'top_group_lowers_risk_share': reasons['top_group_lowers_risk_share'],
            'bill_group_found': bill_group_found,
            'bill_borrower_refit_rows': bill_rows,
            'undefined_bill_fractions': undefined_bills,
            'median_bill_cancellation_fraction': median_fraction,
        }
        (out / 'borrower_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(f'Borrower first-place agreement (correlated groups): {top1:.1%}')
        print(f'Borrower first-place agreement (matched random groups): {control:.1%}')
        median_label = 'undefined' if median_fraction is None else f'{median_fraction:.1%}'
        print(f'Median six-month bill cancellation: {median_label}')
        print(f'Undefined first-reason rows: {reasons["undefined_first_reason_rows"]}')
        share = reasons['top_group_lowers_risk_share']
        share_label = 'undefined' if share is None else f'{share:.1%}'
        print(f'First reason lowered predicted risk: {share_label}')
        if figures:
            if bills is None:
                raise ValueError('The report figures need the six monthly bills to form one group')
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
