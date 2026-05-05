"""
Head vs tail analysis: where does Laplacian regularization help most?

Interactions are partitioned into buckets along one of two axes
(selectable via ``--bucket_by``):

    item_popularity  -- items bucketed by training popularity (standard);
                        head = most popular, tail = least popular
    user_activity    -- users bucketed by train-set interaction count;
                        head = most active users, tail = least active

For each bucket we compute NDCG@k / Recall@k / HitRate@k restricted to
relevant items (or users, for user_activity mode) in that bucket, for
both the baseline (vanilla EASE) and Laplacian-EASE. Outputs the
absolute and relative gain per bucket.

Two modes are supported:

  1. **Single-gamma mode** (default): one gamma value, emit the bar
     plot of per-bucket gains plus a Wilcoxon signed-rank test on the
     per-user NDCG for both overall and per-bucket significance.

  2. **Gamma-sweep mode** (``--gammas "0,3,10,30,50,75"``): run the
     per-bucket analysis for every gamma on the grid and produce a
     line plot of per-bucket abs_gain vs gamma. This is the key
     diagnostic for the head/tail tradeoff: does some gamma help the
     head bucket without crushing the tail, or is the tradeoff
     monotonic?

Example:
    # Single gamma, item-popularity buckets, 5 quintiles, Wilcoxon
    python -m experiments.head_tail_analysis \\
        --dataset ml-1m --k 10 --gamma 50 --n_buckets 5

    # Sweep gamma to diagnose the head/tail tradeoff
    python -m experiments.head_tail_analysis \\
        --dataset ml-1m --k 10 --n_buckets 5 \\
        --gammas "0,3,10,30,50,75"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k
from models import HybridEASE_RP3beta

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
    wilcoxon_paired,
)


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

def _quantile_labels(n_buckets):
    """Bucket labels: head/torso/tail for 3; q1..qN otherwise."""
    if n_buckets == 3:
        return ('head', 'torso', 'tail')
    return tuple(f'q{i+1}' for i in range(n_buckets))


def _item_popularity_buckets(train_df, n_buckets=3, labels=None):
    """Bucket items by training popularity (rank-based equal-frequency).

    Returns a dict ``{item_id -> bucket_label}``. Items are split into
    roughly equal-sized chunks by descending popularity, so label[0]
    = most-popular items, label[-1] = least popular.
    """
    if labels is None:
        labels = _quantile_labels(n_buckets)
    counts = train_df.groupby('item_id')['user_id'].nunique()
    sorted_counts = counts.sort_values(ascending=False)
    chunks = np.array_split(sorted_counts.index.to_numpy(), n_buckets)
    bucket_of = {}
    for label, chunk in zip(labels, chunks):
        for item in chunk:
            bucket_of[item] = label
    return bucket_of


def _user_activity_buckets(train_df, n_buckets=3, labels=None):
    """Bucket users by training interaction count (rank-based).

    Returns a dict ``{user_id -> bucket_label}``. label[0] = most
    active users, label[-1] = least active users.
    """
    if labels is None:
        labels = _quantile_labels(n_buckets)
    counts = train_df.groupby('user_id')['item_id'].nunique()
    sorted_counts = counts.sort_values(ascending=False)
    chunks = np.array_split(sorted_counts.index.to_numpy(), n_buckets)
    bucket_of = {}
    for label, chunk in zip(labels, chunks):
        for user in chunk:
            bucket_of[user] = label
    return bucket_of


# ---------------------------------------------------------------------------
# Per-bucket evaluation
# ---------------------------------------------------------------------------

def _evaluate_by_bucket(model, train_df, test_df, bucket_of, k=10,
                        bucket_by='item_popularity'):
    """Evaluate NDCG@k per bucket and return per-user NDCG arrays.

    Parameters
    ----------
    bucket_by : {'item_popularity', 'user_activity'}
        Determines whether ``bucket_of`` is keyed by item or by user.
        In ``item_popularity`` mode, a user's relevant items are split
        by bucket and we compute NDCG on each subset -- so one user can
        contribute to multiple buckets. In ``user_activity`` mode, each
        user goes entirely into their own bucket and we compute NDCG
        on their full relevant set.

    Returns
    -------
    summary : dict
        ``{bucket: {'ndcg': mean, 'recall': mean, 'hit': mean,
                    'n_users': int}}`` plus an ``overall`` entry.
    per_user_ndcg : dict
        ``{bucket: np.ndarray of per-user NDCG}`` -- needed for paired
        tests downstream. The ``overall`` key holds all users.
    """
    pred_matrix = model.pred
    if hasattr(pred_matrix, 'toarray'):
        pred_matrix = pred_matrix.toarray()

    user_enc = model.ease.user_enc
    item_enc = model.ease.item_enc
    known_users = set(user_enc.classes_)
    known_items = set(item_enc.classes_)

    test_grouped = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    train_grouped = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    bucket_labels = set(bucket_of.values())
    per_bucket = {b: {'ndcg': [], 'recall': [], 'hit': [], 'users': []}
                  for b in bucket_labels}
    overall = {'ndcg': [], 'recall': [], 'hit': [], 'users': []}

    for user_id, relevant in test_grouped.items():
        if user_id not in known_users:
            continue
        relevant = relevant & known_items
        if not relevant:
            continue

        user_idx = user_enc.transform([user_id])[0]
        scores = pred_matrix[user_idx, :].copy()

        for item_id in train_grouped.get(user_id, set()):
            if item_id in known_items:
                scores[item_enc.transform([item_id])[0]] = -np.inf

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        top_k_items = item_enc.inverse_transform(top_k_idx).tolist()

        rel_list = list(relevant)
        u_ndcg = ndcg_at_k(top_k_items, rel_list, k)
        u_rec = recall_at_k(top_k_items, rel_list, k)
        u_hit = hit_rate_at_k(top_k_items, rel_list, k)

        overall['ndcg'].append(u_ndcg)
        overall['recall'].append(u_rec)
        overall['hit'].append(u_hit)
        overall['users'].append(user_id)

        if bucket_by == 'item_popularity':
            # One user can contribute to several item buckets.
            for bucket in bucket_labels:
                rel_in_bucket = [i for i in rel_list
                                 if bucket_of.get(i) == bucket]
                if not rel_in_bucket:
                    continue
                per_bucket[bucket]['ndcg'].append(
                    ndcg_at_k(top_k_items, rel_in_bucket, k))
                per_bucket[bucket]['recall'].append(
                    recall_at_k(top_k_items, rel_in_bucket, k))
                per_bucket[bucket]['hit'].append(
                    hit_rate_at_k(top_k_items, rel_in_bucket, k))
                per_bucket[bucket]['users'].append(user_id)
        elif bucket_by == 'user_activity':
            # Each user falls in exactly one bucket (the user's own).
            bucket = bucket_of.get(user_id)
            if bucket is None:
                # Users in test but not in train (cold-start) are skipped.
                continue
            per_bucket[bucket]['ndcg'].append(u_ndcg)
            per_bucket[bucket]['recall'].append(u_rec)
            per_bucket[bucket]['hit'].append(u_hit)
            per_bucket[bucket]['users'].append(user_id)
        else:
            raise ValueError(f"Unknown bucket_by={bucket_by!r}")

    summary = {
        'overall': {m: float(np.mean(v)) if v else 0.0
                    for m, v in overall.items() if m != 'users'},
    }
    summary['overall']['n_users'] = len(overall['users'])
    for b, metrics in per_bucket.items():
        summary[b] = {m: float(np.mean(v)) if v else 0.0
                      for m, v in metrics.items() if m != 'users'}
        summary[b]['n_users'] = len(metrics['users'])

    per_user_ndcg = {
        'overall': (np.asarray(overall['ndcg'], dtype=float),
                    list(overall['users'])),
    }
    for b, metrics in per_bucket.items():
        per_user_ndcg[b] = (np.asarray(metrics['ndcg'], dtype=float),
                            list(metrics['users']))

    return summary, per_user_ndcg


# Back-compat alias -- the canonical helper now lives in
# experiments/_shared.py and is used by all experiment scripts.
_wilcoxon_paired = wilcoxon_paired


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _fit_ease(train, ease_lambda, rp3_beta, rp3_topK):
    base = HybridEASE_RP3beta()
    base.fit(train, method='score', fusion_alpha=1.0,
             ease_lambda=ease_lambda, rp3_alpha=1.0,
             rp3_beta=rp3_beta, rp3_topK=rp3_topK)
    base.pred = base.ease.X.dot(base.ease.B)
    return base


def _fit_laplacian(train, ease_lambda, rp3_beta, rp3_topK,
                   gamma, graph_source, normalise):
    model = HybridEASE_RP3beta()
    model.fit(train, method='laplacian',
              ease_lambda=ease_lambda, rp3_alpha=1.0,
              rp3_beta=rp3_beta, rp3_topK=rp3_topK,
              graph_reg_gamma=gamma, graph_source=graph_source,
              laplacian_normalise=normalise)
    return model


def run(dataset='ml-small', k=10,
        ease_lambda=None, gamma=None, rp3_beta=0.6, rp3_topK=200,
        graph_source='rp3beta', normalise='none',
        n_buckets=3, bucket_by='item_popularity',
        gammas=None,
        split_mode='temporal', split_seed=0,
        out_dir=None):
    """Run head/tail analysis.

    If ``gammas`` is None (default) the script runs a single-gamma
    analysis (as before). If ``gammas`` is provided, it runs the
    per-bucket gain-vs-gamma sweep and additionally emits a line plot.

    Parameters
    ----------
    split_mode : {'temporal', 'random'}
        'temporal' (default) matches the existing experiment convention.
        'random' uses a random 80/20 per-user split (primary protocol);
        use ``split_seed`` to control reproducibility.
    split_seed : int
        Seed for the random split. Only used when ``split_mode='random'``.
    """
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if gamma is None:
        gamma = 100.0 if dataset == 'ml-1m' else 10.0

    out_dir = ensure_results_dir('head_tail_analysis'
                                 if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(
        dataset, split_mode=split_mode, split_seed=split_seed)
    labels = _quantile_labels(n_buckets)

    if bucket_by == 'item_popularity':
        bucket_of = _item_popularity_buckets(train, n_buckets=n_buckets,
                                             labels=labels)
    elif bucket_by == 'user_activity':
        bucket_of = _user_activity_buckets(train, n_buckets=n_buckets,
                                           labels=labels)
    else:
        raise ValueError(
            f"Unknown bucket_by={bucket_by!r}; "
            "expected 'item_popularity' or 'user_activity'.")

    # ---- Baseline: vanilla EASE (gamma=0) ----
    print("\n[baseline] EASE")
    t0 = time.time()
    base = _fit_ease(train, ease_lambda, rp3_beta, rp3_topK)
    base_summary, base_per_user = _evaluate_by_bucket(
        base, train, test_positive, bucket_of, k=k, bucket_by=bucket_by)
    print(f"  EASE (overall NDCG={base_summary['overall']['ndcg']:.4f}) "
          f"  ({time.time()-t0:.1f}s)")

    if gammas is None:
        return _run_single(
            base, base_summary, base_per_user,
            train, test_positive, bucket_of, labels,
            dataset, k, ease_lambda, gamma, rp3_beta, rp3_topK,
            graph_source, normalise, bucket_by, out_dir)
    else:
        return _run_gamma_sweep(
            base, base_summary, base_per_user,
            train, test_positive, bucket_of, labels,
            dataset, k, ease_lambda, gammas, rp3_beta, rp3_topK,
            graph_source, normalise, bucket_by, out_dir)


def _run_single(base, base_summary, base_per_user,
                train, test_positive, bucket_of, labels,
                dataset, k, ease_lambda, gamma, rp3_beta, rp3_topK,
                graph_source, normalise, bucket_by, out_dir):
    # ---- Laplacian ----
    print(f"\n[Laplacian] source={graph_source}, gamma={gamma}, "
          f"normalise={normalise}, bucket_by={bucket_by}")
    t0 = time.time()
    lap = _fit_laplacian(train, ease_lambda, rp3_beta, rp3_topK,
                         gamma, graph_source, normalise)
    lap_summary, lap_per_user = _evaluate_by_bucket(
        lap, train, test_positive, bucket_of, k=k, bucket_by=bucket_by)
    print(f"  Laplacian (overall NDCG={lap_summary['overall']['ndcg']:.4f}) "
          f"  ({time.time()-t0:.1f}s)")

    # ---- Assemble result table ----
    rows = []
    for bucket in ('overall',) + labels:
        for metric in ('ndcg', 'recall', 'hit'):
            base_val = base_summary[bucket][metric]
            lap_val = lap_summary[bucket][metric]
            row = {
                'dataset': dataset,
                'graph_source': graph_source,
                'normalise': normalise,
                'bucket_by': bucket_by,
                'bucket': bucket,
                'n_users': base_summary[bucket]['n_users'],
                'metric': metric,
                'EASE': base_val,
                'Laplacian': lap_val,
                'abs_gain': lap_val - base_val,
                'rel_gain': (lap_val - base_val) /
                            base_val if base_val > 0 else 0.0,
            }
            if metric == 'ndcg':
                stat, pval, n, sign, mean_diff, median_diff = \
                    _wilcoxon_paired(
                        base_per_user[bucket], lap_per_user[bucket])
                row['wilcoxon_stat'] = stat
                row['wilcoxon_p'] = pval
                row['wilcoxon_n_pairs'] = n
                row['wilcoxon_sign'] = sign
                row['wilcoxon_mean_diff'] = mean_diff
                row['wilcoxon_median_diff'] = median_diff
            rows.append(row)
    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    src_suffix = f'_{graph_source}' if graph_source != 'rp3beta' else ''
    bb_suffix = f'_{bucket_by}' if bucket_by != 'item_popularity' else ''
    csv_path = (out_dir /
                f'head_tail_{dataset}{src_suffix}{bb_suffix}{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot: absolute gain per bucket ----
    ndcg_rows = df[df['metric'] == 'ndcg']
    buckets = ndcg_rows['bucket'].tolist()
    gains = ndcg_rows['abs_gain'].tolist()
    pvals = ndcg_rows['wilcoxon_p'].tolist()
    colors = ['#4c72b0' if g >= 0 else '#c44e52' for g in gains]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(buckets, gains, color=colors)
    for bar, g, p in zip(bars, gains, pvals):
        h = bar.get_height()
        star = ''
        if isinstance(p, float) and not np.isnan(p):
            if p < 0.001:
                star = ' ***'
            elif p < 0.01:
                star = ' **'
            elif p < 0.05:
                star = ' *'
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{g:+.4f}{star}",
                ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel(f'Δ NDCG@{k}  (Laplacian − EASE)')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Head vs tail gain — {dataset} ({graph_source}, '
                 f'by {bucket_by}){title_extra}')
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    fig.tight_layout()
    png_path = (out_dir /
                f'head_tail_{dataset}{src_suffix}{bb_suffix}{suffix}.png')
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    # ---- Console summary ----
    print("\nPer-bucket NDCG summary (paired Wilcoxon p-values):")
    pivot = ndcg_rows.set_index('bucket')[
        ['EASE', 'Laplacian', 'abs_gain', 'rel_gain',
         'wilcoxon_p', 'wilcoxon_n_pairs']]
    print(pivot.to_string())

    return df


def _run_gamma_sweep(base, base_summary, base_per_user,
                     train, test_positive, bucket_of, labels,
                     dataset, k, ease_lambda, gammas, rp3_beta, rp3_topK,
                     graph_source, normalise, bucket_by, out_dir):
    rows = []
    # Seed with the gamma=0 baseline row for every bucket so the sweep
    # plot includes the EASE anchor.
    for bucket in ('overall',) + labels:
        base_val = base_summary[bucket]['ndcg']
        stat, pval, n, sign, mean_diff, median_diff = _wilcoxon_paired(
            base_per_user[bucket], base_per_user[bucket])
        rows.append({
            'dataset': dataset, 'graph_source': graph_source,
            'normalise': normalise, 'bucket_by': bucket_by,
            'gamma': 0.0, 'bucket': bucket,
            'n_users': base_summary[bucket]['n_users'],
            'metric': 'ndcg',
            'EASE': base_val, 'Laplacian': base_val,
            'abs_gain': 0.0, 'rel_gain': 0.0,
            'wilcoxon_stat': stat, 'wilcoxon_p': pval,
            'wilcoxon_n_pairs': n,
            'wilcoxon_sign': sign,
            'wilcoxon_mean_diff': mean_diff,
            'wilcoxon_median_diff': median_diff,
        })

    for gamma in gammas:
        if gamma == 0:
            # Already seeded above.
            continue
        print(f"\n[Laplacian gamma={gamma}] source={graph_source}, "
              f"bucket_by={bucket_by}, normalise={normalise}")
        t0 = time.time()
        lap = _fit_laplacian(train, ease_lambda, rp3_beta, rp3_topK,
                             gamma, graph_source, normalise)
        lap_summary, lap_per_user = _evaluate_by_bucket(
            lap, train, test_positive, bucket_of, k=k, bucket_by=bucket_by)
        print(f"  overall NDCG={lap_summary['overall']['ndcg']:.4f} "
              f"  ({time.time()-t0:.1f}s)")

        for bucket in ('overall',) + labels:
            base_val = base_summary[bucket]['ndcg']
            lap_val = lap_summary[bucket]['ndcg']
            stat, pval, n, sign, mean_diff, median_diff = _wilcoxon_paired(
                base_per_user[bucket], lap_per_user[bucket])
            rows.append({
                'dataset': dataset, 'graph_source': graph_source,
                'normalise': normalise, 'bucket_by': bucket_by,
                'gamma': gamma, 'bucket': bucket,
                'n_users': base_summary[bucket]['n_users'],
                'metric': 'ndcg',
                'EASE': base_val, 'Laplacian': lap_val,
                'abs_gain': lap_val - base_val,
                'rel_gain': (lap_val - base_val) /
                            base_val if base_val > 0 else 0.0,
                'wilcoxon_stat': stat, 'wilcoxon_p': pval,
                'wilcoxon_n_pairs': n,
                'wilcoxon_sign': sign,
                'wilcoxon_mean_diff': mean_diff,
                'wilcoxon_median_diff': median_diff,
            })

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    src_suffix = f'_{graph_source}' if graph_source != 'rp3beta' else ''
    bb_suffix = f'_{bucket_by}' if bucket_by != 'item_popularity' else ''
    csv_path = (out_dir / f'head_tail_gamma_sweep_'
                f'{dataset}{src_suffix}{bb_suffix}{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved gamma-sweep CSV to {csv_path}")

    # ---- Plot: per-bucket absolute gain vs gamma ----
    # Colour by bucket so "head / q1" is consistent across plots.
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
    bucket_colors = dict(zip(labels, palette))
    bucket_colors['overall'] = 'black'

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for bucket in ('overall',) + labels:
        g = df[df['bucket'] == bucket].sort_values('gamma')
        lw = 2.4 if bucket == 'overall' else 1.6
        ls = '--' if bucket == 'overall' else '-'
        ax.plot(g['gamma'], g['abs_gain'], marker='o',
                label=bucket, color=bucket_colors[bucket],
                lw=lw, ls=ls)
    ax.axhline(0, color='grey', lw=0.8)
    # Log x only if all gammas are positive; 0 breaks log scale, so stay linear
    # when gamma=0 is included.
    if df['gamma'].min() > 0:
        ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (Laplacian strength)')
    ax.set_ylabel(f'Δ NDCG@{k}  (Laplacian − EASE)')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Per-bucket gain vs γ — {dataset} ({graph_source}, '
                 f'by {bucket_by}){title_extra}')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    png_path = (out_dir / f'head_tail_gamma_sweep_'
                f'{dataset}{src_suffix}{bb_suffix}{suffix}.png')
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved gamma-sweep plot to {png_path}")

    # ---- Console summary (pivot by bucket × gamma) ----
    print("\nPer-bucket abs_gain matrix (rows = bucket, cols = gamma):")
    pivot = df.pivot_table(index='bucket', columns='gamma',
                           values='abs_gain', aggfunc='first')
    print(pivot.round(5).to_string())

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--gamma', type=float, default=None,
                   help='Single gamma for the single-gamma mode.')
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma list. If given, overrides '
                        '--gamma and runs the per-bucket gain-vs-gamma '
                        'sweep.')
    p.add_argument('--lambda_', dest='ease_lambda', type=float, default=None)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: none.')
    p.add_argument('--n_buckets', type=int, default=3)
    p.add_argument('--bucket_by', default='item_popularity',
                   choices=['item_popularity', 'user_activity'],
                   help='How to form the head/tail buckets. '
                        'item_popularity (default): items bucketed by '
                        'train popularity, and each user contributes '
                        'to multiple buckets. user_activity: users '
                        'bucketed by train interaction count, each '
                        'user contributes to exactly one bucket.')
    p.add_argument('--split_mode', default='temporal',
                   choices=['temporal', 'random'],
                   help='Evaluation protocol. "temporal" (default) uses '
                        'the deterministic temporal split. "random" uses '
                        'a random 80/20 per-user split (primary protocol); '
                        'combine with --split_seed for reproducibility.')
    p.add_argument('--split_seed', type=int, default=0,
                   help='Random-split seed. Only used when '
                        '--split_mode=random. Default: 0.')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]

    run(dataset=args.dataset, k=args.k,
        gamma=args.gamma, ease_lambda=args.ease_lambda,
        rp3_beta=args.rp3_beta, graph_source=args.graph_source,
        normalise=args.normalise,
        n_buckets=args.n_buckets, bucket_by=args.bucket_by,
        split_mode=args.split_mode, split_seed=args.split_seed,
        gammas=gammas)


if __name__ == '__main__':
    main()
