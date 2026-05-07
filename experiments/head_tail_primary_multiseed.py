"""
Head/tail breakdown for Lap-EASE at gamma=3 across all 5 primary-protocol seeds.

Motivation
----------
The existing head/tail analysis (results/head_tail_analysis/*) uses the
temporal split (single seed). On temporal, gamma=3 Lap-EASE shows a strong
q1 (most-active users) gain of +0.0207 and a q5 loss of -0.0017. But the
temporal protocol has a different gamma optimum (~50-100) and is reported only
as a secondary check; the primary protocol (random 80/20 x 5 seeds) is where
headline numbers live.

This script runs the same user-activity head/tail analysis across all 5 primary
seeds and pools the ~30k per-user NDCG observations for a definitive per-bucket
Wilcoxon test. The thesis needs to report whether the head bias:

  a) holds on the primary protocol too => honest disclosure that Laplacian
     regularises toward the popular subspace
  b) disappears / reverses => the temporal q1 gain was a protocol artefact

Bucket scheme: user_activity quintiles (q1 = most active, q5 = least active).
This mirrors the existing temporal result so comparisons are apples-to-apples.

Outputs
-------
results/head_tail_analysis/head_tail_primary_multiseed_ml-1m_sym.csv
    Per-seed x per-bucket rows with EASE/Lap NDCG, abs_gain, within-seed
    Wilcoxon p.

results/head_tail_analysis/head_tail_primary_multiseed_ml-1m_sym_summary.csv
    Per-bucket: mean +/- std across seeds + pooled ~30k Wilcoxon.

results/head_tail_analysis/head_tail_primary_multiseed_ml-1m_sym.png
    Bar chart with +/- std error bars and pooled p-value annotations.

Example
-------
    python -m experiments.head_tail_primary_multiseed --dataset ml-1m
    python -m experiments.head_tail_primary_multiseed --dataset ml-1m --gamma 3
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

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
    wilcoxon_paired,
    PRIMARY_SPLIT_SEEDS,
)
from experiments.head_tail_analysis import (
    _evaluate_by_bucket,
    _user_activity_buckets,
    _quantile_labels,
    _fit_ease,
    _fit_laplacian,
)


def run(dataset='ml-1m', k=10,
        gamma=3.0, ease_lambda=None,
        rp3_beta=0.6, rp3_topK=200,
        graph_source='rp3beta', normalise='sym',
        n_buckets=5, seeds=None,
        out_dir=None):
    """Run user-activity head/tail analysis for Lap-EASE at ``gamma``
    across all ``seeds`` on the primary random 80/20 protocol.

    Pools per-user NDCG observations across seeds for a ~30k-observation
    paired Wilcoxon per bucket.

    Parameters
    ----------
    gamma : float
        Laplacian regularisation strength. Default 3.0 (best primary-protocol
        gamma from gs_ease_multiseed/baselines/primary experiments).
    n_buckets : int
        Number of user-activity quintiles (default 5). q1 = most active.
    seeds : list[int] or None
        Which primary seeds to include. Defaults to PRIMARY_SPLIT_SEEDS (0..4).
    """
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)

    out_dir = ensure_results_dir(
        'head_tail_analysis' if out_dir is None else out_dir)

    labels = tuple(_quantile_labels(n_buckets))
    all_buckets = ('overall',) + labels

    # Accumulators ----------------------------------------------------------
    per_seed_rows = []      # one row per (seed, bucket)
    # For pooled Wilcoxon: aligned lists of per-user NDCG values
    pool_ease = {b: [] for b in all_buckets}
    pool_lap  = {b: [] for b in all_buckets}
    pool_tags = {b: [] for b in all_buckets}   # (seed, user_id) tuples

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"[seed={seed}]  random 80/20 primary protocol")
        print(f"{'='*55}")
        train, test_positive, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)

        bucket_of = _user_activity_buckets(
            train, n_buckets=n_buckets, labels=labels)

        # Fit EASE baseline
        t0 = time.time()
        base = _fit_ease(train, ease_lambda, rp3_beta, rp3_topK)
        base_summary, base_per_user = _evaluate_by_bucket(
            base, train, test_positive, bucket_of,
            k=k, bucket_by='user_activity')
        print(f"  EASE overall NDCG@{k}={base_summary['overall']['ndcg']:.4f}"
              f"  ({time.time()-t0:.1f}s)")

        # Fit Lap-EASE
        t0 = time.time()
        lap = _fit_laplacian(train, ease_lambda, rp3_beta, rp3_topK,
                             gamma, graph_source, normalise)
        lap_summary, lap_per_user = _evaluate_by_bucket(
            lap, train, test_positive, bucket_of,
            k=k, bucket_by='user_activity')
        print(f"  Lap-EASE(g={gamma}) NDCG@{k}={lap_summary['overall']['ndcg']:.4f}"
              f"  ({time.time()-t0:.1f}s)")

        # Accumulate per-seed rows and pool
        for bucket in all_buckets:
            ease_val = base_summary[bucket]['ndcg']
            lap_val  = lap_summary[bucket]['ndcg']
            n_users  = base_summary[bucket]['n_users']
            abs_gain = lap_val - ease_val

            # Within-seed Wilcoxon
            stat, p, n, sign, mean_diff, median_diff = wilcoxon_paired(
                base_per_user[bucket], lap_per_user[bucket])

            per_seed_rows.append({
                'dataset': dataset, 'seed': seed,
                'graph_source': graph_source, 'normalise': normalise,
                'gamma': gamma, 'bucket_by': 'user_activity',
                'bucket': bucket, 'n_users': n_users,
                'EASE_ndcg': ease_val,
                'Lap_ndcg': lap_val,
                'abs_gain': abs_gain,
                'within_seed_wil_stat': stat,
                'within_seed_wil_p': p,
                'within_seed_wil_n_pairs': n,
                'within_seed_wil_sign': sign,
                'within_seed_wil_mean_diff': mean_diff,
                'within_seed_wil_median_diff': median_diff,
            })

            # Pool: align by user_id, tag with (seed, uid) to prevent false
            # cross-seed pairing inside wilcoxon_paired.
            base_arr, base_uids = base_per_user[bucket]
            lap_arr,  lap_uids  = lap_per_user[bucket]
            base_map = dict(zip(base_uids, base_arr))
            lap_map  = dict(zip(lap_uids,  lap_arr))
            common = sorted(set(base_map) & set(lap_map))
            for uid in common:
                pool_ease[bucket].append(base_map[uid])
                pool_lap[bucket].append(lap_map[uid])
                pool_tags[bucket].append((seed, uid))

        # Brief per-seed summary
        print(f"\n  Per-bucket abs_gain@gamma={gamma}:")
        for b in all_buckets:
            seed_df_b = [r for r in per_seed_rows
                         if r['seed'] == seed and r['bucket'] == b]
            if seed_df_b:
                g = seed_df_b[0]['abs_gain']
                p = seed_df_b[0]['within_seed_wil_p']
                print(f"    {b:>10s}  {g:+.5f}  (p={p:.3e})")

    # ---- Per-seed CSV ---------------------------------------------------
    df_seed = pd.DataFrame(per_seed_rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'head_tail_primary_multiseed_{dataset}{suffix}'
    csv_path = out_dir / f'{stem}.csv'
    df_seed.to_csv(csv_path, index=False)
    print(f"\nSaved per-seed CSV to {csv_path}")

    # ---- Summary: mean +/- std + pooled Wilcoxon -------------------------
    summary_rows = []
    for bucket in all_buckets:
        bucket_df = df_seed[df_seed['bucket'] == bucket]
        ease_mean = float(bucket_df['EASE_ndcg'].mean())
        ease_std  = float(bucket_df['EASE_ndcg'].std(ddof=1))
        lap_mean  = float(bucket_df['Lap_ndcg'].mean())
        lap_std   = float(bucket_df['Lap_ndcg'].std(ddof=1))
        gain_mean = float(bucket_df['abs_gain'].mean())
        gain_std  = float(bucket_df['abs_gain'].std(ddof=1))

        # Pooled Wilcoxon across all seeds
        base_pool = np.array(pool_ease[bucket], dtype=float)
        lap_pool  = np.array(pool_lap[bucket],  dtype=float)
        tags      = pool_tags[bucket]
        stat, p, n, sign, mean_diff, median_diff = wilcoxon_paired(
            (base_pool, tags), (lap_pool, tags))

        summary_rows.append({
            'dataset': dataset, 'graph_source': graph_source,
            'normalise': normalise, 'gamma': gamma,
            'bucket_by': 'user_activity',
            'bucket': bucket, 'n_seeds': len(seeds),
            'ease_ndcg_mean': ease_mean, 'ease_ndcg_std': ease_std,
            'lap_ndcg_mean': lap_mean,   'lap_ndcg_std': lap_std,
            'abs_gain_mean': gain_mean,  'abs_gain_std': gain_std,
            'pooled_wil_stat': stat,
            'pooled_wil_p': p,
            'pooled_wil_n_pairs': n,
            'pooled_wil_sign': sign,
            'pooled_wil_mean_diff': mean_diff,
            'pooled_wil_median_diff': median_diff,
        })

    df_summary = pd.DataFrame(summary_rows)
    summ_path = out_dir / f'{stem}_summary.csv'
    df_summary.to_csv(summ_path, index=False)
    print(f"Saved summary CSV to {summ_path}")

    # ---- Console report --------------------------------------------------
    print(f"\nPer-bucket summary (mean +/- std, pooled Wilcoxon, "
          f"n={df_summary['pooled_wil_n_pairs'].max():.0f}):")
    disp = df_summary.set_index('bucket')[
        ['ease_ndcg_mean', 'lap_ndcg_mean',
         'abs_gain_mean', 'abs_gain_std',
         'pooled_wil_p', 'pooled_wil_sign',
         'pooled_wil_n_pairs']]
    print(disp.round(6).to_string())

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    gain_means = df_summary.set_index('bucket')['abs_gain_mean']
    gain_stds  = df_summary.set_index('bucket')['abs_gain_std']
    pool_ps    = df_summary.set_index('bucket')['pooled_wil_p']
    colors = ['#4c72b0' if g >= 0 else '#c44e52'
              for g in [gain_means[b] for b in all_buckets]]
    xs = list(range(len(all_buckets)))

    bars = ax.bar(xs, [gain_means[b] for b in all_buckets],
                  color=colors, alpha=0.85, width=0.6)
    ax.errorbar(xs,
                [gain_means[b] for b in all_buckets],
                yerr=[gain_stds[b] for b in all_buckets],
                fmt='none', color='black', capsize=4, lw=1.4)

    for i, bucket in enumerate(all_buckets):
        g = gain_means[bucket]
        p = pool_ps[bucket]
        std = gain_stds[bucket]
        star = ''
        if isinstance(p, float) and not np.isnan(p):
            if p < 0.001:
                star = '***'
            elif p < 0.01:
                star = '**'
            elif p < 0.05:
                star = '*'
        offset = std + abs(g) * 0.03 + 0.0005
        h = g + offset if g >= 0 else g - offset
        va = 'bottom' if g >= 0 else 'top'
        ax.text(i, h, f"{g:+.4f}{star}", ha='center', va=va, fontsize=8.5)

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(all_buckets))
    ax.set_xlabel('User activity quintile  (q1 = most active)')
    ax.set_ylabel(f'Mean delta NDCG@{k}  (Lap-EASE - EASE, {len(seeds)} seeds)')
    title_extra = ' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Head/tail gain by user activity -- {dataset} '
                 f'(gamma={gamma}{title_extra}, primary protocol)')
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    fig.tight_layout()
    png_path = out_dir / f'{stem}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    return df_seed, df_summary


def main():
    p = argparse.ArgumentParser(
        description='Multi-seed head/tail analysis for Lap-EASE on the '
                    'primary (random 80/20 x 5-seed) protocol.')
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--gamma', type=float, default=3.0,
                   help='Laplacian gamma to evaluate (default: 3.0, '
                        'the best primary-protocol gamma).')
    p.add_argument('--lambda_', dest='ease_lambda', type=float, default=None,
                   help='EASE regularisation lambda. Defaults to 500 '
                        '(ml-1m) or 200 (ml-small).')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--normalise', default='sym',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: sym.')
    p.add_argument('--n_buckets', type=int, default=5,
                   help='Number of user-activity quintiles (default 5).')
    p.add_argument('--seeds', type=str, default='0,1,2,3,4',
                   help='Comma-separated seed list (default: 0,1,2,3,4).')
    p.add_argument('--out_dir', default=None,
                   help='Override results directory. Default: '
                        '"results/head_tail_analysis".')
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    run(dataset=args.dataset, k=args.k,
        gamma=args.gamma, ease_lambda=args.ease_lambda,
        rp3_beta=args.rp3_beta, rp3_topK=args.topK,
        graph_source=args.graph_source, normalise=args.normalise,
        n_buckets=args.n_buckets, seeds=seeds,
        out_dir=args.out_dir)


if __name__ == '__main__':
    main()
