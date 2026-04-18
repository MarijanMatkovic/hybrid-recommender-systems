"""
Multi-seed EDLAE vs Laplacian-EDLAE.

Since EDLAE is fully deterministic given the training matrix (its
closed-form uses the expectation of the dropout mask, not a sampled
mask), the meaningful source of variability across runs is the
**train/test split**. This script repeats the EDLAE baseline and a
sweep of Laplacian-EDLAE gammas across ``n_seeds`` random per-user
splits and reports mean ± std for each metric.

Why this matters: on ML-1M the absolute gain of Laplacian-EDLAE over
vanilla EDLAE can be small (sub-0.001 NDCG@10). Reporting a single
split leaves the reader uncertain whether the gain is signal or noise.
mean ± std over 5 splits (plus the Wilcoxon test on per-user NDCG
pooled across seeds) gives two independent lines of evidence.

Note on seeding: seeding the **dropout mask** is not applicable —
``EDLAE.fit`` uses the analytic expectation of the mask, so the fit is
deterministic given X. The only controllable source of variance is the
train/test split, which is what ``--split_seeds`` / ``--n_seeds``
control.

Example:
    python -m experiments.edlae_multiseed \\
        --dataset ml-1m --n_seeds 5 --dropout 0.75 \\
        --gammas 1,3,10 --normalise sym
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

from evaluation.metrics import evaluate_at_ks
from models import EDLAE, build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import (
    bucket_rows,
    bucketed_metrics_at_ks,
    ensure_results_dir,
    item_popularity_buckets,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_paired,
    write_buckets_csv,
)


class _StandaloneWrapper:
    """Shim so ``evaluate_at_ks`` sees the same ``.ease`` + ``.pred``
    interface a hybrid model exposes."""
    def __init__(self, model):
        self.ease = model
        self.pred = model.pred


def _scale_laplacian(L, G_diag_mean):
    l_diag_mean = np.mean(np.diag(L))
    if l_diag_mean > 0:
        return L * (G_diag_mean / l_diag_mean)
    return L


def _summarise(rows, group_keys, metric_keys):
    """Mean ± std over seeds for each (model, config) combination.

    ``rows`` is a list of per-seed result dicts. Returns a DataFrame
    with one row per ``group_keys`` tuple and columns per metric
    (``<m>_mean``, ``<m>_std``, ``<m>_count``).
    """
    df = pd.DataFrame(rows)
    agg_spec = {m: ['mean', 'std', 'count'] for m in metric_keys}
    out = df.groupby(list(group_keys), dropna=False).agg(agg_spec)
    # Flatten the MultiIndex columns produced by ``agg``.
    out.columns = [f'{m}_{stat}' for m, stat in out.columns]
    return out.reset_index()


def _pool_per_user(per_user_dicts):
    """Concatenate per-user NDCG arrays across seeds, prefixing user IDs
    with the seed to keep pairs unique across splits."""
    all_vals = []
    all_uids = []
    for seed, (vals, uids) in per_user_dicts:
        all_vals.append(np.asarray(vals, dtype=float))
        # Make user IDs unique across seeds so paired tests don't
        # accidentally align users from different splits.
        all_uids.extend((seed, u) for u in uids)
    return np.concatenate(all_vals), all_uids


def run(dataset='ml-small', k=10,
        lambda_=None, dropout=0.5,
        gamma=None, gammas=None,
        graph_source='rp3beta', rp3_beta=0.6, topK=200,
        normalise='none', ks=(10, 20),
        n_seeds=3, split_seeds=None,
        out_dir=None):
    """Run EDLAE + Laplacian-EDLAE (swept over gammas) across several
    random splits.

    Parameters
    ----------
    gammas : Sequence[float] | None
        List of Laplacian strengths to test (one Laplacian-EDLAE row
        per gamma per seed). If both ``gamma`` and ``gammas`` are
        given, they are merged. Back-compat: if only the legacy
        single-valued ``gamma`` is supplied, the sweep is a 1-element
        list.
    n_seeds : int
        Number of random splits (ignored if ``split_seeds`` is given).
    split_seeds : Sequence[int] | None
        Explicit list of seeds (e.g. [0, 1, 2, 3, 4]). Overrides
        ``n_seeds``.
    """
    if lambda_ is None:
        lambda_ = 500 if dataset == 'ml-1m' else 200
    ks = tuple(sorted(set(list(ks) + [k])))
    if split_seeds is None:
        split_seeds = list(range(n_seeds))
    split_seeds = list(split_seeds)

    # Normalise the gamma input into a list.
    gamma_list = []
    if gammas is not None:
        gamma_list.extend(float(g) for g in gammas)
    if gamma is not None:
        gamma_list.append(float(gamma))
    if not gamma_list:
        gamma_list = [30.0]
    # De-duplicate while preserving order so the log reads naturally.
    seen = set()
    gamma_list = [g for g in gamma_list
                  if not (g in seen or seen.add(g))]

    out_dir = ensure_results_dir('edlae_multiseed'
                                 if out_dir is None else out_dir)

    per_seed_rows = []
    # Collected per-user NDCG@k so we can pool across seeds for the
    # paired Wilcoxon tests at the end. Each value is
    # (model_key, seed) -> (per_user_ndcg_at_k, per_user_ids).
    per_user_store = {}
    # Per-bucket rows accumulated across (seed, model, gamma) cells.
    # Bucket-of-item maps are re-derived from each seed's train split,
    # since per-seed splits produce different popularity rankings.
    bucket_rows_all = []

    for seed in split_seeds:
        print(f"\n{'=' * 60}")
        print(f" split seed = {seed}")
        print('=' * 60)

        train, test_positive, _ = load_dataset(dataset,
                                               split_mode='random',
                                               split_seed=seed)
        bucket_of = item_popularity_buckets(train, n_buckets=5)

        # --- EASE reference ---
        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=lambda_, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=topK)
        ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
        res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE       NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_ease[f'NDCG@{max(ks)}']:.4f} "
              f"({t_ease:.1f}s)")

        row_ease = {'dataset': dataset, 'seed': seed, 'model': 'EASE',
                    'dropout': 0.0, 'gamma': 0.0,
                    'graph_source': None, 'normalise': None,
                    'train_time_s': t_ease}
        row_ease.update(metric_cols_at_ks(res_ease, ks, primary_k=k))
        per_seed_rows.append(row_ease)
        per_user_store[('EASE', 0.0, seed)] = (
            res_ease['per_user_ndcg'][k],
            res_ease['per_user_ids'],
        )
        _ease_buckets = bucketed_metrics_at_ks(
            ease_ref, train, test_positive, ks=ks, bucket_of=bucket_of)
        bucket_rows_all.extend(
            bucket_rows({'dataset': dataset, 'seed': seed,
                         'model': 'EASE',
                         'dropout': 0.0, 'gamma': 0.0,
                         'graph_source': None, 'normalise': None},
                        _ease_buckets, ks, n_buckets=5))

        # --- EDLAE baseline ---
        t0 = time.time()
        edlae = EDLAE()
        edlae.fit(train, lambda_=lambda_, dropout=dropout)
        t_edl = time.time() - t0
        res_edl = evaluate_at_ks(_StandaloneWrapper(edlae), train,
                                 test_positive, ks=ks)
        print(f"  EDLAE      NDCG@{k}={res_edl[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_edl[f'NDCG@{max(ks)}']:.4f} "
              f"({t_edl:.1f}s)")

        row_edl = {'dataset': dataset, 'seed': seed, 'model': 'EDLAE',
                   'dropout': dropout, 'gamma': 0.0,
                   'graph_source': None, 'normalise': None,
                   'train_time_s': t_edl}
        row_edl.update(metric_cols_at_ks(res_edl, ks, primary_k=k))
        per_seed_rows.append(row_edl)
        per_user_store[('EDLAE', 0.0, seed)] = (
            res_edl['per_user_ndcg'][k],
            res_edl['per_user_ids'],
        )
        _edl_buckets = bucketed_metrics_at_ks(
            _StandaloneWrapper(edlae), train, test_positive,
            ks=ks, bucket_of=bucket_of)
        bucket_rows_all.extend(
            bucket_rows({'dataset': dataset, 'seed': seed,
                         'model': 'EDLAE',
                         'dropout': dropout, 'gamma': 0.0,
                         'graph_source': None, 'normalise': None},
                        _edl_buckets, ks, n_buckets=5))

        # --- Build L once per seed (depends on the split's train X) ---
        X = ease_ref.ease.X
        W = build_graph(X, source=graph_source, topK=topK,
                        rp3_beta=rp3_beta, implicit=True)
        L_dense, _ = build_laplacian(W, normalise=normalise)
        G_diag_mean = float(np.mean(np.array(
            X.multiply(X).sum(axis=0)).flatten()))
        L_scaled = _scale_laplacian(L_dense, G_diag_mean)

        # --- Laplacian-EDLAE gamma sweep ---
        for g_val in gamma_list:
            t0 = time.time()
            edlae_lap = EDLAE()
            edlae_lap.fit_laplacian(train, L_scaled=L_scaled,
                                    lambda_=lambda_, dropout=dropout,
                                    gamma=g_val)
            t_lap = time.time() - t0
            res_lap = evaluate_at_ks(_StandaloneWrapper(edlae_lap), train,
                                     test_positive, ks=ks)
            print(f"  EDLAE-Lap gamma={g_val:<5g} "
                  f"NDCG@{k}={res_lap[f'NDCG@{k}']:.4f}  "
                  f"NDCG@{max(ks)}={res_lap[f'NDCG@{max(ks)}']:.4f} "
                  f"({t_lap:.1f}s)")

            row_lap = {'dataset': dataset, 'seed': seed,
                       'model': 'EDLAE-Laplacian',
                       'dropout': dropout, 'gamma': g_val,
                       'graph_source': graph_source,
                       'normalise': normalise,
                       'train_time_s': t_lap}
            row_lap.update(metric_cols_at_ks(res_lap, ks, primary_k=k))
            per_seed_rows.append(row_lap)
            per_user_store[('EDLAE-Laplacian', g_val, seed)] = (
                res_lap['per_user_ndcg'][k],
                res_lap['per_user_ids'],
            )
            _lap_buckets = bucketed_metrics_at_ks(
                _StandaloneWrapper(edlae_lap), train, test_positive,
                ks=ks, bucket_of=bucket_of)
            bucket_rows_all.extend(
                bucket_rows({'dataset': dataset, 'seed': seed,
                             'model': 'EDLAE-Laplacian',
                             'dropout': dropout, 'gamma': g_val,
                             'graph_source': graph_source,
                             'normalise': normalise},
                            _lap_buckets, ks, n_buckets=5))

    # ---- Long-form per-seed CSV ----
    df = pd.DataFrame(per_seed_rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'edlae_multiseed_{dataset}{suffix}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-seed CSV to {csv_path}")

    # Companion per-(seed, bucket) CSV. Consumers typically group by
    # ``bucket`` and aggregate mean ± std across ``seed`` to get a
    # per-bucket CI just like the accuracy summary below.
    bpath = write_buckets_csv(out_dir, stem, bucket_rows_all)
    if bpath is not None:
        print(f"Saved per-bucket per-seed CSV to {bpath}")

    # ---- Aggregated summary ----
    # All accuracy + diversity metrics at all ks are aggregated across
    # seeds so the summary CSV carries mean/std/count for every
    # metric consumers might want to plot.
    from experiments._shared import STANDARD_METRICS
    metric_cols = [f'{m}@{kk}' for kk in ks for m in STANDARD_METRICS]
    summary = _summarise(per_seed_rows,
                         group_keys=('dataset', 'model',
                                     'dropout', 'gamma', 'graph_source'),
                         metric_keys=metric_cols)
    sum_path = out_dir / f'{stem}_summary.csv'
    summary.to_csv(sum_path, index=False)
    print(f"Saved aggregated summary to {sum_path}")

    # ---- Paired Wilcoxon per gamma, pooled across seeds ----
    # (stat, p, n_pairs, sign, median_diff) reported both vs EASE and
    # vs vanilla EDLAE. The vs-EDLAE number is the scientifically
    # interesting one: does the Laplacian genuinely beat plain EDLAE
    # once split-to-split noise is averaged out?
    ease_pool = _pool_per_user(
        [(s, per_user_store[('EASE', 0.0, s)]) for s in split_seeds])
    edlae_pool = _pool_per_user(
        [(s, per_user_store[('EDLAE', 0.0, s)]) for s in split_seeds])

    wilcoxon_rows = []
    print("\nPaired Wilcoxon (pooled across seeds):")
    for g_val in gamma_list:
        lap_pool = _pool_per_user(
            [(s, per_user_store[('EDLAE-Laplacian', g_val, s)])
             for s in split_seeds])
        w_vs_ease  = wilcoxon_paired(ease_pool,  lap_pool)
        w_vs_edlae = wilcoxon_paired(edlae_pool, lap_pool)
        # 6-tuple: (stat, p, n_pairs, sign, mean_diff, median_diff).
        # ``sign`` is based on the mean so the column it pairs with is
        # ``mean_diff``; ``median_diff`` is reported alongside for
        # back-compat and as an outlier-robust magnitude.
        wilcoxon_rows.append({
            'gamma': g_val,
            'wilcoxon_vs_EASE_stat':          w_vs_ease[0],
            'wilcoxon_vs_EASE_p':             w_vs_ease[1],
            'wilcoxon_vs_EASE_n_pairs':       w_vs_ease[2],
            'wilcoxon_vs_EASE_sign':          w_vs_ease[3],
            'wilcoxon_vs_EASE_mean_diff':     w_vs_ease[4],
            'wilcoxon_vs_EASE_median_diff':   w_vs_ease[5],
            'wilcoxon_vs_EDLAE_stat':         w_vs_edlae[0],
            'wilcoxon_vs_EDLAE_p':            w_vs_edlae[1],
            'wilcoxon_vs_EDLAE_n_pairs':      w_vs_edlae[2],
            'wilcoxon_vs_EDLAE_sign':         w_vs_edlae[3],
            'wilcoxon_vs_EDLAE_mean_diff':    w_vs_edlae[4],
            'wilcoxon_vs_EDLAE_median_diff':  w_vs_edlae[5],
        })
        sign_str_e  = (f"{w_vs_ease[3]:+d}"
                       if w_vs_ease[3]  != 0 else ' 0')
        sign_str_d  = (f"{w_vs_edlae[3]:+d}"
                       if w_vs_edlae[3] != 0 else ' 0')
        print(f"  gamma={g_val:<5g}  "
              f"vs EASE  p={w_vs_ease[1]:.2e}[{sign_str_e}] "
              f"(mean_diff={w_vs_ease[4]:+.4f})   "
              f"vs EDLAE p={w_vs_edlae[1]:.2e}[{sign_str_d}] "
              f"(mean_diff={w_vs_edlae[4]:+.4f})")

    w_df = pd.DataFrame(wilcoxon_rows)
    w_path = out_dir / f'{stem}_wilcoxon.csv'
    w_df.to_csv(w_path, index=False)
    print(f"Saved Wilcoxon summary to {w_path}")

    # ---- Pretty-print NDCG@k column comparison ----
    print("\nMean ± std across seeds (NDCG):")
    for kk in ks:
        col_mean = f'NDCG@{kk}_mean'
        col_std  = f'NDCG@{kk}_std'
        if col_mean in summary.columns:
            line_rows = summary[['model', 'gamma', col_mean, col_std]]
            print(f"\n  NDCG@{kk}")
            for _, r in line_rows.iterrows():
                mu = r[col_mean]
                sd = r[col_std] if pd.notna(r[col_std]) else 0.0
                tag = r['model']
                if r['model'] == 'EDLAE-Laplacian':
                    tag = f"EDLAE-Lap γ={r['gamma']:g}"
                print(f"    {tag:<22} {mu:.4f} ± {sd:.4f}")

    # ---- Bar plot: NDCG@k per (model, gamma) with ± std whiskers ----
    col_mean = f'NDCG@{k}_mean'
    col_std  = f'NDCG@{k}_std'
    if col_mean in summary.columns:
        fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(summary)), 4))
        xs = np.arange(len(summary))
        labels = []
        for _, r in summary.iterrows():
            if r['model'] == 'EDLAE-Laplacian':
                labels.append(f"EDLAE-Lap\nγ={r['gamma']:g}")
            else:
                labels.append(r['model'])
        ax.bar(xs, summary[col_mean],
               yerr=summary[col_std].fillna(0.0),
               capsize=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(f'NDCG@{k} (mean ± std, n={len(split_seeds)} seeds)')
        title_extra = f' [sym L]' if normalise == 'sym' else ''
        ax.set_title(f'EDLAE multi-seed — {dataset}{title_extra}')
        ax.grid(True, axis='y', ls=':', alpha=0.5)
        fig.tight_layout()
        png_path = out_dir / f'{stem}.png'
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        print(f"Saved plot to {png_path}")

    return df, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--lambda_', dest='lambda_', type=float, default=None)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--gamma', type=float, default=None,
                   help='[Legacy] Single Laplacian strength. Prefer '
                        '--gammas for a sweep.')
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma values to sweep '
                        '(e.g. "1,3,10"). One Laplacian-EDLAE row '
                        'per gamma per seed.')
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'])
    p.add_argument('--ks', type=str, default='10,20',
                   help='Comma-separated list of cut-offs for multi-k '
                        'evaluation. Default: "10,20".')
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--split_seeds', type=str, default=None,
                   help='Explicit comma-separated seed list; overrides '
                        '--n_seeds if given.')
    args = p.parse_args()

    split_seeds = None
    if args.split_seeds:
        split_seeds = [int(x) for x in args.split_seeds.split(',')]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',') if x.strip()]

    run(dataset=args.dataset, k=args.k,
        lambda_=args.lambda_, dropout=args.dropout,
        gamma=args.gamma, gammas=gammas,
        graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        normalise=args.normalise, ks=ks,
        n_seeds=args.n_seeds, split_seeds=split_seeds)


if __name__ == '__main__':
    main()
