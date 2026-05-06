"""
Graph-Shrunk EASE (GS-EASE) — multi-seed primary-protocol evaluation.

Sweeps (gamma, W_source) on the primary random 80/20 × 5-seed protocol
and reports:
  - NDCG@10/20 leaderboard alongside EASE and Lap-EASE baselines
  - Paired Wilcoxon vs EASE and vs Lap-EASE at same (gamma, W_source)
  - Per-bucket popularity breakdown

The key test: GS-EASE must beat Lap-EASE (same gamma, same W_source) to
justify its extra complexity.

Implementation note: Lap-EASE and GS-EASE share A = G + λI + γL, so
``np.linalg.inv(A)`` is computed once per (w_source, gamma) pair and
reused for both models.

Example:
    python -m experiments.gs_ease_multiseed \\
        --dataset ml-1m \\
        --gammas 0.3,1,3,10 \\
        --w_sources rp3beta,binary,itemknn \\
        --normalise sym
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
from models import build_graph, build_laplacian
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
    STANDARD_METRICS,
)


class _Wrapper:
    """Minimal shim so evaluate_at_ks sees .pred and .ease.user/item_enc."""

    def __init__(self, B, X, enc_source):
        """
        Parameters
        ----------
        B : np.ndarray  item-item weight matrix
        X : csr_matrix  user-item interaction matrix
        enc_source : object with .user_enc and .item_enc attributes
            (e.g. ease_ref.ease)
        """
        self.pred = X.dot(B)
        self.ease = enc_source   # evaluate_at_ks reads .ease.user_enc / .item_enc


def _scale_laplacian(L, G_diag_mean):
    l_mean = np.mean(np.diag(L))
    return L * (G_diag_mean / l_mean) if l_mean > 0 else L


def _diag_correct(P, Q):
    """Lagrangian diagonal correction: B_ij = Q_ij - P_ij * Q_jj/P_jj."""
    mu = np.diag(Q) / np.diag(P)
    B = Q - P * mu[np.newaxis, :]
    np.fill_diagonal(B, 0.0)
    return B


def _pool(pu_store, key_list):
    """Pool (values, ids) from pu_store across seeds; make ids unique."""
    all_vals, all_ids = [], []
    for seed, key in key_list:
        vals, ids = pu_store[key]
        all_vals.append(np.asarray(vals, dtype=float))
        all_ids.extend((seed, u) for u in ids)
    return np.concatenate(all_vals), all_ids


def _summarise(rows, group_keys, metric_keys):
    df = pd.DataFrame(rows)
    agg = {m: ['mean', 'std', 'count'] for m in metric_keys}
    out = df.groupby(list(group_keys), dropna=False).agg(agg)
    out.columns = [f'{m}_{s}' for m, s in out.columns]
    return out.reset_index()


def run(dataset='ml-small', k=10,
        lambda_=None,
        gammas=None,
        w_sources=None,
        rp3_beta=0.6, topK=200,
        normalise='sym',
        ks=(10, 20),
        split_seeds=None, n_seeds=5,
        out_dir=None):

    if lambda_ is None:
        lambda_ = 500 if dataset == 'ml-1m' else 200
    ks = tuple(sorted(set(list(ks) + [k])))
    if split_seeds is None:
        split_seeds = list(range(n_seeds))
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0]
    gammas = [float(g) for g in gammas]
    if w_sources is None:
        w_sources = ['rp3beta', 'binary', 'itemknn']

    out_dir = ensure_results_dir('gs_ease_multiseed' if out_dir is None else out_dir)

    per_seed_rows = []
    bucket_rows_all = []
    # key schema: ('EASE', None, 0.0, seed)
    #             ('Lap-EASE', w_source, gamma, seed)
    #             ('GS-EASE',  w_source, gamma, seed)
    pu_store = {}

    for seed in split_seeds:
        print(f"\n{'=' * 60}")
        print(f" split seed = {seed}")
        print('=' * 60)

        train, test_positive, _ = load_dataset(dataset,
                                               split_mode='random',
                                               split_seed=seed)
        bucket_of = item_popularity_buckets(train, n_buckets=5)

        # ---- EASE reference ----
        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=lambda_, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=topK)
        ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
        res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE       NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_ease[f'NDCG@{max(ks)}']:.4f} ({t_ease:.1f}s)")

        r = {'dataset': dataset, 'seed': seed, 'model': 'EASE',
             'w_source': None, 'gamma': 0.0, 'normalise': None,
             'train_time_s': t_ease}
        r.update(metric_cols_at_ks(res_ease, ks, primary_k=k))
        per_seed_rows.append(r)
        pu_store[('EASE', None, 0.0, seed)] = (res_ease['per_user_ndcg'][k],
                                                res_ease['per_user_ids'])
        bucket_rows_all.extend(bucket_rows(
            {'dataset': dataset, 'seed': seed, 'model': 'EASE',
             'w_source': None, 'gamma': 0.0, 'normalise': None},
            bucketed_metrics_at_ks(ease_ref, train, test_positive,
                                   ks=ks, bucket_of=bucket_of),
            ks, n_buckets=5))

        # Pre-compute G once (enc and X from ease_ref)
        enc = ease_ref.ease           # carries .user_enc / .item_enc for _Wrapper
        X = ease_ref.ease.X
        G_raw = X.T.dot(X).toarray()
        G_diag_mean = float(np.mean(np.diag(G_raw)))

        # ---- Per-source: Lap-EASE + GS-EASE (share the same A^{-1}) ----
        for w_source in w_sources:
            W_sparse = build_graph(X, source=w_source, topK=topK,
                                   rp3_beta=rp3_beta, implicit=True)
            W_dense  = W_sparse.toarray().astype(np.float64)
            L_raw, _ = build_laplacian(W_sparse, normalise=normalise)
            L_scaled  = _scale_laplacian(L_raw, G_diag_mean)
            LW = L_scaled @ W_dense   # precompute; used for all GS-EASE at this source

            for g_val in gammas:
                # A = G + λI + γL  (shared by Lap-EASE and GS-EASE)
                t0 = time.time()
                A = G_raw + g_val * L_scaled
                np.fill_diagonal(A, np.diag(A) + lambda_)
                P = np.linalg.inv(A)
                t_inv = time.time() - t0

                # -- Lap-EASE --
                B_lap = _diag_correct(P, P @ G_raw)
                res_lap = evaluate_at_ks(_Wrapper(B_lap, X, enc),
                                         train, test_positive, ks=ks)
                print(f"  Lap-EASE   src={w_source:<8} γ={g_val:<5g} "
                      f"NDCG@{k}={res_lap[f'NDCG@{k}']:.4f} ({t_inv:.1f}s)")
                r = {'dataset': dataset, 'seed': seed, 'model': 'Lap-EASE',
                     'w_source': w_source, 'gamma': g_val, 'normalise': normalise,
                     'train_time_s': t_inv}
                r.update(metric_cols_at_ks(res_lap, ks, primary_k=k))
                per_seed_rows.append(r)
                pu_store[('Lap-EASE', w_source, g_val, seed)] = (
                    res_lap['per_user_ndcg'][k], res_lap['per_user_ids'])
                bucket_rows_all.extend(bucket_rows(
                    {'dataset': dataset, 'seed': seed, 'model': 'Lap-EASE',
                     'w_source': w_source, 'gamma': g_val, 'normalise': normalise},
                    bucketed_metrics_at_ks(_Wrapper(B_lap, X, enc),
                                           train, test_positive,
                                           ks=ks, bucket_of=bucket_of),
                    ks, n_buckets=5))

                # -- GS-EASE (same P, different RHS) --
                t0 = time.time()
                R = G_raw + g_val * LW
                B_gs = _diag_correct(P, P @ R)
                t_gs = time.time() - t0
                res_gs = evaluate_at_ks(_Wrapper(B_gs, X, enc),
                                         train, test_positive, ks=ks)
                print(f"  GS-EASE    src={w_source:<8} γ={g_val:<5g} "
                      f"NDCG@{k}={res_gs[f'NDCG@{k}']:.4f} ({t_gs:.1f}s)")
                r = {'dataset': dataset, 'seed': seed, 'model': 'GS-EASE',
                     'w_source': w_source, 'gamma': g_val, 'normalise': normalise,
                     'train_time_s': t_gs}
                r.update(metric_cols_at_ks(res_gs, ks, primary_k=k))
                per_seed_rows.append(r)
                pu_store[('GS-EASE', w_source, g_val, seed)] = (
                    res_gs['per_user_ndcg'][k], res_gs['per_user_ids'])
                bucket_rows_all.extend(bucket_rows(
                    {'dataset': dataset, 'seed': seed, 'model': 'GS-EASE',
                     'w_source': w_source, 'gamma': g_val, 'normalise': normalise},
                    bucketed_metrics_at_ks(_Wrapper(B_gs, X, enc),
                                           train, test_positive,
                                           ks=ks, bucket_of=bucket_of),
                    ks, n_buckets=5))

    # ---- Persist ----
    df_out = pd.DataFrame(per_seed_rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'gs_ease_multiseed_{dataset}{suffix}'

    csv_path = out_dir / f'{stem}.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved per-seed CSV to {csv_path}")

    bpath = write_buckets_csv(out_dir, stem, bucket_rows_all)
    if bpath:
        print(f"Saved per-bucket CSV to {bpath}")

    metric_cols = [f'{m}@{kk}' for kk in ks for m in STANDARD_METRICS]
    summary = _summarise(df_out.to_dict('records'),
                         group_keys=('dataset', 'model', 'w_source',
                                     'gamma', 'normalise'),
                         metric_keys=metric_cols)
    sum_path = out_dir / f'{stem}_summary.csv'
    summary.to_csv(sum_path, index=False)
    print(f"Saved summary to {sum_path}")

    # ---- Paired Wilcoxon ----
    ease_pool = _pool(pu_store, [(s, ('EASE', None, 0.0, s))
                                  for s in split_seeds])
    wilcoxon_rows = []
    print("\nPaired Wilcoxon (pooled across seeds)  [GS-EASE vs EASE | vs Lap-EASE]:")
    for w_source in w_sources:
        for g_val in gammas:
            gs_pool  = _pool(pu_store,
                [(s, ('GS-EASE', w_source, g_val, s)) for s in split_seeds])
            lap_pool = _pool(pu_store,
                [(s, ('Lap-EASE', w_source, g_val, s)) for s in split_seeds])
            w_ease = wilcoxon_paired(ease_pool, gs_pool)
            w_lap  = wilcoxon_paired(lap_pool, gs_pool)
            se = f"{w_ease[3]:+d}" if w_ease[3] != 0 else ' 0'
            sl = f"{w_lap[3]:+d}"  if w_lap[3]  != 0 else ' 0'
            print(f"  {w_source:<8} γ={g_val:<5g}  "
                  f"vs EASE p={w_ease[1]:.2e}[{se}] diff={w_ease[4]:+.4f}  "
                  f"vs Lap  p={w_lap[1]:.2e}[{sl}] diff={w_lap[4]:+.4f}")
            wilcoxon_rows.append({
                'w_source': w_source, 'gamma': g_val,
                'vs_EASE_stat': w_ease[0],   'vs_EASE_p': w_ease[1],
                'vs_EASE_n_pairs': w_ease[2],'vs_EASE_sign': w_ease[3],
                'vs_EASE_mean_diff': w_ease[4], 'vs_EASE_median_diff': w_ease[5],
                'vs_LapEASE_stat': w_lap[0],  'vs_LapEASE_p': w_lap[1],
                'vs_LapEASE_n_pairs': w_lap[2],'vs_LapEASE_sign': w_lap[3],
                'vs_LapEASE_mean_diff': w_lap[4], 'vs_LapEASE_median_diff': w_lap[5],
            })

    w_df = pd.DataFrame(wilcoxon_rows)
    w_path = out_dir / f'{stem}_wilcoxon.csv'
    w_df.to_csv(w_path, index=False)
    print(f"Saved Wilcoxon to {w_path}")

    # ---- Pretty-print summary ----
    col_m = f'NDCG@{k}_mean'
    col_s = f'NDCG@{k}_std'
    print(f"\nMean ± std NDCG@{k} across seeds:")
    for _, r in summary.iterrows():
        tag = (r['model'] if r['model'] == 'EASE'
               else f"{r['model']}[{r['w_source']}] γ={r['gamma']:g}")
        mu = r[col_m]
        sd = r[col_s] if pd.notna(r[col_s]) else 0.0
        print(f"  {tag:<40} {mu:.4f} ± {sd:.4f}")

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(summary)), 4))
    xs = np.arange(len(summary))
    labels = [('EASE' if r['model'] == 'EASE'
                else f"{r['model']}\n[{r['w_source']}]\nγ={r['gamma']:g}")
              for _, r in summary.iterrows()]
    ax.bar(xs, summary[col_m], yerr=summary[col_s].fillna(0), capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=7)
    ax.set_ylabel(f'NDCG@{k} (mean ± std, {len(split_seeds)} seeds)')
    ax.set_title(
        f'GS-EASE vs Lap-EASE — {dataset}{" [sym L]" if normalise == "sym" else ""}')
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    fig.tight_layout()
    png_path = out_dir / f'{stem}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    return df_out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--lambda_', dest='lambda_', type=float, default=None)
    p.add_argument('--gammas', type=str, default='0.3,1,3,10')
    p.add_argument('--w_sources', type=str, default='rp3beta,binary,itemknn')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--normalise', default='sym', choices=['none', 'sym'])
    p.add_argument('--ks', type=str, default='10,20')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--split_seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    split_seeds = (None if not args.split_seeds
                   else [int(x) for x in args.split_seeds.split(',')])
    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())
    gammas = [float(x) for x in args.gammas.split(',') if x.strip()]
    w_sources = [x.strip() for x in args.w_sources.split(',') if x.strip()]

    run(dataset=args.dataset, k=args.k, lambda_=args.lambda_,
        gammas=gammas, w_sources=w_sources,
        rp3_beta=args.rp3_beta, topK=args.topK,
        normalise=args.normalise, ks=ks,
        split_seeds=split_seeds, n_seeds=args.n_seeds,
        out_dir=args.out_dir)


if __name__ == '__main__':
    main()
