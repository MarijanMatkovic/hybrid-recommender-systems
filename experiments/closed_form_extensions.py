"""
Closed-form Lap-EASE extension sweep -- roadmap items 3, 4, 6.

Three families compared on the primary 5-seed protocol against the
EASE / Lap-EASE baselines:

  3. Gram-shrinkage EASE      : G_tilde = (1-rho) G + rho S_graph
  4. Multi-Laplacian EASE     : M = sum_i alpha_i L_i (collab + content)
  6. Mahalanobis-shrink EASE  : B0 in {W^2, D^-1 W, exp(-tL), V_k V_k^T}

Each family produces one CSV row per (config, seed) plus a Wilcoxon
test against EASE. Per-seed Wilcoxon stats are aggregated to mean +/-
std across seeds.

The Mahalanobis shrinkage anchors are deliberately limited to ones
that are cheap to construct from existing graph machinery.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import eigh

from evaluation.metrics import evaluate_at_ks
from models import build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta
from models.closed_form_extensions import (
    GramShrinkEASE, MultiLaplacianEASE, MahalanobisShrinkEASE,
)

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    ensure_results_dir,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_blank_columns,
    wilcoxon_vs_baseline,
)


def _fit_ease_ref(train, lam, rp3_beta=0.6, rp3_topK=200):
    h = HybridEASE_RP3beta()
    h.fit(train, method='score', fusion_alpha=1.0, ease_lambda=lam,
          rp3_alpha=1.0, rp3_beta=rp3_beta, rp3_topK=rp3_topK)
    return h


def _build_anchor(B0_name, X, W, L, top_k=128, t_heat=1.0):
    """Construct anchor B0 for Mahalanobis shrinkage."""
    if B0_name == 'zero':
        return np.zeros((X.shape[1], X.shape[1]))
    if B0_name == 'W':
        return W.toarray() if sps.issparse(W) else np.asarray(W)
    if B0_name == 'W2':
        Wd = W.toarray() if sps.issparse(W) else np.asarray(W)
        return Wd @ Wd
    if B0_name == 'rw':  # random walk: D^-1 W
        Wd = W.toarray() if sps.issparse(W) else np.asarray(W)
        d = Wd.sum(axis=1) + 1e-12
        return Wd / d[:, None]
    if B0_name == 'heat':  # exp(-t L) via eigendecomposition (truncated)
        # Use a mild truncation for cost: 256 smallest eigenvectors.
        n = L.shape[0]
        kk = min(top_k, n)
        # eigh returns ascending eigenvalues
        w, V = eigh(L, subset_by_index=(0, kk - 1))
        return (V * np.exp(-t_heat * w)[None, :]) @ V.T
    if B0_name == 'lowpass':  # V_k V_k^T (GF-CF ideal low-pass)
        n = L.shape[0]
        kk = min(top_k, n)
        _, V = eigh(L, subset_by_index=(0, kk - 1))
        return V @ V.T
    raise ValueError(f"Unknown B0 anchor: {B0_name}")


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None, rp3_beta=0.6, rp3_topK=200, normalise='sym',
        split_mode='random', seeds=None, out_dir=None,
        gram_rhos=(0.05, 0.1, 0.2),
        multi_lap_weights=((1.0,), (0.5, 0.5)),
        mahal_anchors=('W2', 'rw', 'lowpass'),
        gammas=(1, 3, 10), top_k_eig=128, t_heat=1.0):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'closed_form_extensions' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        if split_mode == 'random':
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
        else:
            train, test_pos, _ = load_dataset(
                dataset, split_mode='temporal')

        # EASE reference
        t0 = time.time()
        ease_ref = _fit_ease_ref(train, ease_lambda, rp3_beta, rp3_topK)
        res_ease = evaluate_at_ks(ease_ref, train, test_pos, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")

        row = {'dataset': dataset, 'seed': seed, 'family': 'EASE',
               'config': 'baseline', 'gamma': 0.0, 'rho': 0.0,
               'train_time_s': t_ease}
        row.update(metric_cols_at_ks(res_ease, ks, primary_k=k))
        row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
        rows.append(row)

        # Build shared graph + Laplacian once per seed
        X = ease_ref.ease.X
        W = build_graph(X, source='rp3beta', topK=rp3_topK,
                        rp3_beta=rp3_beta, implicit=True)
        L_dense, _ = build_laplacian(W, normalise=normalise)
        L_dense = np.asarray(L_dense)

        # ---- 3) GramShrinkEASE -----------------------------------------
        print("\n  [3] Gram-shrinkage EASE")
        S_graph = W.toarray() if sps.issparse(W) else np.asarray(W)
        for rho in gram_rhos:
            t1 = time.time()
            mdl = GramShrinkEASE()
            mdl.fit(train, lambdas=ease_lambda, rho=rho, S_graph=S_graph)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    rho={rho:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}"
                  f"  p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            row = {'dataset': dataset, 'seed': seed,
                   'family': 'GramShrink', 'config': f'rho={rho}',
                   'gamma': 0.0, 'rho': rho, 'train_time_s': dt}
            row.update(metric_cols_at_ks(res, ks, primary_k=k))
            row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
            rows.append(row)

        # ---- 4) MultiLaplacianEASE ------------------------------------
        # On ml-1m we don't have multiple graphs natively wired up; use
        # the rp3beta + p3alpha pair as a proxy for "collab + smoother".
        print("\n  [4] Multi-Laplacian EASE")
        W2 = build_graph(X, source='p3alpha', topK=rp3_topK,
                         rp3_beta=rp3_beta, implicit=True)
        L2_dense, _ = build_laplacian(W2, normalise=normalise)
        L2_dense = np.asarray(L2_dense)
        for weights in multi_lap_weights:
            for gamma in gammas:
                t1 = time.time()
                mdl = MultiLaplacianEASE()
                laps = [L_dense, L2_dense][:len(weights)]
                mdl.fit(train, lambdas=ease_lambda, gamma=gamma,
                        laplacians=laps, weights=list(weights))
                res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                cfg = f"w={list(weights)}_g={gamma}"
                print(f"    {cfg:<22s}  NDCG@{k}={res[f'NDCG@{k}']:.4f}"
                      f"  p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                row = {'dataset': dataset, 'seed': seed,
                       'family': 'MultiLap', 'config': cfg,
                       'gamma': gamma, 'rho': 0.0, 'train_time_s': dt}
                row.update(metric_cols_at_ks(res, ks, primary_k=k))
                row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
                rows.append(row)

        # ---- 6) MahalanobisShrinkEASE ---------------------------------
        print("\n  [6] Mahalanobis-shrinkage EASE")
        for anchor in mahal_anchors:
            t_a0 = time.time()
            B0 = _build_anchor(anchor, X, W, L_dense, top_k=top_k_eig,
                                t_heat=t_heat)
            t_anchor = time.time() - t_a0
            print(f"    [anchor={anchor}] built in {t_anchor:.1f}s, "
                  f"shape={B0.shape}")
            for gamma in gammas:
                t1 = time.time()
                mdl = MahalanobisShrinkEASE()
                mdl.fit(train, lambdas=ease_lambda, gamma=gamma,
                        M=L_dense, B0=B0)
                res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                cfg = f"{anchor}_g={gamma}"
                print(f"    {cfg:<14s}  NDCG@{k}={res[f'NDCG@{k}']:.4f}"
                      f"  p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                row = {'dataset': dataset, 'seed': seed,
                       'family': 'Mahal', 'config': cfg,
                       'gamma': gamma, 'rho': 0.0, 'train_time_s': dt}
                row.update(metric_cols_at_ks(res, ks, primary_k=k))
                row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
                rows.append(row)

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'closed_form_extensions_{dataset}{suffix}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['family', 'config', 'gamma', 'rho']
        metric_cols = [c for c in df.columns if '@' in c
                       or c == 'train_time_s'
                       or c.startswith('wilcoxon_')]
        agg = {c: ['mean', 'std'] for c in metric_cols if c in df.columns}
        s = df.groupby(key_cols, dropna=False).agg(agg)
        s.columns = [f'{m}_{st}' for m, st in s.columns]
        s = s.reset_index()
        s.to_csv(out_path / f'{stem}_summary.csv', index=False)
        print(f"Saved {out_path / (stem + '_summary.csv')}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--gammas', type=str, default='1,3,10')
    p.add_argument('--gram_rhos', type=str, default='0.05,0.1,0.2')
    p.add_argument('--mahal_anchors', type=str,
                   default='W2,rw,lowpass')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--top_k_eig', type=int, default=128,
                   help='Top-k eigenvectors for heat / lowpass anchors.')
    p.add_argument('--t_heat', type=float, default=1.0)
    p.add_argument('--normalise', default='sym', choices=['none', 'sym'])
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    gammas = [float(x) for x in args.gammas.split(',')]
    gram_rhos = [float(x) for x in args.gram_rhos.split(',')]
    mahal_anchors = [x for x in args.mahal_anchors.split(',') if x.strip()]
    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, rp3_beta=args.rp3_beta,
        rp3_topK=args.topK, normalise=args.normalise,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir,
        gram_rhos=gram_rhos, mahal_anchors=mahal_anchors, gammas=gammas,
        top_k_eig=args.top_k_eig, t_heat=args.t_heat)


if __name__ == '__main__':
    main()
