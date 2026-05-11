"""
Smola-Kondor spectral kernel x graph source sweep (roadmap item 5).

For each (graph source, spectral kernel) combination, fits Lap-EASE
using ``M = build_kernel(name, L_graph)`` as the regulariser and reports
NDCG@10/20 vs the EASE baseline. Uses the primary 5-seed protocol by
default; pass ``--split_mode temporal`` for the single-split sanity
check.

Output:
    results/spectral_kernels/spectral_kernels_{dataset}_sym.csv
        rows: (seed, graph_source, kernel, gamma) + metric columns
              + within-seed Wilcoxon vs EASE.
    results/spectral_kernels/spectral_kernels_{dataset}_sym_summary.csv
        same with mean +/- std across seeds.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models import build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta
from models.closed_form_extensions import MahalanobisShrinkEASE
from models.spectral_kernels import KERNELS, build_kernel

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    ensure_results_dir,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_blank_columns,
    wilcoxon_vs_baseline,
)


def _fit_ease_ref(train, ease_lambda, rp3_beta=0.6, rp3_topK=200, k=10, ks=(10, 20)):
    ease_ref = HybridEASE_RP3beta()
    ease_ref.fit(train, method='score', fusion_alpha=1.0,
                 ease_lambda=ease_lambda, rp3_alpha=1.0,
                 rp3_beta=rp3_beta, rp3_topK=rp3_topK)
    return ease_ref


def _fit_kernel(train, ease_lambda, gamma, M, X_target=None):
    """Fit a Mahalanobis-shrink EASE with B0=0, returns the fitted model."""
    model = MahalanobisShrinkEASE()
    model.fit(train, lambdas=ease_lambda, gamma=gamma, M=M, B0=None)
    return model


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None, gammas=(1, 3, 10),
        kernels=('laplacian', 'reg_laplacian', 'heat',
                 'p_step_2', 'inv_cosine'),
        graph_sources=('rp3beta',),
        rp3_beta=0.6, rp3_topK=200, normalise='sym',
        split_mode='random', seeds=None, out_dir=None,
        sigma2_heat=1.0, sigma2_reg=1.0, p_step_a=2.0):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]

    out_path = ensure_results_dir(
        'spectral_kernels' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*62}")
        print(f"[seed={seed}]  split_mode={split_mode}")
        print(f"{'='*62}")
        if split_mode == 'random':
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
        else:
            train, test_pos, _ = load_dataset(
                dataset, split_mode='temporal')

        # EASE reference
        t0 = time.time()
        ease_ref = _fit_ease_ref(train, ease_lambda, rp3_beta, rp3_topK,
                                  k=k, ks=ks)
        res_ease = evaluate_at_ks(ease_ref, train, test_pos, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")
        row = {'dataset': dataset, 'seed': seed, 'split_mode': split_mode,
               'graph_source': '-', 'kernel': 'EASE', 'gamma': 0.0,
               'train_time_s': t_ease}
        row.update(metric_cols_at_ks(res_ease, ks, primary_k=k))
        row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
        rows.append(row)

        for src in graph_sources:
            X = ease_ref.ease.X
            W = build_graph(X, source=src, topK=rp3_topK,
                            rp3_beta=rp3_beta, implicit=True)
            L_dense, _ = build_laplacian(W, normalise=normalise)
            print(f"\n  [graph_source={src}]  building kernels on "
                  f"{L_dense.shape[0]}x{L_dense.shape[0]} L...")

            kernel_kw = {
                'reg_laplacian': {'sigma2': sigma2_reg},
                'heat':          {'sigma2': sigma2_heat},
                'p_step_2':      {'a': p_step_a},
                'p_step_3':      {'a': p_step_a},
            }

            for kname in kernels:
                kw = kernel_kw.get(kname, {})
                t_k0 = time.time()
                M = build_kernel(kname, L_dense, **kw)
                t_kbuild = time.time() - t_k0
                for gamma in gammas:
                    t1 = time.time()
                    model = _fit_kernel(
                        train, ease_lambda, gamma, M)
                    res = evaluate_at_ks(model, train, test_pos, ks=ks)
                    dt = time.time() - t1
                    w = wilcoxon_vs_baseline(res_ease, res, k=k)
                    row = {'dataset': dataset, 'seed': seed,
                           'split_mode': split_mode, 'graph_source': src,
                           'kernel': kname, 'gamma': gamma,
                           'kernel_build_s': t_kbuild,
                           'train_time_s': dt}
                    row.update(metric_cols_at_ks(res, ks, primary_k=k))
                    row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
                    print(f"    {kname:<14s} g={gamma:<6g}  "
                          f"NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                          f"p_vs_EASE={w[1]:.2e}[{w[3]:+d}]  "
                          f"({dt:.1f}s)")
                    rows.append(row)

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'spectral_kernels_{dataset}{suffix}'
    csv_path = out_path / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-seed CSV to {csv_path}")

    # Summary across seeds
    if df['seed'].nunique() > 1:
        key_cols = ['graph_source', 'kernel', 'gamma']
        metric_cols = [c for c in df.columns if '@' in c
                       or c.startswith('train_time_s')
                       or c.startswith('wilcoxon_')]
        agg = {m: ['mean', 'std'] for m in metric_cols if m in df.columns}
        s = df.groupby(key_cols, dropna=False).agg(agg)
        s.columns = [f'{m}_{stat}' for m, stat in s.columns]
        s = s.reset_index()
        spath = out_path / f'{stem}_summary.csv'
        s.to_csv(spath, index=False)
        print(f"Saved summary to {spath}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--gammas', type=str, default='1,3,10')
    p.add_argument('--kernels', type=str,
                   default='laplacian,reg_laplacian,heat,p_step_2,inv_cosine')
    p.add_argument('--graph_sources', type=str, default='rp3beta')
    p.add_argument('--normalise', default='sym', choices=['none', 'sym'])
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--sigma2_heat', type=float, default=1.0)
    p.add_argument('--sigma2_reg', type=float, default=1.0)
    p.add_argument('--p_step_a', type=float, default=2.0)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    gammas = [float(x) for x in args.gammas.split(',') if x.strip()]
    kernels = [x for x in args.kernels.split(',') if x.strip()]
    graph_sources = [x for x in args.graph_sources.split(',') if x.strip()]
    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)

    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, gammas=gammas, kernels=kernels,
        graph_sources=graph_sources, rp3_beta=args.rp3_beta,
        rp3_topK=args.topK, normalise=args.normalise,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir,
        sigma2_heat=args.sigma2_heat, sigma2_reg=args.sigma2_reg,
        p_step_a=args.p_step_a)


if __name__ == '__main__':
    main()
