"""
CEASE + Add-EASE side-information baselines (roadmap item 2).

Compares Jeunen et al.'s closed-form side-information EASE variants
against vanilla EASE and Lap-EASE on the same primary protocol. Side
information is the MovieLens genre indicator matrix.

Without these baselines, the thesis's "Hybrid-Matrix" / "Lap-EASE"
claims have no published reference point for the side-info regime.
"""

from __future__ import annotations
import argparse
import time

import pandas as pd

from data import load_movielens, load_movielens_1m
from evaluation.metrics import evaluate_at_ks
from models.cease import CEASE, AddEASE, build_genre_tag_matrix
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    ensure_results_dir,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_blank_columns,
    wilcoxon_vs_baseline,
)


def _load_movies(dataset):
    """Load the movies metadata DataFrame for genre extraction."""
    if dataset == 'ml-1m':
        _, movies = load_movielens_1m('data/ml-1m', min_interactions=5)
    elif dataset == 'ml-small':
        _, movies = load_movielens('data/ml-latest-small', min_interactions=5)
    else:
        raise ValueError(
            "CEASE/Add-EASE side-info baselines require movie genre data; "
            f"dataset={dataset!r} not supported here.")
    return movies


def run(dataset='ml-1m', k=10, ks=(10, 20), ease_lambda=None,
        cease_alpha_T=(0.1, 1.0, 5.0),
        add_alphas=(0.3, 0.5, 0.7),
        add_lambda_T=(1.0, 10.0),
        rp3_beta=0.6, rp3_topK=200,
        split_mode='random', seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'cease_baselines' if out_dir is None else out_dir)

    movies = _load_movies(dataset)
    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        if split_mode == 'random':
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
        else:
            train, test_pos, _ = load_dataset(
                dataset, split_mode='temporal')

        # Vanilla EASE reference (for Wilcoxon)
        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=ease_lambda, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=rp3_topK)
        res_ease = evaluate_at_ks(ease_ref, train, test_pos, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")
        rows.append({
            'dataset': dataset, 'seed': seed, 'family': 'EASE',
            'config': 'baseline', 'lambda': ease_lambda, 'alpha_T': 0.0,
            'alpha': 0.0, 'lambda_T': 0.0, 'train_time_s': t_ease,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Build T from genres using the EASE encoder (same item space)
        item_enc = ease_ref.ease.item_enc
        T, genres = build_genre_tag_matrix(movies, item_enc)
        print(f"  Built T = (n_genres={len(genres)}, n_items={T.shape[1]}) "
              f"with {T.nnz} nonzeros ({T.nnz / (T.shape[0]*T.shape[1])*100:.1f}% dense)")

        # ---- CEASE ----------------------------------------------------
        print("\n  [CEASE]")
        for a_T in cease_alpha_T:
            t1 = time.time()
            mdl = CEASE()
            mdl.fit(train, T, lambda_=ease_lambda, alpha_T=a_T)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    alpha_T={a_T:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'family': 'CEASE',
                'config': f'alpha_T={a_T}', 'lambda': ease_lambda,
                'alpha_T': a_T, 'alpha': 0.0, 'lambda_T': 0.0,
                'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', w),
            })

        # ---- Add-EASE -------------------------------------------------
        print("\n  [Add-EASE]")
        for lT in add_lambda_T:
            for a in add_alphas:
                t1 = time.time()
                mdl = AddEASE()
                mdl.fit(train, T, lambda_X=ease_lambda, lambda_T=lT, alpha=a)
                res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                print(f"    alpha={a:<4g} lambda_T={lT:<5g}  "
                      f"NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                      f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                rows.append({
                    'dataset': dataset, 'seed': seed, 'family': 'AddEASE',
                    'config': f'alpha={a},lambdaT={lT}',
                    'lambda': ease_lambda, 'alpha_T': 0.0,
                    'alpha': a, 'lambda_T': lT, 'train_time_s': dt,
                    **metric_cols_at_ks(res, ks, primary_k=k),
                    **wilcoxon_columns('wilcoxon_vs_EASE', w),
                })

    df = pd.DataFrame(rows)
    stem = f'cease_baselines_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['family', 'config']
        metric_cols = [c for c in df.columns if '@' in c
                       or c == 'train_time_s' or c.startswith('wilcoxon_')]
        agg = {c: ['mean', 'std'] for c in metric_cols if c in df.columns}
        s = df.groupby(key_cols, dropna=False).agg(agg)
        s.columns = [f'{m}_{st}' for m, st in s.columns]
        s.reset_index().to_csv(out_path / f'{stem}_summary.csv', index=False)
        print(f"Saved {out_path / (stem + '_summary.csv')}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m'],
                   help='netflix-prize not supported (no genre metadata)')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--cease_alpha_T', type=str, default='0.1,1.0,5.0')
    p.add_argument('--add_alphas', type=str, default='0.3,0.5,0.7')
    p.add_argument('--add_lambda_T', type=str, default='1.0,10.0')
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    cease_alpha_T = [float(x) for x in args.cease_alpha_T.split(',')]
    add_alphas = [float(x) for x in args.add_alphas.split(',')]
    add_lambda_T = [float(x) for x in args.add_lambda_T.split(',')]
    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)

    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda,
        cease_alpha_T=cease_alpha_T, add_alphas=add_alphas,
        add_lambda_T=add_lambda_T,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
