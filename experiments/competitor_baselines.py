"""
Direct competitor closed-form linear AEs (roadmap items 10, 12, 13).

  10. L^3AE                -- Moon, Park, Lee, CIKM 2025
  12. DAN normalisation    -- Park et al., SIGIR 2025
  13. SVD-AE               -- Hong et al., IJCAI 2024
      (+ SVD-AE on a Turbo-CF / Chebyshev pre-filtered X)

All three are competitors in the closed-form linear-AE family that
must be benchmarked before claiming novelty for Lap-EASE. Each is
swept over its key hyper-parameter (alpha_S / alpha / k respectively)
and compared per seed against vanilla EASE via Wilcoxon.

Output:
    results/competitor_baselines/competitor_baselines_{dataset}.csv
    results/competitor_baselines/competitor_baselines_{dataset}_summary.csv
"""

from __future__ import annotations
import argparse
import time

import pandas as pd

from data import load_movielens, load_movielens_1m
from evaluation.metrics import evaluate_at_ks
from models.dan import DAN_EASE
from models.svd_ae import SVDAE
from models.l3ae import L3AE, build_genre_cosine_similarity
from models.cease import build_genre_tag_matrix
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
    if dataset == 'ml-1m':
        _, movies = load_movielens_1m('data/ml-1m', min_interactions=5)
    elif dataset == 'ml-small':
        _, movies = load_movielens('data/ml-latest-small', min_interactions=5)
    else:
        movies = None
    return movies


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None,
        dan_alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
        svd_ks=(64, 128, 256),
        svd_filters=('identity', 'turbo_cf'),
        l3ae_alphas=(0.1, 1.0, 5.0),
        split_mode='random', seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'competitor_baselines' if out_dir is None else out_dir)

    movies = _load_movies(dataset) if dataset != 'netflix-prize' else None
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
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=ease_lambda, rp3_alpha=1.0,
                     rp3_beta=0.6, rp3_topK=200)
        res_ease = evaluate_at_ks(ease_ref, train, test_pos, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")
        rows.append({
            'dataset': dataset, 'seed': seed, 'family': 'EASE',
            'config': 'baseline', 'train_time_s': t_ease,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # ---- 12) DAN -------------------------------------------------
        print("\n  [12] DAN (degree-aware normalisation)")
        for a in dan_alphas:
            t1 = time.time()
            mdl = DAN_EASE(); mdl.fit(train, lambdas=ease_lambda, alpha=a)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    alpha={a:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'family': 'DAN',
                'config': f'alpha={a}', 'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', w),
            })

        # ---- 13) SVD-AE (plain + with filter) ------------------------
        print("\n  [13] SVD-AE (+/- polynomial filter)")
        for filt in svd_filters:
            for r in svd_ks:
                t1 = time.time()
                mdl = SVDAE()
                fname = None if filt == 'identity' else filt
                fkw = {'alpha': 0.7, 'K': 2} if filt == 'turbo_cf' else None
                mdl.fit(train, k=r, lambdas=10.0,
                        filter_name=fname, filter_kw=fkw)
                res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                print(f"    filt={filt:<10s} k={r:<4d}  "
                      f"NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                      f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                rows.append({
                    'dataset': dataset, 'seed': seed, 'family': 'SVD-AE',
                    'config': f'filt={filt},k={r}', 'train_time_s': dt,
                    **metric_cols_at_ks(res, ks, primary_k=k),
                    **wilcoxon_columns('wilcoxon_vs_EASE', w),
                })

        # ---- 10) L^3AE (needs movies for genre similarity) -----------
        if movies is not None:
            print("\n  [10] L^3AE (genre-cosine S)")
            T, _ = build_genre_tag_matrix(movies, ease_ref.ease.item_enc)
            S = build_genre_cosine_similarity(T)
            for aS in l3ae_alphas:
                t1 = time.time()
                mdl = L3AE(); mdl.fit(train, S, lambdas=ease_lambda,
                                       alpha_S=aS)
                res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                print(f"    alpha_S={aS:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                      f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                rows.append({
                    'dataset': dataset, 'seed': seed, 'family': 'L3AE',
                    'config': f'alpha_S={aS}', 'train_time_s': dt,
                    **metric_cols_at_ks(res, ks, primary_k=k),
                    **wilcoxon_columns('wilcoxon_vs_EASE', w),
                })
        else:
            print("\n  [10] L^3AE -- skipped (no genre metadata for "
                  f"dataset={dataset!r})")

    df = pd.DataFrame(rows)
    stem = f'competitor_baselines_{dataset}'
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
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--dan_alphas', type=str, default='0.0,0.25,0.5,0.75,1.0')
    p.add_argument('--svd_ks', type=str, default='64,128,256')
    p.add_argument('--svd_filters', type=str, default='identity,turbo_cf')
    p.add_argument('--l3ae_alphas', type=str, default='0.1,1.0,5.0')
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    dan_alphas = [float(x) for x in args.dan_alphas.split(',')]
    svd_ks = [int(x) for x in args.svd_ks.split(',')]
    svd_filters = [x for x in args.svd_filters.split(',') if x.strip()]
    l3ae_alphas = [float(x) for x in args.l3ae_alphas.split(',')]

    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda,
        dan_alphas=dan_alphas, svd_ks=svd_ks, svd_filters=svd_filters,
        l3ae_alphas=l3ae_alphas,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
