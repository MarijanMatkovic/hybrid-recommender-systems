"""
Faruqui-retrofit refinement of EASE B columns (roadmap item 11).

Fits vanilla EASE, then iteratively retrofits each column of B so
that it stays close to the original anchor B[:, i] AND smooths over a
genre-cosine similarity graph. Reports NDCG before vs after retrofit
plus a Wilcoxon test on the per-user gain.

Closed-form fixed point via Jacobi iteration (5-10 sweeps).
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd  # noqa: F401

from data import load_movielens, load_movielens_1m
from evaluation.metrics import evaluate_at_ks
from models.hybrid import HybridEASE_RP3beta
from models.lazy_pred import make_pred
from models.cease import build_genre_tag_matrix
from models.l3ae import build_genre_cosine_similarity
from fusion.retrofit import retrofit

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
        _, movies = load_movielens('data/ml-latest-small',
                                    min_interactions=5)
    else:
        movies = None
    return movies


class _Wrap:
    def __init__(self, ease, pred):
        self.ease = ease; self.pred = pred


def _topk_threshold(S, topK):
    """Per-row top-K thresholding of a dense similarity matrix.
    Without this the Faruqui denominator
    ``denom_i = alpha + sum_j N[i,j]`` is dominated by ~hundreds of
    small entries, which drowns the anchor for any reasonable alpha
    and oversmooths the retrofitted columns.
    """
    if topK is None or topK >= S.shape[1]:
        return S
    out = np.zeros_like(S)
    idx = np.argpartition(-S, topK, axis=1)[:, :topK]
    rows = np.arange(S.shape[0])[:, None]
    out[rows, idx] = S[rows, idx]
    return out


def run(dataset='ml-1m', k=10, ks=(10, 20), ease_lambda=None,
        alphas=(5.0, 20.0, 50.0, 100.0), n_iters=(3, 5, 10),
        topK_graph=20,
        split_mode='random', seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'retrofit_ease' if out_dir is None else out_dir)

    movies = _load_movies(dataset)
    if movies is None:
        raise ValueError(
            "Retrofit experiment requires genre metadata; "
            f"dataset={dataset!r} not supported.")

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        if split_mode == 'random':
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
        else:
            train, test_pos, _ = load_dataset(dataset, split_mode='temporal')

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
            'alpha': 0.0, 'n_iter': 0, 'train_time_s': t_ease,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Build neighbour graph = genre cosine similarity. Threshold to
        # top-K per row so the Faruqui denominator stays close to alpha
        # (full dense N over-smooths -- see _topk_threshold).
        T, _ = build_genre_tag_matrix(movies, ease_ref.ease.item_enc)
        S_full = build_genre_cosine_similarity(T)
        np.fill_diagonal(S_full, 0.0)
        S = _topk_threshold(S_full, topK_graph)
        avg_neighbours = float((S > 0).sum(axis=1).mean())
        print(f"  Genre graph top-K={topK_graph}: avg neighbours/row = "
              f"{avg_neighbours:.1f}, mean weight = {S[S > 0].mean():.3f}")
        B0 = ease_ref.ease.B

        print("\n  [11] Faruqui retrofit of EASE columns over genre cosine")
        for alpha in alphas:
            for ni in n_iters:
                t1 = time.time()
                Q = retrofit(B0, S, alpha=alpha, n_iter=ni,
                             restore_diag_zero=True)
                pred_q = make_pred(ease_ref.ease.X, Q)
                wrapped = _Wrap(ease_ref.ease, pred_q)
                res = evaluate_at_ks(wrapped, train, test_pos, ks=ks)
                dt = time.time() - t1
                w = wilcoxon_vs_baseline(res_ease, res, k=k)
                print(f"    alpha={alpha:<4g} n_iter={ni:<3d}  "
                      f"NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                      f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                rows.append({
                    'dataset': dataset, 'seed': seed,
                    'family': 'Retrofit', 'alpha': alpha, 'n_iter': ni,
                    'train_time_s': dt,
                    **metric_cols_at_ks(res, ks, primary_k=k),
                    **wilcoxon_columns('wilcoxon_vs_EASE', w),
                })

    df = pd.DataFrame(rows)
    stem = f'retrofit_ease_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")
    if df['seed'].nunique() > 1:
        key_cols = ['family', 'alpha', 'n_iter']
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
    p.add_argument('--dataset', default='ml-1m', choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--alphas', type=str, default='5.0,20.0,50.0,100.0',
                   help='Anchor weight on the original B in the Faruqui '
                        'Jacobi update. Should roughly match the sum of '
                        'neighbour similarities so anchor remains '
                        'influential. With topK_graph=20 and genre '
                        'cosine ~= 0.5 per edge, alpha in [5, 100] is '
                        'the useful range.')
    p.add_argument('--n_iters', type=str, default='3,5,10')
    p.add_argument('--topK_graph', type=int, default=20,
                   help='Top-K thresholding of the genre cosine graph '
                        'before retrofitting. Smaller K keeps the '
                        'anchor influential.')
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    alphas = [float(x) for x in args.alphas.split(',')]
    n_iters = [int(x) for x in args.n_iters.split(',')]
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, alphas=alphas, n_iters=n_iters,
        topK_graph=args.topK_graph,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
