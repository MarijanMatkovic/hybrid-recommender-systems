"""
Signed-Laplacian EASE sweep (roadmap item 15).

Encodes 'liked' vs 'disliked' co-occurrence as signed edges via
Kunegis's signed-Laplacian (PSD for any real-valued W), then runs
EASE with M = L_sig as the regulariser. MovieLens ratings 4-5 give
the positive signal; ratings 1-3 give the negative signal.

Compares against vanilla EASE and against standard Lap-EASE (positive-
only graph) at the same gamma.

No published "signed-Laplacian EASE" exists -- this is a novel modality
for the thesis.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd

from data import load_movielens, load_movielens_1m
from evaluation.metrics import evaluate_at_ks
from models.signed_laplacian import (
    SignedLaplacianEASE, build_signed_item_graph)
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


def _load_full_ratings(dataset):
    """Return the full ratings DataFrame (un-split) so we can build a
    signed graph from the user-rating signal."""
    if dataset == 'ml-1m':
        ratings, _ = load_movielens_1m('data/ml-1m', min_interactions=5)
    elif dataset == 'ml-small':
        ratings, _ = load_movielens('data/ml-latest-small',
                                    min_interactions=5)
    else:
        raise ValueError(
            f"Signed-Laplacian needs explicit rating signals; "
            f"dataset={dataset!r} not supported.")
    return ratings


def run(dataset='ml-1m', k=10, ks=(10, 20), ease_lambda=None,
        gammas=(1, 3, 10, 30), pos_thr=4.0, neg_thr=3.0, topK_graph=200,
        split_mode='random', seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'signed_laplacian' if out_dir is None else out_dir)

    ratings_full = _load_full_ratings(dataset)
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
            'gamma': 0.0, 'train_time_s': t_ease,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Build signed item graph from training-portion ratings only.
        # (Don't leak test signal.)
        train_ids = set(train['user_id'].unique())
        train_ratings = ratings_full[
            ratings_full['user_id'].isin(train_ids)]
        item_enc = ease_ref.ease.item_enc
        t_graph0 = time.time()
        W_signed = build_signed_item_graph(
            train_ratings, item_enc,
            pos_thr=pos_thr, neg_thr=neg_thr, topK=topK_graph)
        t_graph = time.time() - t_graph0
        pos_count = int(np.sum(W_signed > 0))
        neg_count = int(np.sum(W_signed < 0))
        print(f"  Signed graph built in {t_graph:.1f}s: "
              f"+edges={pos_count}, -edges={neg_count}, "
              f"min={W_signed.min():.1f}, max={W_signed.max():.1f}")

        print("\n  [15] Signed-Laplacian EASE (sweep gamma)")
        for g in gammas:
            t1 = time.time()
            mdl = SignedLaplacianEASE()
            mdl.fit(train, lambdas=ease_lambda, gamma=g, W_signed=W_signed)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    gamma={g:<6g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed,
                'family': 'SignedLap', 'gamma': g, 'train_time_s': dt,
                'pos_edges': pos_count, 'neg_edges': neg_count,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', w),
            })

    df = pd.DataFrame(rows)
    stem = f'signed_laplacian_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")
    if df['seed'].nunique() > 1:
        key_cols = ['family', 'gamma']
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
    p.add_argument('--gammas', type=str, default='1,3,10,30')
    p.add_argument('--pos_thr', type=float, default=4.0)
    p.add_argument('--neg_thr', type=float, default=3.0)
    p.add_argument('--topK_graph', type=int, default=200)
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    gammas = [float(x) for x in args.gammas.split(',')]
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, gammas=gammas,
        pos_thr=args.pos_thr, neg_thr=args.neg_thr,
        topK_graph=args.topK_graph,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
