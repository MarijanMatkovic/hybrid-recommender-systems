"""
Re-ranking ablation: DPP-MAP, PPR diffusion, and GF-CF -> EASE cascade
(roadmap items 17, 19, 20).

All three operate on top of a fitted EASE: they reshape the per-user
score vector before top-k selection. The cascade additionally requires
a fitted GF-CF for candidate retrieval. Each is reported per seed with
Wilcoxon vs the EASE-only baseline.

Memory note: PPR re-ranking on the full pred matrix is (n_users x
n_items) dense -- ml-1m is fine, Netflix needs per-user diffusion.
DPP-MAP and cascade operate per-user from the start so both scale.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps

from evaluation.metrics import evaluate_at_ks
from models import EASE, build_graph
from models.lazy_pred import make_pred
from models.gf_cf import GFCF
from models.poly_filter import normalised_item_adjacency
from fusion import (
    dpp_rerank_pred, ppr_rerank_full, topn_candidates, cascade_rerank,
    materialise_pred,
)

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    ensure_results_dir,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_blank_columns,
    wilcoxon_paired,
)


class _Wrap:
    def __init__(self, ease, pred):
        self.ease = ease; self.pred = pred


def run(dataset='ml-1m', k=10, ks=(10, 20), ease_lambda=None,
        ppr_alphas=(0.3, 0.5, 0.7), ppr_iters=10,
        dpp_top_pools=(100, 200), dpp_graph_source='rp3beta',
        cascade_top_ns=(200, 500), cascade_gfcf_k=256, cascade_gfcf_alpha=0.3,
        seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)
    out_path = ensure_results_dir(
        'rerank_ablation' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        train, test_pos, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)

        # Base EASE
        t0 = time.time()
        ease = EASE(); ease.fit(train, lambda_=ease_lambda)
        res_e = evaluate_at_ks(_Wrap(ease, ease.pred), train, test_pos, ks=ks)
        t_e = time.time() - t0
        print(f"  EASE NDCG@{k}={res_e[f'NDCG@{k}']:.4f}  ({t_e:.1f}s)")
        rows.append({
            'dataset': dataset, 'seed': seed, 'method': 'EASE',
            'config': '-', 'train_time_s': t_e,
            **metric_cols_at_ks(res_e, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Item-item graph A_tilde (for PPR + DPP-MAP)
        S_ease = materialise_pred(ease.pred)
        W = build_graph(ease.X, source=dpp_graph_source, topK=200,
                        rp3_beta=0.6, implicit=True)
        A_tilde = normalised_item_adjacency(ease.X)
        if sps.issparse(A_tilde):
            A_tilde_dense = A_tilde.toarray()
        else:
            A_tilde_dense = np.asarray(A_tilde)

        # ---- 19) PPR diffusion ---------------------------------------
        print("\n  [19] PPR score-diffusion re-ranker")
        for a in ppr_alphas:
            t1 = time.time()
            S_ppr = ppr_rerank_full(S_ease, A_tilde_dense,
                                     alpha=a, n_iter=ppr_iters)
            res = evaluate_at_ks(_Wrap(ease, S_ppr), train, test_pos, ks=ks)
            dt = time.time() - t1
            wstat = wilcoxon_paired(
                (res_e['per_user_ndcg'][k], res_e['per_user_ids']),
                (res['per_user_ndcg'][k], res['per_user_ids']))
            print(f"    PPR alpha={a:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={wstat[1]:.2e}[{wstat[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'method': 'PPR',
                'config': f'alpha={a}', 'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', wstat),
            })

        # ---- 17) DPP-MAP greedy re-ranker ----------------------------
        print(f"\n  [17] DPP-MAP greedy re-rank ({dpp_graph_source})")
        for pool in dpp_top_pools:
            t1 = time.time()
            S_dpp = dpp_rerank_pred(S_ease, W, k=max(ks), top_pool=pool)
            res = evaluate_at_ks(_Wrap(ease, S_dpp), train, test_pos, ks=ks)
            dt = time.time() - t1
            wstat = wilcoxon_paired(
                (res_e['per_user_ndcg'][k], res_e['per_user_ids']),
                (res['per_user_ndcg'][k], res['per_user_ids']))
            print(f"    DPP top_pool={pool:<4d}  NDCG@{k}={res[f'NDCG@{k}']:.4f}"
                  f"  p={wstat[1]:.2e}[{wstat[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'method': 'DPP-MAP',
                'config': f'top_pool={pool}', 'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', wstat),
            })

        # ---- 20) Cascade GF-CF -> EASE -------------------------------
        print("\n  [20] Cascade GF-CF -> EASE")
        t1 = time.time()
        gfcf = GFCF()
        gfcf.fit(train, k=cascade_gfcf_k, alpha=cascade_gfcf_alpha)
        S_gfcf = materialise_pred(gfcf.pred)
        t_gf = time.time() - t1
        print(f"    GF-CF fit: {t_gf:.1f}s, NDCG@{k} on GF-CF alone: ")
        res_gf = evaluate_at_ks(_Wrap(gfcf, gfcf.pred), train, test_pos, ks=ks)
        print(f"      {res_gf[f'NDCG@{k}']:.4f}")
        rows.append({
            'dataset': dataset, 'seed': seed, 'method': 'GF-CF',
            'config': f'k={cascade_gfcf_k},alpha={cascade_gfcf_alpha}',
            'train_time_s': t_gf,
            **metric_cols_at_ks(res_gf, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })
        for top_n in cascade_top_ns:
            t1 = time.time()
            cand = topn_candidates(S_gfcf, top_n=top_n)
            S_casc = cascade_rerank(S_ease, cand)
            res = evaluate_at_ks(_Wrap(ease, S_casc), train, test_pos, ks=ks)
            dt = time.time() - t1
            wstat = wilcoxon_paired(
                (res_e['per_user_ndcg'][k], res_e['per_user_ids']),
                (res['per_user_ndcg'][k], res['per_user_ids']))
            print(f"    Cascade top_n={top_n:<4d}  NDCG@{k}={res[f'NDCG@{k}']:.4f}"
                  f"  p={wstat[1]:.2e}[{wstat[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'method': 'Cascade',
                'config': f'top_n={top_n}', 'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', wstat),
            })

    df = pd.DataFrame(rows)
    stem = f'rerank_ablation_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['method', 'config']
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
                   help='netflix-prize not supported (full pred too big)')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--ppr_alphas', type=str, default='0.3,0.5,0.7')
    p.add_argument('--ppr_iters', type=int, default=10)
    p.add_argument('--dpp_top_pools', type=str, default='100,200')
    p.add_argument('--dpp_graph_source', default='rp3beta')
    p.add_argument('--cascade_top_ns', type=str, default='200,500')
    p.add_argument('--cascade_gfcf_k', type=int, default=256)
    p.add_argument('--cascade_gfcf_alpha', type=float, default=0.3)
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    ppr_alphas = [float(x) for x in args.ppr_alphas.split(',')]
    dpp_pools = [int(x) for x in args.dpp_top_pools.split(',')]
    cascade_top_ns = [int(x) for x in args.cascade_top_ns.split(',')]

    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda,
        ppr_alphas=ppr_alphas, ppr_iters=args.ppr_iters,
        dpp_top_pools=dpp_pools, dpp_graph_source=args.dpp_graph_source,
        cascade_top_ns=cascade_top_ns,
        cascade_gfcf_k=args.cascade_gfcf_k,
        cascade_gfcf_alpha=args.cascade_gfcf_alpha,
        seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
