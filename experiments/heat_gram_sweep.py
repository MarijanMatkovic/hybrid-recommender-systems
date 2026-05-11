"""
Heat / PPR pre-smoothed Gram EASE sweep (roadmap item 14).

Multiplicative shaping G_tilde = h(L_tilde)^T G h(L_tilde) with
h in {expm(-t L), (1-alpha)(I - alpha A)^-1}. Composable with
additive Lap-EASE rather than redundant, so this is an orthogonal
axis of improvement.
"""

from __future__ import annotations
import argparse
import time

import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models.heat_gram import HeatGramEASE
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


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None, heat_ts=(0.5, 1.0, 2.0),
        ppr_alphas=(0.3, 0.5, 0.7),
        top_k_eig=256,
        split_mode='random', seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS) if split_mode == 'random' else [0]
    out_path = ensure_results_dir(
        'heat_gram' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        if split_mode == 'random':
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
        else:
            train, test_pos, _ = load_dataset(
                dataset, split_mode='temporal')

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
            'kind': '-', 'param': 0.0, 'train_time_s': t_ease,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Heat
        print("\n  [14] Heat-Gram EASE")
        for t in heat_ts:
            t1 = time.time()
            mdl = HeatGramEASE()
            mdl.fit(train, lambdas=ease_lambda, filter_kind='heat',
                    t_heat=t, top_k_eig=top_k_eig)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    heat t={t:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed,
                'family': 'HeatGram', 'kind': 'heat', 'param': t,
                'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', w),
            })

        # PPR
        print("\n  [14] PPR-Gram EASE")
        for a in ppr_alphas:
            t1 = time.time()
            mdl = HeatGramEASE()
            mdl.fit(train, lambdas=ease_lambda, filter_kind='ppr',
                    ppr_alpha=a)
            res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
            dt = time.time() - t1
            w = wilcoxon_vs_baseline(res_ease, res, k=k)
            print(f"    ppr alpha={a:<5g}  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                  f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed,
                'family': 'HeatGram', 'kind': 'ppr', 'param': a,
                'train_time_s': dt,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', w),
            })

    df = pd.DataFrame(rows)
    stem = f'heat_gram_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")
    if df['seed'].nunique() > 1:
        key_cols = ['family', 'kind', 'param']
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
    p.add_argument('--heat_ts', type=str, default='0.5,1.0,2.0')
    p.add_argument('--ppr_alphas', type=str, default='0.3,0.5,0.7')
    p.add_argument('--top_k_eig', type=int, default=256)
    p.add_argument('--split_mode', default='random',
                   choices=['random', 'temporal'])
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    heat_ts = [float(x) for x in args.heat_ts.split(',')]
    ppr_alphas = [float(x) for x in args.ppr_alphas.split(',')]
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, heat_ts=heat_ts,
        ppr_alphas=ppr_alphas, top_k_eig=args.top_k_eig,
        split_mode=args.split_mode, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
