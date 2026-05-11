"""
CAR / SAR Bayesian-prior EASE sweep (roadmap item 18).

Closed-form solver with rho interpolating between ridge (rho=0) and
Lap-EASE (rho near 1). For now we grid-search (mu, rho); the type-II
marginal-likelihood (empirical Bayes) hook is a future extension.

Both CAR (Prec = D - rho W) and SAR (Prec = (I-rho W)^T (I-rho W))
priors are run on the same item-item graph used by Lap-EASE.
"""

from __future__ import annotations
import argparse
import time

import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models import build_graph
from models.car_sar_ease import CARSAR_EASE
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


def run(dataset='ml-1m', k=10, ks=(10, 20), ease_lambda=None,
        mus=(1, 3, 10, 30), rhos=(0.1, 0.5, 0.9),
        priors=('car', 'sar'),
        graph_source='rp3beta', topK=200, rp3_beta=0.6,
        seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)
    out_path = ensure_results_dir(
        'car_sar' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        train, test_pos, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)

        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=ease_lambda, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=topK)
        res_ease = evaluate_at_ks(ease_ref, train, test_pos, ks=ks)
        t_e = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_e:.1f}s)")
        rows.append({
            'dataset': dataset, 'seed': seed, 'family': 'EASE',
            'prior': '-', 'mu': 0.0, 'rho': 0.0, 'train_time_s': t_e,
            **metric_cols_at_ks(res_ease, ks, primary_k=k),
            **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
        })

        # Shared graph
        X = ease_ref.ease.X
        W = build_graph(X, source=graph_source, topK=topK,
                        rp3_beta=rp3_beta, implicit=True)

        for prior in priors:
            print(f"\n  [18] CAR/SAR Bayesian EASE -- prior={prior}")
            for rho in rhos:
                for mu in mus:
                    t1 = time.time()
                    mdl = CARSAR_EASE()
                    mdl.fit(train, W, lambdas=ease_lambda, mu=mu,
                            rho=rho, prior=prior)
                    res = evaluate_at_ks(mdl, train, test_pos, ks=ks)
                    dt = time.time() - t1
                    w = wilcoxon_vs_baseline(res_ease, res, k=k)
                    print(f"    {prior} rho={rho:<5g} mu={mu:<5g}  "
                          f"NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
                          f"p={w[1]:.2e}[{w[3]:+d}]  ({dt:.1f}s)")
                    rows.append({
                        'dataset': dataset, 'seed': seed,
                        'family': 'CAR-SAR-EASE', 'prior': prior,
                        'mu': mu, 'rho': rho, 'train_time_s': dt,
                        **metric_cols_at_ks(res, ks, primary_k=k),
                        **wilcoxon_columns('wilcoxon_vs_EASE', w),
                    })

    df = pd.DataFrame(rows)
    stem = f'car_sar_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['family', 'prior', 'mu', 'rho']
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
    p.add_argument('--mus', type=str, default='1,3,10,30')
    p.add_argument('--rhos', type=str, default='0.1,0.5,0.9')
    p.add_argument('--priors', type=str, default='car,sar')
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    mus = [float(x) for x in args.mus.split(',')]
    rhos = [float(x) for x in args.rhos.split(',')]
    priors = [x for x in args.priors.split(',') if x.strip()]
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, mus=mus, rhos=rhos, priors=priors,
        graph_source=args.graph_source, topK=args.topK,
        rp3_beta=args.rp3_beta, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
