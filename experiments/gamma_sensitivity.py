"""
Gamma sensitivity sweep for Laplacian-regularized EASE.

Plots NDCG@k (and optionally MAP@k) versus gamma on a log-scaled x-axis,
for a handful of (lambda, rp3_beta) configurations. Saves both CSV and
PNG plot under ``results/gamma_sensitivity/``.

Example:
    python -m experiments.gamma_sensitivity --dataset ml-small --k 10

The gamma grid covers 5 orders of magnitude by default so the log-scale
plot is meaningful.
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
from models import HybridEASE_RP3beta

from experiments._shared import ensure_results_dir, load_dataset


def run(dataset='ml-small', k=10,
        gammas=None, lambdas=None, rp3_betas=None, rp3_topK=200,
        graph_source='rp3beta', normalise='none',
        ks=(10, 20),
        out_dir=None):
    if gammas is None:
        gammas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0,
                  100.0, 300.0, 1000.0]
    if lambdas is None:
        lambdas = [100, 500] if dataset == 'ml-1m' else [50, 200]
    if rp3_betas is None:
        rp3_betas = [0.3, 0.6]
    # Guarantee the "primary" k (used by the plot) is one of the cut-offs.
    ks = tuple(sorted(set(list(ks) + [k])))

    out_dir = ensure_results_dir('gamma_sensitivity'
                                 if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)

    rows = []
    for lam in lambdas:
        for rp3_b in rp3_betas:
            print(f"\n[gamma sweep] lambda={lam}, rp3_beta={rp3_b}, "
                  f"source={graph_source}, normalise={normalise}")
            for gamma in gammas:
                t0 = time.time()
                model = HybridEASE_RP3beta()
                model.fit(train, method='laplacian',
                          ease_lambda=lam, rp3_alpha=1.0,
                          rp3_beta=rp3_b, rp3_topK=rp3_topK,
                          graph_reg_gamma=gamma,
                          graph_source=graph_source,
                          laplacian_normalise=normalise)
                res = evaluate_at_ks(model, train, test_positive, ks=ks)
                dt = time.time() - t0
                row = {
                    'dataset': dataset,
                    'lambda': lam,
                    'rp3_beta': rp3_b,
                    'rp3_topK': rp3_topK,
                    'graph_source': graph_source,
                    'normalise': normalise,
                    'gamma': gamma,
                    'train_time_s': dt,
                }
                for kk in ks:
                    for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                        row[f'{m}@{kk}'] = res[f'{m}@{kk}']
                # Back-compat aliases at the primary k used for the plot.
                row['NDCG@k']    = res[f'NDCG@{k}']
                row['MAP@k']     = res[f'MAP@{k}']
                row['HitRate@k'] = res[f'HitRate@{k}']
                row['Recall@k']  = res[f'Recall@{k}']
                rows.append(row)
                print(f"  gamma={gamma:<8.3f} "
                      f"NDCG@{k}={res[f'NDCG@{k}']:.4f} "
                      f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
                      f"HR@{k}={res[f'HitRate@{k}']:.4f}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    csv_path = out_dir / f'gamma_sensitivity_{dataset}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot NDCG vs gamma (log x) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (lam, rp3_b), g in df.groupby(['lambda', 'rp3_beta']):
        g = g.sort_values('gamma')
        ax.plot(g['gamma'], g['NDCG@k'], marker='o',
                label=f'λ={lam}, β_rp3={rp3_b}')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (Laplacian strength, log scale)')
    ax.set_ylabel(f'NDCG@{k}')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Gamma sensitivity — {dataset}{title_extra}')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = out_dir / f'gamma_sensitivity_{dataset}{suffix}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma values (optional)')
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: none.')
    p.add_argument('--ks', type=str, default='10,20',
                   help='Comma-separated list of cut-offs for multi-k '
                        'evaluation (NDCG@10, NDCG@20, ...). '
                        'Default: "10,20".')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k, gammas=gammas,
        graph_source=args.graph_source, normalise=args.normalise,
        ks=ks)


if __name__ == '__main__':
    main()
