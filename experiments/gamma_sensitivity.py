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

from experiments._shared import (
    bucket_rows,
    bucketed_metrics_at_ks,
    ensure_results_dir,
    item_popularity_buckets,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_vs_baseline,
    write_buckets_csv,
)


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
    # One bucketing of items by train popularity, reused across every
    # (lambda, rp3_beta, gamma) cell so per-bucket rows are directly
    # comparable. Five buckets = q1..q5 from most to least popular.
    bucket_of = item_popularity_buckets(train, n_buckets=5)
    # Buffer of per-bucket rows accumulated across the whole sweep and
    # flushed once at the end as a companion ``_buckets.csv``.
    bucket_rows_all = []

    rows = []
    # Per-lambda EASE baseline (no Laplacian): used as the reference model
    # for the paired Wilcoxon signed-rank test on per-user NDCG@k. rp3_beta
    # has no effect on pure EASE, so we fit it once per lambda.
    baselines = {}
    for lam in lambdas:
        print(f"\n[baseline] EASE lambda={lam} (no Laplacian)")
        t0 = time.time()
        base = HybridEASE_RP3beta()
        base.fit(train, method='score', fusion_alpha=1.0,
                 ease_lambda=lam, rp3_alpha=1.0,
                 rp3_beta=rp3_betas[0], rp3_topK=rp3_topK)
        # fusion_alpha=1.0 reduces to EASE after min-max normalisation;
        # evaluate via pure X @ B to stay apples-to-apples with the
        # Laplacian-EASE scoring.
        base.pred = base.ease.X.dot(base.ease.B)
        res_base = evaluate_at_ks(base, train, test_positive, ks=ks)
        dt_base = time.time() - t0
        print(f"  NDCG@{k}={res_base[f'NDCG@{k}']:.4f} "
              f"NDCG@{max(ks)}={res_base[f'NDCG@{max(ks)}']:.4f} "
              f"({dt_base:.1f}s)")
        baselines[lam] = res_base
        # Emit the per-bucket EASE rows so the companion CSV has an
        # anchor every consumer can diff against.
        base_buckets = bucketed_metrics_at_ks(
            base, train, test_positive, ks=ks, bucket_of=bucket_of)
        base_row_meta = {
            'dataset': dataset, 'model': 'EASE',
            'lambda': lam, 'rp3_beta': rp3_betas[0],
            'graph_source': None, 'normalise': None,
            'gamma': 0.0,
        }
        bucket_rows_all.extend(
            bucket_rows(base_row_meta, base_buckets, ks, n_buckets=5))

    for lam in lambdas:
        res_base = baselines[lam]
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
                w_result = wilcoxon_vs_baseline(res_base, res, k=k)
                w_p, w_sign = w_result[1], w_result[3]
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
                # Accuracy + diversity metrics at every k (plus the
                # ``@k`` aliases at the primary k for the plot).
                row.update(metric_cols_at_ks(res, ks, primary_k=k))
                row.update(wilcoxon_columns('wilcoxon_vs_EASE',
                                             w_result))
                rows.append(row)
                # Per-bucket NDCG/Recall/HitRate for this cell.
                lap_buckets = bucketed_metrics_at_ks(
                    model, train, test_positive, ks=ks,
                    bucket_of=bucket_of)
                lap_row_meta = {
                    'dataset': dataset, 'model': 'Laplacian-EASE',
                    'lambda': lam, 'rp3_beta': rp3_b,
                    'graph_source': graph_source,
                    'normalise': normalise, 'gamma': gamma,
                }
                bucket_rows_all.extend(
                    bucket_rows(lap_row_meta, lap_buckets, ks,
                                n_buckets=5))
                sign_str = (f'{w_sign:+d}'
                            if w_sign != 0 else ' 0')
                print(f"  gamma={gamma:<8.3f} "
                      f"NDCG@{k}={res[f'NDCG@{k}']:.4f} "
                      f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
                      f"HR@{k}={res[f'HitRate@{k}']:.4f} "
                      f"p(vs EASE)={w_p:.2e}  sign={sign_str}  "
                      f"({dt:.1f}s)")

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'gamma_sensitivity_{dataset}{suffix}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    # Companion per-bucket CSV (one row per bucket × config cell). See
    # ``experiments/_shared.bucketed_metrics_at_ks`` for semantics.
    bpath = write_buckets_csv(out_dir, stem, bucket_rows_all)
    if bpath is not None:
        print(f"Saved per-bucket CSV to {bpath}")

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
