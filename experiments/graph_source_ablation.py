"""
Graph-source ablation for Laplacian-regularized EASE.

Compares four graph sources for the Laplacian penalty:
    - rp3beta  (reference)
    - p3alpha
    - itemknn  (cosine similarity top-K)
    - binary   (binary co-occurrence)

All share identical EASE hyperparameters so the only moving part is the
graph used to build L.

Example:
    python -m experiments.graph_source_ablation --dataset ml-small --k 10
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
from models.graph_sources import VALID_SOURCES

from experiments._shared import (
    bucket_rows,
    bucketed_metrics_at_ks,
    ensure_results_dir,
    item_popularity_buckets,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_blank_columns,
    wilcoxon_columns,
    wilcoxon_vs_baseline,
    write_buckets_csv,
)


def run(dataset='ml-small', k=10,
        ease_lambda=None, gammas=None,
        topK=200, sources=None,
        rp3_beta=0.6, p3_alpha=1.0, itemknn_shrink=0.0,
        normalise='none', ks=(10, 20),
        out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    if sources is None:
        sources = list(VALID_SOURCES)
    ks = tuple(sorted(set(list(ks) + [k])))

    out_dir = ensure_results_dir('graph_source_ablation'
                                 if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)
    bucket_of = item_popularity_buckets(train, n_buckets=5)
    bucket_rows_all = []

    rows = []
    # Baseline: EASE with no Laplacian (gamma=0 via the standard path).
    print("\n[baseline] EASE (no Laplacian)")
    t0 = time.time()
    base = HybridEASE_RP3beta()
    base.fit(train, method='score', fusion_alpha=1.0,
             ease_lambda=ease_lambda, rp3_alpha=1.0,
             rp3_beta=rp3_beta, rp3_topK=topK)
    # fusion_alpha=1.0 + score method reduces to pure EASE after min-max
    # normalisation. To stay apples-to-apples, evaluate via the plain EASE
    # prediction ``X @ B``.
    res_base = evaluate_at_ks(base, train, test_positive, ks=ks)
    dt_base = time.time() - t0
    print(f"  NDCG@{k}={res_base[f'NDCG@{k}']:.4f} "
          f"NDCG@{max(ks)}={res_base[f'NDCG@{max(ks)}']:.4f} "
          f"({dt_base:.1f}s)")

    for source in sources:
        print(f"\n[source={source}]  normalise={normalise}")
        for gamma in gammas:
            t0 = time.time()
            model = HybridEASE_RP3beta()
            model.fit(train, method='laplacian',
                      ease_lambda=ease_lambda, rp3_alpha=1.0,
                      rp3_beta=rp3_beta, rp3_topK=topK,
                      graph_reg_gamma=gamma,
                      graph_source=source,
                      p3_alpha=p3_alpha,
                      itemknn_shrink=itemknn_shrink,
                      laplacian_normalise=normalise)
            res = evaluate_at_ks(model, train, test_positive, ks=ks)
            dt = time.time() - t0
            w_result = wilcoxon_vs_baseline(res_base, res, k=k)
            w_p, w_sign = w_result[1], w_result[3]
            row = {
                'dataset': dataset,
                'source': source,
                'normalise': normalise,
                'ease_lambda': ease_lambda,
                'gamma': gamma,
                'rp3_beta': rp3_beta,
                'topK': topK,
                'train_time_s': dt,
            }
            row.update(metric_cols_at_ks(res, ks, primary_k=k))
            row.update(wilcoxon_columns('wilcoxon_vs_EASE', w_result))
            rows.append(row)
            # Per-bucket row for the Laplacian cell.
            lap_buckets = bucketed_metrics_at_ks(
                model, train, test_positive, ks=ks,
                bucket_of=bucket_of)
            bucket_rows_all.extend(
                bucket_rows({'dataset': dataset,
                             'model': 'Laplacian-EASE',
                             'source': source,
                             'normalise': normalise,
                             'ease_lambda': ease_lambda,
                             'rp3_beta': rp3_beta,
                             'topK': topK,
                             'gamma': gamma},
                            lap_buckets, ks, n_buckets=5))
            # The p-value alone is ambiguous about direction, so the
            # log line reports the sign too (+1 = model wins, -1 = loses).
            sign_str = f'{w_sign:+d}' if w_sign != 0 else ' 0'
            print(f"  gamma={gamma:<6.2f} "
                  f"NDCG@{k}={res[f'NDCG@{k}']:.4f} "
                  f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
                  f"HR@{k}={res[f'HitRate@{k}']:.4f} "
                  f"p={w_p:.2e}  sign={sign_str}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'graph_source_ablation_{dataset}{suffix}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    bpath = write_buckets_csv(out_dir, stem, bucket_rows_all)
    if bpath is not None:
        print(f"Saved per-bucket CSV to {bpath}")

    # ---- Plot NDCG vs gamma per source ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for src, g in df.groupby('source'):
        g = g.sort_values('gamma')
        ax.plot(g['gamma'], g['NDCG@k'], marker='o', label=src)
    ax.axhline(res_base[f'NDCG@{k}'], ls='--', color='grey',
               label=f"EASE baseline ({res_base[f'NDCG@{k}']:.4f})")
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (Laplacian strength, log scale)')
    ax.set_ylabel(f'NDCG@{k}')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Graph-source ablation — {dataset}{title_extra}')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = out_dir / f'graph_source_ablation_{dataset}{suffix}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    # Best per source
    print("\nBest gamma per source (by NDCG@k):")
    best = (df.sort_values('NDCG@k', ascending=False)
              .groupby('source').head(1)
              .sort_values('NDCG@k', ascending=False))
    print(best[['source', 'gamma', 'NDCG@k', 'MAP@k',
                'HitRate@k']].to_string(index=False))

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--sources', type=str, default=None,
                   help='Comma-separated list of graph sources to test')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: none.')
    p.add_argument('--ks', type=str, default='10,20',
                   help='Comma-separated list of cut-offs for multi-k '
                        'evaluation (NDCG@10, NDCG@20, ...). '
                        'Default: "10,20".')
    args = p.parse_args()

    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(',')]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k, topK=args.topK,
        ease_lambda=args.ease_lambda, rp3_beta=args.rp3_beta,
        sources=sources, normalise=args.normalise, ks=ks)


if __name__ == '__main__':
    main()
