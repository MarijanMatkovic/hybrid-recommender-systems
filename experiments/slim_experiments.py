"""
SLIM baseline + Laplacian-regularized SLIM.

Runs:
    1) Vanilla SLIM (ElasticNet, non-negative, per-item CD).
    2) Laplacian-SLIM for a grid of gamma values (ISTA solver).

Compares both against EASE as reference. Writes a CSV + NDCG-vs-gamma
plot under ``results/slim/``.

SLIM fits column-by-column and is much slower than EASE. On ``ml-small``
it takes ~20-40 s; on ``ml-1m`` it is not practical without further
optimisation -- for thesis purposes, use ml-small for SLIM experiments
and demonstrate that the Laplacian generalisation applies to SLIM too.

Example:
    python -m experiments.slim_experiments --dataset ml-small --k 10
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
from models import SLIM, build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
    wilcoxon_blank_columns,
    wilcoxon_columns,
    wilcoxon_vs_baseline,
)


class _StandaloneWrapper:
    """Mimic the interface ``evaluate`` expects (``.ease`` + ``.pred``)."""
    def __init__(self, slim_model):
        self.ease = slim_model      # SLIM exposes the same LabelEncoders
        self.pred = slim_model.pred


def _scale_laplacian(L, G_diag_mean):
    """Scale the Laplacian so that mean(diag(L)) ≈ G_diag_mean."""
    l_diag_mean = np.mean(np.diag(L))
    if l_diag_mean > 0:
        return L * (G_diag_mean / l_diag_mean)
    return L


def run(dataset='ml-small', k=10,
        l1_reg=1e-4, beta=1e-3, positive=True,
        gammas=None, graph_source='rp3beta',
        rp3_beta=0.6, topK=200,
        n_iter=200,
        normalise='none', ks=(10, 20),
        out_dir=None):
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0, 30.0]
    ks = tuple(sorted(set(list(ks) + [k])))

    out_dir = ensure_results_dir('slim' if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)

    rows = []

    # ---- 1) Vanilla SLIM ----
    print("\n[SLIM] vanilla (ElasticNet)")
    t0 = time.time()
    slim = SLIM()
    slim.fit(train, l1_reg=l1_reg, beta=beta, positive=positive,
             max_iter=30, tol=1e-4)
    t_slim = time.time() - t0
    res_slim = evaluate_at_ks(_StandaloneWrapper(slim), train,
                              test_positive, ks=ks)
    print(f"  SLIM NDCG@{k}={res_slim[f'NDCG@{k}']:.4f} "
          f"NDCG@{max(ks)}={res_slim[f'NDCG@{max(ks)}']:.4f} "
          f"({t_slim:.1f}s)")
    _row = {
        'dataset': dataset, 'model': 'SLIM',
        'gamma': 0.0, 'graph_source': None, 'normalise': None,
        'train_time_s': t_slim,
    }
    for kk in ks:
        for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
            _row[f'{m}@{kk}'] = res_slim[f'{m}@{kk}']
    _row['NDCG@k']    = res_slim[f'NDCG@{k}']
    _row['MAP@k']     = res_slim[f'MAP@{k}']
    _row['HitRate@k'] = res_slim[f'HitRate@{k}']
    _row['Recall@k']  = res_slim[f'Recall@{k}']
    # SLIM is the main baseline in this script.
    _row.update(wilcoxon_blank_columns('wilcoxon_vs_SLIM'))
    _row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
    rows.append(_row)

    # ---- 2) EASE reference (same data) ----
    print("\n[reference] EASE")
    t0 = time.time()
    ease_ref = HybridEASE_RP3beta()
    ease_ref.fit(train, method='score', fusion_alpha=1.0,
                 ease_lambda=500 if dataset == 'ml-1m' else 200,
                 rp3_alpha=1.0, rp3_beta=rp3_beta, rp3_topK=topK)
    ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
    res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=ks)
    t_ease = time.time() - t0
    print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f} "
          f"NDCG@{max(ks)}={res_ease[f'NDCG@{max(ks)}']:.4f} "
          f"({t_ease:.1f}s)")
    _row = {
        'dataset': dataset, 'model': 'EASE',
        'gamma': 0.0, 'graph_source': None, 'normalise': None,
        'train_time_s': t_ease,
    }
    for kk in ks:
        for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
            _row[f'{m}@{kk}'] = res_ease[f'{m}@{kk}']
    _row['NDCG@k']    = res_ease[f'NDCG@{k}']
    _row['MAP@k']     = res_ease[f'MAP@{k}']
    _row['HitRate@k'] = res_ease[f'HitRate@{k}']
    _row['Recall@k']  = res_ease[f'Recall@{k}']
    # EASE vs SLIM: paired test over common users.
    w_ease_vs_slim = wilcoxon_vs_baseline(res_slim, res_ease, k=k)
    _row.update(wilcoxon_columns('wilcoxon_vs_SLIM', w_ease_vs_slim))
    _row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
    rows.append(_row)

    # ---- 3) Build Laplacian once ----
    X = ease_ref.ease.X
    W = build_graph(X, source=graph_source, topK=topK,
                    rp3_beta=rp3_beta, implicit=True)
    L_dense, _ = build_laplacian(W, normalise=normalise)
    G_diag_mean = float(np.mean(np.array(
        X.multiply(X).sum(axis=0)).flatten()))  # ≈ mean(diag(X^T X))
    L_scaled = _scale_laplacian(L_dense, G_diag_mean)

    # ---- 4) Laplacian SLIM sweep ----
    for gamma in gammas:
        print(f"\n[SLIM-Laplacian] gamma={gamma}  normalise={normalise}")
        t0 = time.time()
        slim_lap = SLIM()
        slim_lap.fit_laplacian(
            train, L_scaled=L_scaled, beta=beta, l1_reg=l1_reg,
            gamma=gamma, positive=positive, n_iter=n_iter,
        )
        t = time.time() - t0
        res = evaluate_at_ks(_StandaloneWrapper(slim_lap), train,
                             test_positive, ks=ks)
        w_slim = wilcoxon_vs_baseline(res_slim, res, k=k)
        w_ease = wilcoxon_vs_baseline(res_ease, res, k=k)
        w_p_slim,  w_sign_slim  = w_slim[1], w_slim[3]
        w_p_ease,  w_sign_ease  = w_ease[1], w_ease[3]
        sign_slim_str = (f'{w_sign_slim:+d}'
                         if w_sign_slim != 0 else ' 0')
        sign_ease_str = (f'{w_sign_ease:+d}'
                         if w_sign_ease != 0 else ' 0')
        print(f"  NDCG@{k}={res[f'NDCG@{k}']:.4f} "
              f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
              f"p(vs SLIM)={w_p_slim:.2e}[{sign_slim_str}] "
              f"p(vs EASE)={w_p_ease:.2e}[{sign_ease_str}] "
              f"({t:.1f}s)")
        _row = {
            'dataset': dataset, 'model': 'SLIM-Laplacian',
            'gamma': gamma, 'graph_source': graph_source,
            'normalise': normalise, 'train_time_s': t,
        }
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                _row[f'{m}@{kk}'] = res[f'{m}@{kk}']
        _row['NDCG@k']    = res[f'NDCG@{k}']
        _row['MAP@k']     = res[f'MAP@{k}']
        _row['HitRate@k'] = res[f'HitRate@{k}']
        _row['Recall@k']  = res[f'Recall@{k}']
        _row.update(wilcoxon_columns('wilcoxon_vs_SLIM', w_slim))
        _row.update(wilcoxon_columns('wilcoxon_vs_EASE', w_ease))
        rows.append(_row)

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    csv_path = out_dir / f'slim_{dataset}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    slim_lap_rows = df[df['model'] == 'SLIM-Laplacian'].sort_values('gamma')
    ax.plot(slim_lap_rows['gamma'], slim_lap_rows['NDCG@k'],
            marker='o', label='SLIM + Laplacian')
    ax.axhline(res_slim[f'NDCG@{k}'], ls='--', color='tab:orange',
               label=f"SLIM baseline ({res_slim[f'NDCG@{k}']:.4f})")
    ax.axhline(res_ease[f'NDCG@{k}'], ls=':', color='grey',
               label=f"EASE reference ({res_ease[f'NDCG@{k}']:.4f})")
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (Laplacian strength, log scale)')
    ax.set_ylabel(f'NDCG@{k}')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'SLIM + Laplacian — {dataset}{title_extra}')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = out_dir / f'slim_{dataset}{suffix}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--l1_reg', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1e-3)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma values (optional)')
    p.add_argument('--n_iter', type=int, default=200)
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

    run(dataset=args.dataset, k=args.k,
        l1_reg=args.l1_reg, beta=args.beta,
        gammas=gammas, graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        n_iter=args.n_iter, normalise=args.normalise, ks=ks)


if __name__ == '__main__':
    main()
