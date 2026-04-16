"""
EDLAE baseline + Laplacian-regularized EDLAE.

Runs:
    1) Vanilla EDLAE (edge-dropout denoising closed form).
    2) Laplacian-EDLAE for a grid of gamma values (also closed form --
       same system with an extra gamma*L on the Gram matrix).

Compares both against EASE. EDLAE is a *stronger* linear baseline than
EASE, so this is the key experiment showing Laplacian regularization
continues to pay off even on top of a better denoising prior.

Example:
    python -m experiments.edlae_experiments --dataset ml-small --k 10
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

from evaluation.metrics import evaluate
from models import EDLAE, build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import ensure_results_dir, load_dataset


class _StandaloneWrapper:
    """Mimic the interface ``evaluate`` expects (``.ease`` + ``.pred``)."""
    def __init__(self, model):
        self.ease = model           # EDLAE exposes the same LabelEncoders
        self.pred = model.pred


def _scale_laplacian(L, G_diag_mean):
    l_diag_mean = np.mean(np.diag(L))
    if l_diag_mean > 0:
        return L * (G_diag_mean / l_diag_mean)
    return L


def run(dataset='ml-small', k=10,
        lambda_=None, dropout=0.5,
        gammas=None, graph_source='rp3beta',
        rp3_beta=0.6, topK=200,
        dropouts=None,
        normalise='none',
        out_dir=None):
    if lambda_ is None:
        lambda_ = 500 if dataset == 'ml-1m' else 200
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    if dropouts is None:
        dropouts = [0.25, 0.5, 0.75]

    out_dir = ensure_results_dir('edlae' if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)

    rows = []

    # ---- 1) EASE reference ----
    print("\n[reference] EASE")
    t0 = time.time()
    ease_ref = HybridEASE_RP3beta()
    ease_ref.fit(train, method='score', fusion_alpha=1.0,
                 ease_lambda=lambda_, rp3_alpha=1.0,
                 rp3_beta=rp3_beta, rp3_topK=topK)
    ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
    res_ease = evaluate(ease_ref, train, test_positive, k=k)
    t_ease = time.time() - t0
    print(f"  EASE NDCG={res_ease['NDCG@k']:.4f}  ({t_ease:.1f}s)")
    rows.append({
        'dataset': dataset,
        'model': 'EASE',
        'dropout': 0.0,
        'gamma': 0.0,
        'graph_source': None,
        'NDCG@k': res_ease['NDCG@k'],
        'MAP@k': res_ease['MAP@k'],
        'HitRate@k': res_ease['HitRate@k'],
        'Recall@k': res_ease['Recall@k'],
        'train_time_s': t_ease,
    })

    # ---- 2) Vanilla EDLAE sweep over dropout ----
    best_edlae_res = None
    best_edlae_dropout = None
    for p in dropouts:
        print(f"\n[EDLAE] dropout={p}")
        t0 = time.time()
        edlae = EDLAE()
        edlae.fit(train, lambda_=lambda_, dropout=p)
        t = time.time() - t0
        res = evaluate(_StandaloneWrapper(edlae), train, test_positive, k=k)
        print(f"  NDCG={res['NDCG@k']:.4f}  ({t:.1f}s)")
        rows.append({
            'dataset': dataset,
            'model': 'EDLAE',
            'dropout': p,
            'gamma': 0.0,
            'graph_source': None,
            'NDCG@k': res['NDCG@k'],
            'MAP@k': res['MAP@k'],
            'HitRate@k': res['HitRate@k'],
            'Recall@k': res['Recall@k'],
            'train_time_s': t,
        })
        if best_edlae_res is None or res['NDCG@k'] > best_edlae_res['NDCG@k']:
            best_edlae_res = res
            best_edlae_dropout = p

    print(f"\nBest EDLAE dropout: {best_edlae_dropout} "
          f"(NDCG={best_edlae_res['NDCG@k']:.4f})")

    # ---- 3) Build Laplacian once ----
    X = ease_ref.ease.X
    W = build_graph(X, source=graph_source, topK=topK,
                    rp3_beta=rp3_beta, implicit=True)
    L_dense, _ = build_laplacian(W, normalise=normalise)
    G_diag_mean = float(np.mean(np.array(
        X.multiply(X).sum(axis=0)).flatten()))
    L_scaled = _scale_laplacian(L_dense, G_diag_mean)

    # ---- 4) Laplacian-EDLAE sweep at the best dropout ----
    for gamma in gammas:
        print(f"\n[EDLAE-Laplacian] dropout={best_edlae_dropout}, "
              f"gamma={gamma}, normalise={normalise}")
        t0 = time.time()
        edlae_lap = EDLAE()
        edlae_lap.fit_laplacian(
            train, L_scaled=L_scaled,
            lambda_=lambda_, dropout=best_edlae_dropout,
            gamma=gamma,
        )
        t = time.time() - t0
        res = evaluate(_StandaloneWrapper(edlae_lap), train, test_positive,
                       k=k)
        print(f"  NDCG={res['NDCG@k']:.4f}  ({t:.1f}s)")
        rows.append({
            'dataset': dataset,
            'model': 'EDLAE-Laplacian',
            'dropout': best_edlae_dropout,
            'gamma': gamma,
            'graph_source': graph_source,
            'normalise': normalise,
            'NDCG@k': res['NDCG@k'],
            'MAP@k': res['MAP@k'],
            'HitRate@k': res['HitRate@k'],
            'Recall@k': res['Recall@k'],
            'train_time_s': t,
        })

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    csv_path = out_dir / f'edlae_{dataset}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lap_rows = df[df['model'] == 'EDLAE-Laplacian'].sort_values('gamma')
    ax.plot(lap_rows['gamma'], lap_rows['NDCG@k'],
            marker='o', label=f'EDLAE + Laplacian (dropout={best_edlae_dropout})')
    ax.axhline(best_edlae_res['NDCG@k'], ls='--', color='tab:orange',
               label=f"EDLAE baseline ({best_edlae_res['NDCG@k']:.4f})")
    ax.axhline(res_ease['NDCG@k'], ls=':', color='grey',
               label=f"EASE reference ({res_ease['NDCG@k']:.4f})")
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (Laplacian strength, log scale)')
    ax.set_ylabel(f'NDCG@{k}')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'EDLAE + Laplacian — {dataset}{title_extra}')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = out_dir / f'edlae_{dataset}{suffix}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--lambda_', dest='lambda_', type=float, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma values (optional)')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: none.')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]

    dropouts = None
    if args.dropout is not None:
        dropouts = [args.dropout]

    run(dataset=args.dataset, k=args.k,
        lambda_=args.lambda_, gammas=gammas,
        graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        dropouts=dropouts, normalise=args.normalise)


if __name__ == '__main__':
    main()
