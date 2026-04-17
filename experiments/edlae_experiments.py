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

from evaluation.metrics import evaluate_at_ks
from models import EDLAE, build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
    wilcoxon_vs_baseline,
)


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
        normalise='none', ks=(10, 20),
        n_seeds=1, split_seeds=None,
        out_dir=None):
    """Single-split EDLAE + Laplacian-EDLAE sweep.

    For multi-seed confidence intervals use
    ``experiments.edlae_multiseed`` instead. The ``n_seeds`` / ``split_seeds``
    kwargs are kept for API completeness but ignored here (this function
    always uses the deterministic temporal split).
    """
    del n_seeds, split_seeds  # handled by edlae_multiseed, not here
    if lambda_ is None:
        lambda_ = 500 if dataset == 'ml-1m' else 200
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    if dropouts is None:
        dropouts = [0.25, 0.5, 0.75]
    ks = tuple(sorted(set(list(ks) + [k])))

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
    res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=ks)
    t_ease = time.time() - t0
    print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f} "
          f"NDCG@{max(ks)}={res_ease[f'NDCG@{max(ks)}']:.4f} "
          f"({t_ease:.1f}s)")
    _row = {
        'dataset': dataset, 'model': 'EASE',
        'dropout': 0.0, 'gamma': 0.0,
        'graph_source': None, 'normalise': None,
        'train_time_s': t_ease,
    }
    for kk in ks:
        for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
            _row[f'{m}@{kk}'] = res_ease[f'{m}@{kk}']
    _row['NDCG@k']    = res_ease[f'NDCG@{k}']
    _row['MAP@k']     = res_ease[f'MAP@{k}']
    _row['HitRate@k'] = res_ease[f'HitRate@{k}']
    _row['Recall@k']  = res_ease[f'Recall@{k}']
    # EASE is the baseline in this script -- its own row carries a
    # nan/0 Wilcoxon triple so the column schema is identical for
    # every row.
    _row['wilcoxon_stat_vs_EASE']    = float('nan')
    _row['wilcoxon_p_vs_EASE']       = float('nan')
    _row['wilcoxon_n_pairs_vs_EASE'] = 0
    rows.append(_row)

    # ---- 2) Vanilla EDLAE sweep over dropout ----
    best_edlae_res = None
    best_edlae_dropout = None
    for p in dropouts:
        print(f"\n[EDLAE] dropout={p}")
        t0 = time.time()
        edlae = EDLAE()
        edlae.fit(train, lambda_=lambda_, dropout=p)
        t = time.time() - t0
        res = evaluate_at_ks(_StandaloneWrapper(edlae), train,
                             test_positive, ks=ks)
        w_stat, w_p, w_n = wilcoxon_vs_baseline(res_ease, res, k=k)
        print(f"  NDCG@{k}={res[f'NDCG@{k}']:.4f} "
              f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
              f"p(vs EASE)={w_p:.2e}  ({t:.1f}s)")
        _row = {
            'dataset': dataset,
            'model': 'EDLAE',
            'dropout': p,
            'gamma': 0.0,
            'graph_source': None,
            'normalise': None,
            'train_time_s': t,
        }
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                _row[f'{m}@{kk}'] = res[f'{m}@{kk}']
        _row['NDCG@k']    = res[f'NDCG@{k}']
        _row['MAP@k']     = res[f'MAP@{k}']
        _row['HitRate@k'] = res[f'HitRate@{k}']
        _row['Recall@k']  = res[f'Recall@{k}']
        _row['wilcoxon_stat_vs_EASE']    = w_stat
        _row['wilcoxon_p_vs_EASE']       = w_p
        _row['wilcoxon_n_pairs_vs_EASE'] = w_n
        rows.append(_row)
        if (best_edlae_res is None
                or res[f'NDCG@{k}'] > best_edlae_res[f'NDCG@{k}']):
            best_edlae_res = res
            best_edlae_dropout = p

    print(f"\nBest EDLAE dropout: {best_edlae_dropout} "
          f"(NDCG@{k}={best_edlae_res[f'NDCG@{k}']:.4f})")

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
        res = evaluate_at_ks(_StandaloneWrapper(edlae_lap), train,
                             test_positive, ks=ks)
        w_stat, w_p, w_n = wilcoxon_vs_baseline(res_ease, res, k=k)
        print(f"  NDCG@{k}={res[f'NDCG@{k}']:.4f} "
              f"NDCG@{max(ks)}={res[f'NDCG@{max(ks)}']:.4f} "
              f"p(vs EASE)={w_p:.2e}  ({t:.1f}s)")
        _row = {
            'dataset': dataset,
            'model': 'EDLAE-Laplacian',
            'dropout': best_edlae_dropout,
            'gamma': gamma,
            'graph_source': graph_source,
            'normalise': normalise,
            'train_time_s': t,
        }
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                _row[f'{m}@{kk}'] = res[f'{m}@{kk}']
        _row['NDCG@k']    = res[f'NDCG@{k}']
        _row['MAP@k']     = res[f'MAP@{k}']
        _row['HitRate@k'] = res[f'HitRate@{k}']
        _row['Recall@k']  = res[f'Recall@{k}']
        _row['wilcoxon_stat_vs_EASE']    = w_stat
        _row['wilcoxon_p_vs_EASE']       = w_p
        _row['wilcoxon_n_pairs_vs_EASE'] = w_n
        rows.append(_row)

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
    ax.axhline(best_edlae_res[f'NDCG@{k}'], ls='--', color='tab:orange',
               label=f"EDLAE baseline ({best_edlae_res[f'NDCG@{k}']:.4f})")
    ax.axhline(res_ease[f'NDCG@{k}'], ls=':', color='grey',
               label=f"EASE reference ({res_ease[f'NDCG@{k}']:.4f})")
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
    p.add_argument('--ks', type=str, default='10,20',
                   help='Comma-separated list of cut-offs for multi-k '
                        'evaluation (NDCG@10, NDCG@20, ...). '
                        'Default: "10,20".')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]

    dropouts = None
    if args.dropout is not None:
        dropouts = [args.dropout]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k,
        lambda_=args.lambda_, gammas=gammas,
        graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        dropouts=dropouts, normalise=args.normalise,
        ks=ks)


if __name__ == '__main__':
    main()
