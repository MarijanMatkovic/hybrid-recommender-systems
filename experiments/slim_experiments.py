"""
SLIM baseline + Laplacian-regularized SLIM.

Runs:
    1) Vanilla SLIM on a 3x3 (alpha, l1_ratio) ElasticNet grid to pick
       the best-tuned SLIM baseline (earlier runs used a single
       undertuned config, which put SLIM below EASE and made the
       "SLIM-Lap > SLIM" claim suspect).
    2) Laplacian-SLIM for a grid of gamma values (ISTA solver) on top
       of the best-tuned SLIM hyper-parameters.

Compares both against EASE as reference. Writes a CSV + NDCG-vs-gamma
plot under ``results/slim/``.

SLIM fits column-by-column and is much slower than EASE. On ``ml-small``
it takes ~20-40 s; on ``ml-1m`` it is not practical without further
optimisation -- for thesis purposes, use ml-small for SLIM experiments
and demonstrate that the Laplacian generalisation applies to SLIM too.

Hyperparameter sweep contract:
    * If the caller passes explicit ``l1_reg``/``beta``, we run a
      single-point fit (back-compat and smoke-test contract).
    * Otherwise we sweep
          alpha    in {1e-4, 1e-3, 1e-2}
          l1_ratio in {0.1, 0.5, 0.9}
      (9 configurations) and pick the row with the best NDCG@k as the
      "SLIM" reference for the Laplacian comparison. All 9 grid rows
      are written to the CSV (model="SLIM", columns ``slim_alpha``,
      ``slim_l1_ratio``) so the table is auditable.

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
        l1_reg=None, beta=None, positive=True,
        alphas=None, l1_ratios=None,
        gammas=None, graph_source='rp3beta',
        rp3_beta=0.6, topK=200,
        max_iter=5000, tol=1e-4,
        n_iter=200,
        normalise='none', ks=(10, 20),
        split_mode='temporal', split_seed=0,
        out_dir=None):
    """Run vanilla SLIM (optionally on a (alpha, l1_ratio) grid) and a
    Laplacian-SLIM gamma sweep on top of the best SLIM configuration.

    Parameters
    ----------
    l1_reg, beta : float or None
        Single-point ElasticNet regularisation. If EITHER is non-None
        the grid is skipped and one SLIM row is produced -- this keeps
        back-compat with older smoke tests and ad-hoc calls.
    alphas : Sequence[float] or None
        ElasticNet ``alpha = l1_reg + beta`` values to grid over when
        ``l1_reg``/``beta`` are not supplied. Default: [1e-4, 1e-3, 1e-2].
    l1_ratios : Sequence[float] or None
        ElasticNet ``l1_ratio = l1_reg / alpha`` values. Default:
        [0.1, 0.5, 0.9].
    max_iter : int
        Per-column CD iterations (default 5000, see ``SLIM.fit`` docstring
        for why 1000 is too low).
    """
    if gammas is None:
        gammas = [0.3, 1.0, 3.0, 10.0, 30.0]
    ks = tuple(sorted(set(list(ks) + [k])))

    # ---- Resolve the (alpha, l1_ratio) grid ----
    # Legacy single-point path: caller supplied explicit l1_reg and/or
    # beta. We leave any missing value at the old single-point default
    # so older scripts still work.
    if (l1_reg is not None) or (beta is not None):
        l1_reg_eff = 1e-4 if l1_reg is None else l1_reg
        beta_eff   = 1e-3 if beta   is None else beta
        alpha_here = l1_reg_eff + beta_eff
        ratio_here = l1_reg_eff / (alpha_here + 1e-12)
        slim_grid = [(alpha_here, ratio_here)]
    else:
        if alphas is None:
            alphas = [1e-4, 1e-3, 1e-2]
        if l1_ratios is None:
            l1_ratios = [0.1, 0.5, 0.9]
        slim_grid = [(float(a), float(r))
                     for a in alphas for r in l1_ratios]

    out_dir = ensure_results_dir('slim' if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(
        dataset, split_mode=split_mode, split_seed=split_seed)
    bucket_of = item_popularity_buckets(train, n_buckets=5)
    bucket_rows_all = []

    rows = []

    # ---- 1) Vanilla SLIM sweep over (alpha, l1_ratio) ----
    print(f"\n[SLIM] vanilla ElasticNet grid -- {len(slim_grid)} configs, "
          f"max_iter={max_iter}")
    slim_results = []        # list of (alpha, l1_ratio, slim_model, res, t)
    for alpha_v, ratio_v in slim_grid:
        l1_part = alpha_v * ratio_v
        l2_part = alpha_v * (1.0 - ratio_v)
        t0 = time.time()
        slim = SLIM()
        slim.fit(train, l1_reg=l1_part, beta=l2_part, positive=positive,
                 max_iter=max_iter, tol=tol)
        t_slim = time.time() - t0
        res_slim = evaluate_at_ks(_StandaloneWrapper(slim), train,
                                  test_positive, ks=ks)
        print(f"  SLIM alpha={alpha_v:<9g} l1_ratio={ratio_v:<4g} "
              f"NDCG@{k}={res_slim[f'NDCG@{k}']:.4f} "
              f"NDCG@{max(ks)}={res_slim[f'NDCG@{max(ks)}']:.4f} "
              f"({t_slim:.1f}s)")
        _row = {
            'dataset': dataset, 'model': 'SLIM',
            'gamma': 0.0, 'graph_source': None, 'normalise': None,
            'slim_alpha': alpha_v, 'slim_l1_ratio': ratio_v,
            'train_time_s': t_slim,
        }
        _row.update(metric_cols_at_ks(res_slim, ks, primary_k=k))
        _row.update(wilcoxon_blank_columns('wilcoxon_vs_SLIM'))
        _row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
        rows.append(_row)
        _slim_buckets = bucketed_metrics_at_ks(
            _StandaloneWrapper(slim), train, test_positive,
            ks=ks, bucket_of=bucket_of)
        bucket_rows_all.extend(
            bucket_rows({'dataset': dataset, 'model': 'SLIM',
                         'gamma': 0.0, 'graph_source': None,
                         'normalise': None,
                         'slim_alpha': alpha_v,
                         'slim_l1_ratio': ratio_v},
                        _slim_buckets, ks, n_buckets=5))
        slim_results.append((alpha_v, ratio_v, slim, res_slim, t_slim))

    # Pick best SLIM by NDCG@k; that's the reference the Laplacian
    # variants are compared against. Using the best config avoids the
    # "undertuned baseline" critique -- if Lap still wins here, it
    # wins against a well-tuned competitor.
    best_idx = int(np.argmax([r[3][f'NDCG@{k}'] for r in slim_results]))
    best_alpha, best_ratio, slim, res_slim, _ = slim_results[best_idx]
    print(f"\n  [SLIM baseline picked] alpha={best_alpha:g} "
          f"l1_ratio={best_ratio:g} "
          f"NDCG@{k}={res_slim[f'NDCG@{k}']:.4f}")

    # ---- 2) EASE reference (same data) ----
    print("\n[reference] EASE")
    t0 = time.time()
    ease_ref = HybridEASE_RP3beta()
    ease_ref.fit(train, method='score', fusion_alpha=1.0,
                 ease_lambda=500 if dataset == 'ml-1m' else 200,
                 rp3_alpha=1.0, rp3_beta=rp3_beta, rp3_topK=topK)
    # hybrid.fit(method='score', fusion_alpha=1.0) already sets self.pred
    # to make_pred(X, B) (lazy on Netflix); no eager override needed.
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
    _row.update(metric_cols_at_ks(res_ease, ks, primary_k=k))
    # EASE vs SLIM: paired test over common users.
    w_ease_vs_slim = wilcoxon_vs_baseline(res_slim, res_ease, k=k)
    _row.update(wilcoxon_columns('wilcoxon_vs_SLIM', w_ease_vs_slim))
    _row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
    rows.append(_row)
    _ease_buckets = bucketed_metrics_at_ks(
        ease_ref, train, test_positive, ks=ks, bucket_of=bucket_of)
    bucket_rows_all.extend(
        bucket_rows({'dataset': dataset, 'model': 'EASE',
                     'gamma': 0.0, 'graph_source': None,
                     'normalise': None},
                    _ease_buckets, ks, n_buckets=5))

    # ---- 3) Build Laplacian once ----
    X = ease_ref.ease.X
    W = build_graph(X, source=graph_source, topK=topK,
                    rp3_beta=rp3_beta, implicit=True)
    L_dense, _ = build_laplacian(W, normalise=normalise)
    G_diag_mean = float(np.mean(np.array(
        X.multiply(X).sum(axis=0)).flatten()))  # ≈ mean(diag(X^T X))
    L_scaled = _scale_laplacian(L_dense, G_diag_mean)

    # ---- 4) Laplacian SLIM sweep on top of the best SLIM config ----
    # Use the winning (alpha, l1_ratio) from the vanilla grid as the
    # Laplacian's ridge / L1 strengths too. This keeps the comparison
    # fair: the Laplacian adds a *third* term on top of an already-
    # well-tuned SLIM, rather than competing against a crippled SLIM.
    lap_l1 = best_alpha * best_ratio
    lap_l2 = best_alpha * (1.0 - best_ratio)
    for gamma in gammas:
        print(f"\n[SLIM-Laplacian] gamma={gamma}  normalise={normalise}")
        t0 = time.time()
        slim_lap = SLIM()
        slim_lap.fit_laplacian(
            train, L_scaled=L_scaled, beta=lap_l2, l1_reg=lap_l1,
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
            'normalise': normalise,
            'slim_alpha': best_alpha, 'slim_l1_ratio': best_ratio,
            'train_time_s': t,
        }
        _row.update(metric_cols_at_ks(res, ks, primary_k=k))
        _row.update(wilcoxon_columns('wilcoxon_vs_SLIM', w_slim))
        _row.update(wilcoxon_columns('wilcoxon_vs_EASE', w_ease))
        rows.append(_row)
        _lap_buckets = bucketed_metrics_at_ks(
            _StandaloneWrapper(slim_lap), train, test_positive,
            ks=ks, bucket_of=bucket_of)
        bucket_rows_all.extend(
            bucket_rows({'dataset': dataset,
                         'model': 'SLIM-Laplacian',
                         'gamma': gamma,
                         'graph_source': graph_source,
                         'normalise': normalise,
                         'slim_alpha': best_alpha,
                         'slim_l1_ratio': best_ratio},
                        _lap_buckets, ks, n_buckets=5))

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    seed_tag = f'_seed{split_seed}' if split_mode == 'random' else ''
    stem = f'slim_{dataset}{suffix}{seed_tag}'
    csv_path = out_dir / f'{stem}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    bpath = write_buckets_csv(out_dir, stem, bucket_rows_all)
    if bpath is not None:
        print(f"Saved per-bucket CSV to {bpath}")

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
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    # Single-point overrides: if EITHER is supplied we skip the grid
    # and run one SLIM fit at those exact values.
    p.add_argument('--l1_reg', type=float, default=None,
                   help='Single-point L1 strength. If given (together '
                        'with --beta, or alone), overrides the '
                        '--alphas/--l1_ratios grid sweep.')
    p.add_argument('--beta', type=float, default=None,
                   help='Single-point L2 strength. See --l1_reg.')
    # 3x3 grid (default). Both lists can be extended/shortened.
    p.add_argument('--alphas', type=str, default=None,
                   help='Comma-separated ElasticNet alpha (=l1+l2) grid. '
                        'Default: "1e-4,1e-3,1e-2".')
    p.add_argument('--l1_ratios', type=str, default=None,
                   help='Comma-separated ElasticNet l1_ratio grid. '
                        'Default: "0.1,0.5,0.9".')
    p.add_argument('--max_iter', type=int, default=5000,
                   help='Per-column CD iterations cap (default 5000; '
                        'lower values trigger sklearn convergence '
                        'warnings and depress the SLIM baseline).')
    p.add_argument('--tol', type=float, default=1e-4)
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
    p.add_argument('--split_mode', default='temporal',
                   choices=['temporal', 'random'],
                   help='Evaluation protocol. "temporal" (default) uses '
                        'the deterministic temporal split. "random" uses '
                        'a random 80/20 per-user split (primary protocol).')
    p.add_argument('--split_seed', type=int, default=0,
                   help='Random-split seed. Only used when '
                        '--split_mode=random. Default: 0.')
    p.add_argument('--out_dir', default=None,
                   help='Override results directory. Default: '
                        '"results/slim". Use this to keep primary-split '
                        'and temporal-split runs in separate folders.')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]

    alphas = None
    if args.alphas:
        alphas = [float(x) for x in args.alphas.split(',') if x.strip()]

    l1_ratios = None
    if args.l1_ratios:
        l1_ratios = [float(x)
                     for x in args.l1_ratios.split(',') if x.strip()]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k,
        l1_reg=args.l1_reg, beta=args.beta,
        alphas=alphas, l1_ratios=l1_ratios,
        max_iter=args.max_iter, tol=args.tol,
        gammas=gammas, graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        n_iter=args.n_iter, normalise=args.normalise, ks=ks,
        split_mode=args.split_mode, split_seed=args.split_seed,
        out_dir=args.out_dir)


if __name__ == '__main__':
    main()
