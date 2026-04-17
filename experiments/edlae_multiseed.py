"""
Multi-seed EDLAE vs Laplacian-EDLAE.

Since EDLAE is fully deterministic given the training matrix (its
closed-form uses the expectation of the dropout mask, not a sampled
mask), the meaningful source of variability across runs is the
**train/test split**. This script repeats the EDLAE baseline and the
best Laplacian-EDLAE config across ``n_seeds`` random per-user splits
and reports mean ± std for each metric.

Why this matters: on ML-1M the absolute gain of Laplacian-EDLAE over
vanilla EDLAE can be small (sub-0.001 NDCG@10). Reporting a single
split leaves the reader uncertain whether the gain is signal or noise.
mean ± std over 3 splits (plus the Wilcoxon test on per-user NDCG
inside the head_tail_analysis script, which runs on a single split)
gives two independent lines of evidence.

Example:
    python -m experiments.edlae_multiseed \\
        --dataset ml-1m --n_seeds 3 --dropout 0.5 --gamma 30
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

from experiments._shared import ensure_results_dir, load_dataset


class _StandaloneWrapper:
    """Shim so ``evaluate_at_ks`` sees the same ``.ease`` + ``.pred``
    interface a hybrid model exposes."""
    def __init__(self, model):
        self.ease = model
        self.pred = model.pred


def _scale_laplacian(L, G_diag_mean):
    l_diag_mean = np.mean(np.diag(L))
    if l_diag_mean > 0:
        return L * (G_diag_mean / l_diag_mean)
    return L


def _summarise(rows, group_keys, metric_keys):
    """Mean ± std over seeds for each (model, config) combination.

    ``rows`` is a list of per-seed result dicts. Returns a DataFrame
    with one row per ``group_keys`` tuple and two columns per metric
    (``<m>_mean``, ``<m>_std``).
    """
    df = pd.DataFrame(rows)
    agg_spec = {m: ['mean', 'std', 'count'] for m in metric_keys}
    out = df.groupby(list(group_keys), dropna=False).agg(agg_spec)
    # Flatten the MultiIndex columns produced by ``agg``.
    out.columns = [f'{m}_{stat}' for m, stat in out.columns]
    return out.reset_index()


def run(dataset='ml-small', k=10,
        lambda_=None, dropout=0.5, gamma=30.0,
        graph_source='rp3beta', rp3_beta=0.6, topK=200,
        normalise='none', ks=(10, 20),
        n_seeds=3, split_seeds=None,
        out_dir=None):
    """Run EDLAE + Laplacian-EDLAE across several random splits.

    Parameters
    ----------
    n_seeds : int
        Number of random splits (ignored if ``split_seeds`` is given).
    split_seeds : Sequence[int] | None
        Explicit list of seeds (e.g. [0, 1, 2, 3, 4]). Overrides
        ``n_seeds``.
    """
    if lambda_ is None:
        lambda_ = 500 if dataset == 'ml-1m' else 200
    ks = tuple(sorted(set(list(ks) + [k])))
    if split_seeds is None:
        split_seeds = list(range(n_seeds))
    split_seeds = list(split_seeds)

    out_dir = ensure_results_dir('edlae_multiseed'
                                 if out_dir is None else out_dir)

    per_seed_rows = []

    for seed in split_seeds:
        print(f"\n{'=' * 60}")
        print(f" split seed = {seed}")
        print('=' * 60)

        train, test_positive, _ = load_dataset(dataset,
                                               split_mode='random',
                                               split_seed=seed)

        # --- EASE reference ---
        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=lambda_, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=topK)
        ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
        res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=ks)
        t_ease = time.time() - t0
        print(f"  EASE       NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_ease[f'NDCG@{max(ks)}']:.4f} "
              f"({t_ease:.1f}s)")

        row_ease = {'dataset': dataset, 'seed': seed, 'model': 'EASE',
                    'dropout': 0.0, 'gamma': 0.0,
                    'graph_source': None, 'normalise': None,
                    'train_time_s': t_ease}
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                row_ease[f'{m}@{kk}'] = res_ease[f'{m}@{kk}']
        per_seed_rows.append(row_ease)

        # --- EDLAE baseline ---
        t0 = time.time()
        edlae = EDLAE()
        edlae.fit(train, lambda_=lambda_, dropout=dropout)
        t_edl = time.time() - t0
        res_edl = evaluate_at_ks(_StandaloneWrapper(edlae), train,
                                 test_positive, ks=ks)
        print(f"  EDLAE      NDCG@{k}={res_edl[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_edl[f'NDCG@{max(ks)}']:.4f} "
              f"({t_edl:.1f}s)")

        row_edl = {'dataset': dataset, 'seed': seed, 'model': 'EDLAE',
                   'dropout': dropout, 'gamma': 0.0,
                   'graph_source': None, 'normalise': None,
                   'train_time_s': t_edl}
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                row_edl[f'{m}@{kk}'] = res_edl[f'{m}@{kk}']
        per_seed_rows.append(row_edl)

        # --- Laplacian-EDLAE (build L on this split's train X) ---
        X = ease_ref.ease.X
        W = build_graph(X, source=graph_source, topK=topK,
                        rp3_beta=rp3_beta, implicit=True)
        L_dense, _ = build_laplacian(W, normalise=normalise)
        G_diag_mean = float(np.mean(np.array(
            X.multiply(X).sum(axis=0)).flatten()))
        L_scaled = _scale_laplacian(L_dense, G_diag_mean)

        t0 = time.time()
        edlae_lap = EDLAE()
        edlae_lap.fit_laplacian(train, L_scaled=L_scaled,
                                lambda_=lambda_, dropout=dropout,
                                gamma=gamma)
        t_lap = time.time() - t0
        res_lap = evaluate_at_ks(_StandaloneWrapper(edlae_lap), train,
                                 test_positive, ks=ks)
        print(f"  EDLAE-Lap  NDCG@{k}={res_lap[f'NDCG@{k}']:.4f}  "
              f"NDCG@{max(ks)}={res_lap[f'NDCG@{max(ks)}']:.4f} "
              f"({t_lap:.1f}s)")

        row_lap = {'dataset': dataset, 'seed': seed,
                   'model': 'EDLAE-Laplacian',
                   'dropout': dropout, 'gamma': gamma,
                   'graph_source': graph_source, 'normalise': normalise,
                   'train_time_s': t_lap}
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                row_lap[f'{m}@{kk}'] = res_lap[f'{m}@{kk}']
        per_seed_rows.append(row_lap)

    # ---- Long-form per-seed CSV ----
    df = pd.DataFrame(per_seed_rows)
    suffix = '_sym' if normalise == 'sym' else ''
    csv_path = out_dir / f'edlae_multiseed_{dataset}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-seed CSV to {csv_path}")

    # ---- Aggregated summary ----
    metric_cols = [f'{m}@{kk}' for kk in ks
                   for m in ('NDCG', 'MAP', 'HitRate', 'Recall')]
    summary = _summarise(per_seed_rows,
                         group_keys=('dataset', 'model',
                                     'dropout', 'gamma', 'graph_source'),
                         metric_keys=metric_cols)
    sum_path = out_dir / f'edlae_multiseed_{dataset}{suffix}_summary.csv'
    summary.to_csv(sum_path, index=False)
    print(f"Saved aggregated summary to {sum_path}")

    # ---- Pretty-print a NDCG@k column comparison ----
    print("\nMean ± std across seeds (NDCG):")
    for kk in ks:
        col_mean = f'NDCG@{kk}_mean'
        col_std  = f'NDCG@{kk}_std'
        if col_mean in summary.columns:
            line_rows = summary[['model', col_mean, col_std]]
            print(f"\n  NDCG@{kk}")
            for _, r in line_rows.iterrows():
                mu = r[col_mean]
                sd = r[col_std] if pd.notna(r[col_std]) else 0.0
                print(f"    {r['model']:<18} {mu:.4f} ± {sd:.4f}")

    # ---- Bar plot: NDCG@k per model with ± std whiskers ----
    col_mean = f'NDCG@{k}_mean'
    col_std  = f'NDCG@{k}_std'
    if col_mean in summary.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = np.arange(len(summary))
        ax.bar(xs, summary[col_mean],
               yerr=summary[col_std].fillna(0.0),
               capsize=5, color=['#888', '#4c72b0', '#55a868'])
        ax.set_xticks(xs)
        ax.set_xticklabels(summary['model'], rotation=0)
        ax.set_ylabel(f'NDCG@{k} (mean ± std, n={len(split_seeds)} seeds)')
        title_extra = f' [sym L]' if normalise == 'sym' else ''
        ax.set_title(f'EDLAE multi-seed — {dataset}{title_extra}')
        ax.grid(True, axis='y', ls=':', alpha=0.5)
        fig.tight_layout()
        png_path = out_dir / f'edlae_multiseed_{dataset}{suffix}.png'
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        print(f"Saved plot to {png_path}")

    return df, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--lambda_', dest='lambda_', type=float, default=None)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--gamma', type=float, default=30.0,
                   help='Laplacian strength for the Laplacian-EDLAE '
                        'condition. Pick this from a prior gamma sweep.')
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'])
    p.add_argument('--ks', type=str, default='10,20',
                   help='Comma-separated list of cut-offs for multi-k '
                        'evaluation. Default: "10,20".')
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--split_seeds', type=str, default=None,
                   help='Explicit comma-separated seed list; overrides '
                        '--n_seeds if given.')
    args = p.parse_args()

    split_seeds = None
    if args.split_seeds:
        split_seeds = [int(x) for x in args.split_seeds.split(',')]

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k,
        lambda_=args.lambda_, dropout=args.dropout, gamma=args.gamma,
        graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, topK=args.topK,
        normalise=args.normalise, ks=ks,
        n_seeds=args.n_seeds, split_seeds=split_seeds)


if __name__ == '__main__':
    main()
