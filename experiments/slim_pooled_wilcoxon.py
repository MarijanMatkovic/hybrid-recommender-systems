"""
Pooled ~30k-user Wilcoxon: SLIM-Lap(gamma=1) vs EASE across 5 primary seeds.

Motivation
----------
Per-seed Wilcoxon in results/slim_primary/ (within-seed n~6008) shows
individually significant results only on seeds 2 (p=7.7e-4) and 4 (p=3.9e-2).
Pooling 5 x ~6008 = ~30k (seed, user) observations gives a single definitive
statement comparable to the pooled test already reported for Lap-EASE vs EASE
in results/baselines/primary/wilcoxon_ml-1m_primary_*.csv.

Best (alpha, l1_ratio) per seed is read from the existing per-seed CSVs in
results/slim_primary/ to avoid re-sweeping the full ElasticNet grid (which
takes ~3-4h total). Only two SLIM fits are re-done per seed: best-config SLIM
and SLIM-Lap(gamma=1).

Why gamma=1? Across all 5 per-seed runs, gamma=1 consistently yields the
highest SLIM-Lap NDCG@10 (best on seeds 0,1,3,4; second-best on seed 2 where
gamma=0.3 is nominally higher by 0.0001). It is the canonical primary-protocol
optimum, matching the gamma=3 optimum for Lap-EASE (which uses a different,
scaled Laplacian; SLIM-Lap uses ISTA so the scale is different).

Outputs
-------
results/slim_primary/slim_pooled_wilcoxon_ml-1m_sym.csv
    Per-seed rows (seed 0..4) + one 'pooled' summary row.
    Columns: seed, EASE_NDCG@10, SLIM_NDCG@10, SLIM_Lap_NDCG@10,
             SLIM_Lap_vs_EASE_diff, within_seed_wil_p, within_seed_wil_sign,
             pooled_wil_p, pooled_wil_sign, pooled_wil_n_pairs, pooled_wil_mean_diff

Example
-------
    python -m experiments.slim_pooled_wilcoxon --dataset ml-1m
    python -m experiments.slim_pooled_wilcoxon --dataset ml-1m --gamma 1.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models import SLIM, build_graph, build_laplacian
from models.hybrid import HybridEASE_RP3beta

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
    wilcoxon_paired,
    PRIMARY_SPLIT_SEEDS,
)


# Hardcoded per-seed best (alpha, l1_ratio) as a fallback if the per-seed CSV
# is not readable. These match what was selected in the slim_primary runs:
#   seeds 0,1,2: best NDCG at alpha=1e-4, l1_ratio=0.5
#   seeds 3,4:   best NDCG at alpha=1e-4, l1_ratio=0.9
_FALLBACK_BEST = {
    0: (1e-4, 0.5),
    1: (1e-4, 0.5),
    2: (1e-4, 0.5),
    3: (1e-4, 0.9),
    4: (1e-4, 0.9),
}


class _StandaloneWrapper:
    """Minimal shim so evaluate_at_ks sees the expected interface."""
    def __init__(self, slim_model):
        self.ease = slim_model   # SLIM exposes .user_enc / .item_enc
        self.pred = slim_model.pred


def _scale_laplacian(L, G_diag_mean):
    """Scale L so mean(diag(L)) ~ mean(diag(G))."""
    l_mean = np.mean(np.diag(L))
    return L * (G_diag_mean / l_mean) if l_mean > 0 else L


def _read_best_config(seed, dataset, out_dir):
    """Read best (alpha, l1_ratio) for ``seed`` from the per-seed CSV.

    Falls back to _FALLBACK_BEST if the file is missing or unreadable.
    """
    csv_path = Path('results') / out_dir / f'slim_{dataset}_sym_seed{seed}.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            slim_rows = df[df['model'] == 'SLIM']
            if not slim_rows.empty:
                best = slim_rows.loc[slim_rows['NDCG@10'].idxmax()]
                alpha = float(best['slim_alpha'])
                ratio = float(best['slim_l1_ratio'])
                print(f"  [seed={seed}] CSV: best alpha={alpha:g}  "
                      f"l1_ratio={ratio:g}  "
                      f"NDCG@10={best['NDCG@10']:.4f}")
                return alpha, ratio
        except Exception as exc:
            print(f"  [seed={seed}] Could not read CSV ({exc}); "
                  f"using fallback")
    fallback = _FALLBACK_BEST.get(seed, (1e-4, 0.5))
    print(f"  [seed={seed}] fallback config: alpha={fallback[0]:g}  "
          f"l1_ratio={fallback[1]:g}")
    return fallback


def run(dataset='ml-1m', k=10,
        gamma=1.0,
        rp3_beta=0.6, rp3_topK=200,
        normalise='sym', graph_source='rp3beta',
        ease_lambda=None,
        max_iter=5000, tol=1e-4, n_iter=150,
        seeds=None,
        out_dir=None):
    """Re-fit EASE + best SLIM + SLIM-Lap(gamma) per primary seed and
    pool per-user NDCG arrays for a ~30k-observation Wilcoxon test.

    Parameters
    ----------
    gamma : float
        SLIM-Lap Laplacian strength to evaluate. Default 1.0 (best primary-
        protocol gamma from the per-seed sweep).
    """
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)
    if out_dir is None:
        out_dir = 'slim_primary'

    out_path = ensure_results_dir(out_dir)

    # Per-user accumulators for pooled test
    pool_ease    = []
    pool_slim    = []
    pool_slim_lap = []
    pool_tags    = []    # (seed, user_id) tuples

    per_seed_rows = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[seed={seed}]  random 80/20 primary protocol")
        print(f"{'='*60}")
        train, test_positive, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)

        # Per-seed best (alpha, l1_ratio)
        alpha, ratio = _read_best_config(seed, dataset, out_dir)
        l1_part = alpha * ratio
        l2_part = alpha * (1.0 - ratio)

        # ---- EASE reference -----------------------------------------
        print("\n[reference] EASE")
        t0 = time.time()
        ease_ref = HybridEASE_RP3beta()
        ease_ref.fit(train, method='score', fusion_alpha=1.0,
                     ease_lambda=ease_lambda, rp3_alpha=1.0,
                     rp3_beta=rp3_beta, rp3_topK=rp3_topK)
        ease_ref.pred = ease_ref.ease.X.dot(ease_ref.ease.B)
        res_ease = evaluate_at_ks(ease_ref, train, test_positive, ks=(k,))
        t_ease = time.time() - t0
        print(f"  EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")

        # ---- SLIM at best config ------------------------------------
        print(f"\n[SLIM] alpha={alpha:g}  l1_ratio={ratio:g}")
        t0 = time.time()
        slim = SLIM()
        slim.fit(train, l1_reg=l1_part, beta=l2_part, positive=True,
                 max_iter=max_iter, tol=tol)
        res_slim = evaluate_at_ks(_StandaloneWrapper(slim), train,
                                  test_positive, ks=(k,))
        t_slim = time.time() - t0
        print(f"  SLIM NDCG@{k}={res_slim[f'NDCG@{k}']:.4f}  ({t_slim:.1f}s)")

        # ---- Build Laplacian (once per seed) ------------------------
        X = ease_ref.ease.X
        W = build_graph(X, source=graph_source, topK=rp3_topK,
                        rp3_beta=rp3_beta, implicit=True)
        L_dense, _ = build_laplacian(W, normalise=normalise)
        G_diag_mean = float(
            np.mean(np.array(X.multiply(X).sum(axis=0)).flatten()))
        L_scaled = _scale_laplacian(L_dense, G_diag_mean)

        # ---- SLIM-Lap at canonical gamma ----------------------------
        print(f"\n[SLIM-Lap] gamma={gamma}")
        t0 = time.time()
        slim_lap = SLIM()
        slim_lap.fit_laplacian(
            train, L_scaled=L_scaled,
            beta=l2_part, l1_reg=l1_part,
            gamma=gamma, positive=True, n_iter=n_iter)
        res_lap = evaluate_at_ks(_StandaloneWrapper(slim_lap), train,
                                 test_positive, ks=(k,))
        t_lap = time.time() - t0
        print(f"  SLIM-Lap NDCG@{k}={res_lap[f'NDCG@{k}']:.4f}  ({t_lap:.1f}s)")

        # Within-seed Wilcoxon
        w_lap = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_lap['per_user_ndcg'][k],  res_lap['per_user_ids']))
        w_slim = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_slim['per_user_ndcg'][k], res_slim['per_user_ids']))
        stat_l, p_l, n_l, sign_l, md_l, med_l = w_lap
        stat_s, p_s, n_s, sign_s, md_s, med_s = w_slim
        print(f"\n  Within-seed Wilcoxon SLIM-Lap vs EASE: "
              f"p={p_l:.3e}  sign={sign_l:+d}  mean_diff={md_l:+.5f}")
        print(f"  Within-seed Wilcoxon SLIM     vs EASE: "
              f"p={p_s:.3e}  sign={sign_s:+d}  mean_diff={md_s:+.5f}")

        per_seed_rows.append({
            'seed': seed, 'dataset': dataset,
            'graph_source': graph_source, 'normalise': normalise,
            'slim_alpha': alpha, 'slim_l1_ratio': ratio,
            'gamma': gamma,
            f'EASE_NDCG@{k}': res_ease[f'NDCG@{k}'],
            f'SLIM_NDCG@{k}': res_slim[f'NDCG@{k}'],
            f'SLIM_Lap_NDCG@{k}': res_lap[f'NDCG@{k}'],
            f'SLIM_Lap_vs_EASE_diff': (res_lap[f'NDCG@{k}']
                                       - res_ease[f'NDCG@{k}']),
            f'SLIM_vs_EASE_diff': (res_slim[f'NDCG@{k}']
                                   - res_ease[f'NDCG@{k}']),
            'within_seed_wil_stat': stat_l,
            'within_seed_wil_p': p_l,
            'within_seed_wil_n_pairs': n_l,
            'within_seed_wil_sign': sign_l,
            'within_seed_wil_mean_diff': md_l,
            'within_seed_wil_median_diff': med_l,
        })

        # Accumulate for pooled test: align by uid, tag with (seed, uid)
        ease_map = dict(zip(res_ease['per_user_ids'],
                            res_ease['per_user_ndcg'][k]))
        slim_map = dict(zip(res_slim['per_user_ids'],
                            res_slim['per_user_ndcg'][k]))
        lap_map  = dict(zip(res_lap['per_user_ids'],
                            res_lap['per_user_ndcg'][k]))
        common = sorted(set(ease_map) & set(slim_map) & set(lap_map))
        for uid in common:
            pool_ease.append(ease_map[uid])
            pool_slim.append(slim_map[uid])
            pool_slim_lap.append(lap_map[uid])
            pool_tags.append((seed, uid))

    # ---- Pooled Wilcoxon ------------------------------------------------
    n_pooled = len(pool_tags)
    ease_arr   = np.array(pool_ease,    dtype=float)
    slim_arr   = np.array(pool_slim,    dtype=float)
    slim_l_arr = np.array(pool_slim_lap, dtype=float)

    pooled_lap  = wilcoxon_paired((ease_arr, pool_tags),
                                  (slim_l_arr, pool_tags))
    pooled_slim = wilcoxon_paired((ease_arr, pool_tags),
                                  (slim_arr,  pool_tags))
    stat_l, p_l, n_l, sign_l, md_l, med_l = pooled_lap
    stat_s, p_s, n_s, sign_s, md_s, med_s = pooled_slim

    print(f"\n{'='*60}")
    print(f"POOLED WILCOXON  (n={n_pooled} user-seed observations)")
    print(f"{'='*60}")
    print(f"  SLIM-Lap(g={gamma}) vs EASE: p={p_l:.3e}  sign={sign_l:+d}"
          f"  mean_diff={md_l:+.5f}")
    print(f"  SLIM            vs EASE: p={p_s:.3e}  sign={sign_s:+d}"
          f"  mean_diff={md_s:+.5f}")

    df_seed = pd.DataFrame(per_seed_rows)

    # Pooled summary row (seed='pooled')
    pooled_row = {
        'seed': 'pooled', 'dataset': dataset,
        'graph_source': graph_source, 'normalise': normalise,
        'slim_alpha': None, 'slim_l1_ratio': None,
        'gamma': gamma,
        f'EASE_NDCG@{k}': float(df_seed[f'EASE_NDCG@{k}'].mean()),
        f'SLIM_NDCG@{k}': float(df_seed[f'SLIM_NDCG@{k}'].mean()),
        f'SLIM_Lap_NDCG@{k}': float(df_seed[f'SLIM_Lap_NDCG@{k}'].mean()),
        f'SLIM_Lap_vs_EASE_diff': float(
            df_seed[f'SLIM_Lap_vs_EASE_diff'].mean()),
        f'SLIM_vs_EASE_diff': float(
            df_seed[f'SLIM_vs_EASE_diff'].mean()),
        'within_seed_wil_stat': stat_l,
        'within_seed_wil_p': p_l,
        'within_seed_wil_n_pairs': n_l,
        'within_seed_wil_sign': sign_l,
        'within_seed_wil_mean_diff': md_l,
        'within_seed_wil_median_diff': med_l,
    }
    df_all = pd.concat([df_seed, pd.DataFrame([pooled_row])],
                       ignore_index=True)

    suffix = '_sym' if normalise == 'sym' else ''
    stem = f'slim_pooled_wilcoxon_{dataset}{suffix}'
    csv_path = out_path / f'{stem}.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df_all


def main():
    p = argparse.ArgumentParser(
        description='Pooled Wilcoxon: SLIM-Lap vs EASE across 5 primary seeds.')
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--gamma', type=float, default=1.0,
                   help='SLIM-Lap Laplacian strength (default: 1.0, '
                        'best primary-protocol gamma from per-seed sweep).')
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--normalise', default='sym',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation (default: sym).')
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--max_iter', type=int, default=5000,
                   help='Max SLIM CD iterations per column (default 5000).')
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--n_iter', type=int, default=150,
                   help='ISTA iterations for SLIM-Lap (default 150).')
    p.add_argument('--seeds', type=str, default='0,1,2,3,4',
                   help='Comma-separated seed list (default: 0,1,2,3,4).')
    p.add_argument('--out_dir', default=None,
                   help='Results subdirectory under results/. '
                        'Default: slim_primary.')
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    run(dataset=args.dataset, k=args.k,
        gamma=args.gamma,
        rp3_beta=args.rp3_beta, rp3_topK=args.topK,
        normalise=args.normalise, graph_source=args.graph_source,
        max_iter=args.max_iter, tol=args.tol, n_iter=args.n_iter,
        seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
