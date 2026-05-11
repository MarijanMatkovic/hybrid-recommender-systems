"""
Score-fusion ablation -- roadmap items 1 (RRF + CombMNZ) and 9 (inverse-
variance fusion using Wager-Wang-Liang dropout variance).

For each seed of the primary protocol, fits the three base models
(EASE, RP3beta, EDLAE) and compares 4 fusion strategies against each
single base:

  * Linear alpha-blend (existing Hybrid-Score, fusion_alpha=0.5 fixed)
  * RRF                  (closed-form, no tuning, scale-invariant)
  * CombMNZ z-score      (closed-form, consensus-rewarding)
  * Inverse-variance     (uses EDLAE's closed-form predictive variance)

Linear alpha-blend is included as the existing-method comparator.
RRF and CombMNZ are the canonical rank- and score-fusion baselines;
inverse-variance is the theoretically-novel contribution.

The Hybrid-Score guard (61 GB allocation) requires this experiment to
materialise full pred matrices -- runs on ml-1m only by default.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps

from evaluation.metrics import evaluate_at_ks
from models import EASE, EDLAE, RP3beta
from models.lazy_pred import make_pred

from fusion import (
    reciprocal_rank_fusion, combmnz_zscore,
    inverse_variance_fusion, edlae_dropout_variance, materialise_pred,
)

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    ensure_results_dir,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_columns,
    wilcoxon_blank_columns,
    wilcoxon_paired,
)


class _Wrap:
    def __init__(self, ease_like, pred):
        self.ease = ease_like
        self.pred = pred


def _eval(model_or_wrap, train, test_pos, k, ks):
    if not hasattr(model_or_wrap, 'ease'):
        # Plain EASE / EDLAE
        wrapped = _Wrap(model_or_wrap, model_or_wrap.pred)
    else:
        wrapped = model_or_wrap
    return evaluate_at_ks(wrapped, train, test_pos, ks=ks)


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None, edlae_dropout=0.5,
        rp3_beta=0.6, rp3_topK=200,
        rrf_k=60,
        seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)
    out_path = ensure_results_dir(
        'score_fusion' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        train, test_pos, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)

        # ---- Base models ----------------------------------------------
        print("\n  [base] EASE")
        t0 = time.time()
        ease = EASE(); ease.fit(train, lambda_=ease_lambda)
        res_ease = _eval(ease, train, test_pos, k, ks)
        t_ease = time.time() - t0
        print(f"    EASE NDCG@{k}={res_ease[f'NDCG@{k}']:.4f}  ({t_ease:.1f}s)")

        print("  [base] RP3beta")
        t0 = time.time()
        rp3 = RP3beta()
        W = rp3.fit(ease.X, alpha=1.0, beta=rp3_beta, topK=rp3_topK,
                    implicit=True)
        # pred_rp3 = X @ W (sparse-sparse, then densify per-row via LazyPred)
        pred_rp3 = make_pred(ease.X, W)
        wrapped_rp3 = _Wrap(ease, pred_rp3)
        res_rp3 = evaluate_at_ks(wrapped_rp3, train, test_pos, ks=ks)
        t_rp3 = time.time() - t0
        print(f"    RP3 NDCG@{k}={res_rp3[f'NDCG@{k}']:.4f}  ({t_rp3:.1f}s)")

        print("  [base] EDLAE")
        t0 = time.time()
        edlae = EDLAE(); edlae.fit(train, lambda_=ease_lambda,
                                    dropout=edlae_dropout)
        res_edlae = _eval(edlae, train, test_pos, k, ks)
        t_edlae = time.time() - t0
        print(f"    EDLAE NDCG@{k}={res_edlae[f'NDCG@{k}']:.4f}  ({t_edlae:.1f}s)")

        # Pre-compute base scores for fusion (dense materialisation).
        # On ml-1m this is ~165 MB per matrix; ml-small ~7 MB; OK.
        S_ease  = materialise_pred(ease.pred)
        S_rp3   = materialise_pred(pred_rp3)
        S_edlae = materialise_pred(edlae.pred)

        # Pre-compute EDLAE dropout variance for inverse-variance fusion
        var_edlae = edlae_dropout_variance(ease.X, edlae.B, edlae_dropout)
        # For models without a closed-form variance, use a constant
        # placeholder (1.0 -> equal weight). Slim & adjustable.
        var_const = np.ones_like(S_ease)

        for label, res in [('EASE', res_ease),
                           ('RP3beta', res_rp3),
                           ('EDLAE', res_edlae)]:
            row = {'dataset': dataset, 'seed': seed, 'method': label,
                   'fusion': '-'}
            row.update(metric_cols_at_ks(res, ks, primary_k=k))
            row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
            rows.append(row)

        # ---- Fusion ----------------------------------------------------
        print("\n  [fusion] Linear alpha=0.5  EASE+RP3+EDLAE")
        S_lin = (S_ease + S_rp3 + S_edlae) / 3.0
        wrapped_lin = _Wrap(ease, S_lin)
        res_lin = evaluate_at_ks(wrapped_lin, train, test_pos, ks=ks)
        w_lin = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_lin['per_user_ndcg'][k], res_lin['per_user_ids']))
        print(f"    Linear NDCG@{k}={res_lin[f'NDCG@{k}']:.4f}  "
              f"p_vs_EASE={w_lin[1]:.2e}[{w_lin[3]:+d}]")
        rows.append({'dataset': dataset, 'seed': seed,
                     'method': 'Fusion', 'fusion': 'linear-uniform',
                     **metric_cols_at_ks(res_lin, ks, primary_k=k),
                     **wilcoxon_columns('wilcoxon_vs_EASE', w_lin)})

        print("  [fusion] Reciprocal Rank Fusion (RRF)")
        S_rrf = reciprocal_rank_fusion([S_ease, S_rp3, S_edlae], k=rrf_k)
        wrapped = _Wrap(ease, S_rrf)
        res_rrf = evaluate_at_ks(wrapped, train, test_pos, ks=ks)
        w_rrf = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_rrf['per_user_ndcg'][k], res_rrf['per_user_ids']))
        print(f"    RRF    NDCG@{k}={res_rrf[f'NDCG@{k}']:.4f}  "
              f"p_vs_EASE={w_rrf[1]:.2e}[{w_rrf[3]:+d}]")
        rows.append({'dataset': dataset, 'seed': seed,
                     'method': 'Fusion', 'fusion': f'rrf_k={rrf_k}',
                     **metric_cols_at_ks(res_rrf, ks, primary_k=k),
                     **wilcoxon_columns('wilcoxon_vs_EASE', w_rrf)})

        print("  [fusion] CombMNZ z-score")
        S_cmnz = combmnz_zscore([S_ease, S_rp3, S_edlae])
        wrapped = _Wrap(ease, S_cmnz)
        res_cmnz = evaluate_at_ks(wrapped, train, test_pos, ks=ks)
        w_cmnz = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_cmnz['per_user_ndcg'][k], res_cmnz['per_user_ids']))
        print(f"    CombMNZ NDCG@{k}={res_cmnz[f'NDCG@{k}']:.4f}  "
              f"p_vs_EASE={w_cmnz[1]:.2e}[{w_cmnz[3]:+d}]")
        rows.append({'dataset': dataset, 'seed': seed,
                     'method': 'Fusion', 'fusion': 'combmnz-z',
                     **metric_cols_at_ks(res_cmnz, ks, primary_k=k),
                     **wilcoxon_columns('wilcoxon_vs_EASE', w_cmnz)})

        print("  [fusion] Inverse-variance (Wager-Wang-Liang)")
        S_iv = inverse_variance_fusion(
            [S_ease, S_rp3, S_edlae],
            [var_const, var_const, var_edlae])
        wrapped = _Wrap(ease, S_iv)
        res_iv = evaluate_at_ks(wrapped, train, test_pos, ks=ks)
        w_iv = wilcoxon_paired(
            (res_ease['per_user_ndcg'][k], res_ease['per_user_ids']),
            (res_iv['per_user_ndcg'][k], res_iv['per_user_ids']))
        print(f"    InvVar NDCG@{k}={res_iv[f'NDCG@{k}']:.4f}  "
              f"p_vs_EASE={w_iv[1]:.2e}[{w_iv[3]:+d}]")
        rows.append({'dataset': dataset, 'seed': seed,
                     'method': 'Fusion', 'fusion': f'inv_var_p={edlae_dropout}',
                     **metric_cols_at_ks(res_iv, ks, primary_k=k),
                     **wilcoxon_columns('wilcoxon_vs_EASE', w_iv)})

    df = pd.DataFrame(rows)
    stem = f'score_fusion_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['method', 'fusion']
        metric_cols = [c for c in df.columns if '@' in c
                       or c.startswith('wilcoxon_')]
        agg = {c: ['mean', 'std'] for c in metric_cols if c in df.columns}
        s = df.groupby(key_cols, dropna=False).agg(agg)
        s.columns = [f'{m}_{st}' for m, st in s.columns]
        s.reset_index().to_csv(out_path / f'{stem}_summary.csv', index=False)
        print(f"Saved {out_path / (stem + '_summary.csv')}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-1m',
                   choices=['ml-small', 'ml-1m'],
                   help='netflix-prize not supported (full pred too big)')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--edlae_dropout', type=float, default=0.5)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--rrf_k', type=int, default=60)
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()
    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, edlae_dropout=args.edlae_dropout,
        rp3_beta=args.rp3_beta, rp3_topK=args.topK,
        rrf_k=args.rrf_k, seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
