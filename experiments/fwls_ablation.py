"""
Feature-Weighted Linear Stacking ablation (roadmap item 16).

Per-user adaptive ensemble of {EASE, RP3beta, EDLAE} where the per-
model weight is a linear function of user meta-features (log history
length + bias). Closed-form ridge in the augmented feature space.

For FWLS validation we hold out a fraction of training interactions
(default 10%) as the supervision signal; the base models are fit on
the remaining 90%. This avoids the test-leak common to "fit FWLS on
test" demos.
"""

from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models import EASE, EDLAE, RP3beta
from models.lazy_pred import make_pred
from fusion import (
    user_meta_features, fit_fwls, apply_fwls, sample_negatives,
    materialise_pred,
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
    def __init__(self, ease, pred):
        self.ease = ease; self.pred = pred


def _holdout_train(train_df, frac=0.1, seed=42):
    """Split train_df into (model_train, fwls_val)."""
    rng = np.random.default_rng(seed)
    n = len(train_df)
    n_val = int(n * frac)
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return (train_df.iloc[train_idx].reset_index(drop=True),
            train_df.iloc[val_idx].reset_index(drop=True))


def run(dataset='ml-1m', k=10, ks=(10, 20),
        ease_lambda=None, edlae_dropout=0.5, rp3_beta=0.6, rp3_topK=200,
        fwls_lambdas=(0.1, 1.0, 10.0), fwls_neg_per_pos=5,
        fwls_val_frac=0.1,
        seeds=None, out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if seeds is None:
        seeds = list(PRIMARY_SPLIT_SEEDS)
    out_path = ensure_results_dir('fwls' if out_dir is None else out_dir)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\n[seed={seed}]\n{'='*60}")
        train_full, test_pos, _ = load_dataset(
            dataset, split_mode='random', split_seed=seed)
        train_main, train_val = _holdout_train(
            train_full, frac=fwls_val_frac, seed=seed)
        print(f"  Train split for base models: {len(train_main):,} "
              f"interactions; FWLS-val: {len(train_val):,}")

        # Fit base models on the "main" portion
        print("  [base] EASE")
        t0 = time.time()
        ease = EASE(); ease.fit(train_main, lambda_=ease_lambda)
        t_e = time.time() - t0
        res_e = evaluate_at_ks(_Wrap(ease, ease.pred), train_full, test_pos, ks=ks)
        print(f"    EASE NDCG@{k}={res_e[f'NDCG@{k}']:.4f}  ({t_e:.1f}s)")

        print("  [base] RP3beta")
        t0 = time.time()
        rp3 = RP3beta()
        W = rp3.fit(ease.X, alpha=1.0, beta=rp3_beta, topK=rp3_topK,
                    implicit=True)
        pred_rp3 = make_pred(ease.X, W)
        res_rp3 = evaluate_at_ks(_Wrap(ease, pred_rp3), train_full, test_pos, ks=ks)
        t_rp3 = time.time() - t0
        print(f"    RP3  NDCG@{k}={res_rp3[f'NDCG@{k}']:.4f}  ({t_rp3:.1f}s)")

        print("  [base] EDLAE")
        t0 = time.time()
        edlae = EDLAE(); edlae.fit(train_main, lambda_=ease_lambda,
                                    dropout=edlae_dropout)
        res_ed = evaluate_at_ks(_Wrap(edlae, edlae.pred),
                                 train_full, test_pos, ks=ks)
        t_ed = time.time() - t0
        print(f"    EDLAE NDCG@{k}={res_ed[f'NDCG@{k}']:.4f}  ({t_ed:.1f}s)")

        # Dense materialise base scores for fusion
        S_e  = materialise_pred(ease.pred)
        S_rp = materialise_pred(pred_rp3)
        S_ed = materialise_pred(edlae.pred)

        phi = user_meta_features(ease.X)

        # Record base results
        for label, res in [('EASE', res_e), ('RP3beta', res_rp3),
                           ('EDLAE', res_ed)]:
            rows.append({
                'dataset': dataset, 'seed': seed, 'method': label,
                'fwls_lambda': 0.0,
                **metric_cols_at_ks(res, ks, primary_k=k),
                **wilcoxon_blank_columns('wilcoxon_vs_EASE'),
            })

        # Build (user, item) pairs from the FWLS holdout
        u_enc = ease.user_enc
        i_enc = ease.item_enc
        train_val = train_val[
            train_val['user_id'].isin(u_enc.classes_) &
            train_val['item_id'].isin(i_enc.classes_)
        ]
        u_idx = u_enc.transform(train_val['user_id'].to_numpy())
        i_idx = i_enc.transform(train_val['item_id'].to_numpy())
        pos_pairs = list(zip(u_idx, i_idx))
        neg_pairs = sample_negatives(
            pos_pairs, n_items=S_e.shape[1],
            n_per_pos=fwls_neg_per_pos, seed=seed)

        print(f"  FWLS val: {len(pos_pairs):,} positives, "
              f"{len(neg_pairs):,} negatives")

        for lam in fwls_lambdas:
            t0 = time.time()
            w = fit_fwls([S_e, S_rp, S_ed], phi, pos_pairs, neg_pairs,
                         lambda_=lam)
            S_fused = apply_fwls([S_e, S_rp, S_ed], phi, w)
            res_f = evaluate_at_ks(_Wrap(ease, S_fused), train_full,
                                    test_pos, ks=ks)
            dt = time.time() - t0
            wstat = wilcoxon_paired(
                (res_e['per_user_ndcg'][k], res_e['per_user_ids']),
                (res_f['per_user_ndcg'][k], res_f['per_user_ids']))
            print(f"    FWLS lambda={lam:<5g}  NDCG@{k}={res_f[f'NDCG@{k}']:.4f}  "
                  f"p_vs_EASE={wstat[1]:.2e}[{wstat[3]:+d}]  ({dt:.1f}s)")
            rows.append({
                'dataset': dataset, 'seed': seed, 'method': 'FWLS',
                'fwls_lambda': lam,
                **metric_cols_at_ks(res_f, ks, primary_k=k),
                **wilcoxon_columns('wilcoxon_vs_EASE', wstat),
            })

    df = pd.DataFrame(rows)
    stem = f'fwls_{dataset}'
    df.to_csv(out_path / f'{stem}.csv', index=False)
    print(f"\nSaved {out_path / (stem + '.csv')}")

    if df['seed'].nunique() > 1:
        key_cols = ['method', 'fwls_lambda']
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
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--edlae_dropout', type=float, default=0.5)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--topK', type=int, default=200)
    p.add_argument('--fwls_lambdas', type=str, default='0.1,1.0,10.0')
    p.add_argument('--fwls_neg_per_pos', type=int, default=5)
    p.add_argument('--fwls_val_frac', type=float, default=0.1)
    p.add_argument('--seeds', type=str, default=None)
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()

    seeds = ([int(x) for x in args.seeds.split(',') if x.strip()]
             if args.seeds else None)
    fwls_lambdas = [float(x) for x in args.fwls_lambdas.split(',')]
    run(dataset=args.dataset, k=args.k, ks=(args.k, max(args.k, 20)),
        ease_lambda=args.ease_lambda, edlae_dropout=args.edlae_dropout,
        rp3_beta=args.rp3_beta, rp3_topK=args.topK,
        fwls_lambdas=fwls_lambdas, fwls_neg_per_pos=args.fwls_neg_per_pos,
        fwls_val_frac=args.fwls_val_frac,
        seeds=seeds, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
