"""
Graph-Enhanced EASEr: Focused Laplacian Sweep
===============================================
~38 experiments, estimated ~2 hours on ML-1M.

Usage:
  python main.py --dataset ml-1m --k 10
"""

import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from data import load_movielens, temporal_train_test_split, load_movielens_1m
from models import EASE, RP3beta, HybridEASE_RP3beta
from evaluation.metrics import evaluate, print_results


class StandaloneWrapper:
    def __init__(self, ease_model, pred_matrix):
        self.ease = ease_model
        self.pred = pred_matrix


def run_full_sweep(dataset='ml-1m', k=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_global = time.time()

    # ── Load Data ─────────────────────────────────────────
    if dataset == 'ml-1m':
        ratings, movies = load_movielens_1m('data/ml-1m', min_interactions=5)
        threshold = 4.0
    else:
        ratings, movies = load_movielens('data/ml-latest-small', min_interactions=5)
        threshold = 3.5

    train, test = temporal_train_test_split(ratings, test_ratio=0.2)
    test_positive = test[test['rating'] >= threshold].copy()
    print(f"Dataset: {dataset}")
    print(f"Train: {len(train)} | Test positive (>={threshold}): {len(test_positive)}")
    print(f"Users: {train['user_id'].nunique()} | Items: {train['item_id'].nunique()}\n")

    all_results = []
    experiment_count = 0

    METRIC_COLS = ['Precision@k', 'Recall@k', 'NDCG@k', 'HitRate@k',
                   'MAP@k', 'MRR@k', 'Coverage', 'Gini', 'Novelty']

    def log_result(model_name, params, results, train_time):
        nonlocal experiment_count
        experiment_count += 1
        row = {
            'model': model_name,
            'params': str(params),
            **params,
            **{m: results[m] for m in METRIC_COLS},
            'n_users': results['n_users_evaluated'],
            'train_time': train_time,
        }
        all_results.append(row)
        elapsed = time.time() - t_global
        print(f"  [{experiment_count:>3d}] {model_name:<18s} | "
              f"NDCG={results['NDCG@k']:.4f} "
              f"MAP={results['MAP@k']:.4f} "
              f"HR={results['HitRate@k']:.4f} "
              f"MRR={results['MRR@k']:.4f} "
              f"Cov={results['Coverage']:.4f} "
              f"Gini={results['Gini']:.4f} "
              f"Nov={results['Novelty']:.1f} | "
              f"{train_time:.1f}s | {elapsed/60:.0f}min")

    '''
    # ==========================================================
    # 1/6  EASE  (10 experiments)
    # ==========================================================
    print("=" * 75)
    print("  1/6  EASE")
    print("=" * 75)

    ease_lambdas = [1, 5, 10, 25, 50, 100, 200, 500, 750, 1000]
    ease_cache = {}

    for lam in ease_lambdas:
        t0 = time.time()
        ease = EASE()
        B, X = ease.fit(train, lambda_=lam, implicit=True)
        t = time.time() - t0
        ease_cache[lam] = (ease, B, X)
        wrapped = StandaloneWrapper(ease, ease.pred)
        results = evaluate(wrapped, train, test_positive, k=k)
        log_result('EASE', {'lambda': lam}, results, t)

    # ==========================================================
    # 2/6  RP3beta  (28 experiments)
    # ==========================================================
    print("\n" + "=" * 75)
    print("  2/6  RP3beta")
    print("=" * 75)

    base_ease, _, base_X = ease_cache[200]

    rp3_betas = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rp3_topKs = [100, 200, 300, 500]

    for beta, topK in itertools.product(rp3_betas, rp3_topKs):
        t0 = time.time()
        rp3 = RP3beta()
        W = rp3.fit(base_X, alpha=1.0, beta=beta, topK=topK, implicit=True)
        pred_rp3 = base_X.dot(W).toarray()
        t = time.time() - t0
        wrapped = StandaloneWrapper(base_ease, pred_rp3)
        results = evaluate(wrapped, train, test_positive, k=k)
        log_result('RP3beta', {
            'rp3_beta': beta, 'rp3_topK': topK
        }, results, t)

    # ==========================================================
    # 3/6  Hybrid-Score  (72 experiments)
    # ==========================================================
    print("\n" + "=" * 75)
    print("  3/6  Hybrid-Score")
    print("=" * 75)

    h_lambdas = [100, 200, 500]
    h_rp3_betas = [0.3, 0.6]
    h_rp3_topKs = [200, 500]
    h_fusion_alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for lam, rp3_b, rp3_tk, f_alpha in itertools.product(
        h_lambdas, h_rp3_betas, h_rp3_topKs, h_fusion_alphas
    ):
        t0 = time.time()
        h = HybridEASE_RP3beta()
        h.fit(train, method='score', fusion_alpha=f_alpha,
              ease_lambda=lam, rp3_alpha=1.0, rp3_beta=rp3_b,
              rp3_topK=rp3_tk)
        t = time.time() - t0
        results = evaluate(h, train, test_positive, k=k)
        log_result('Hybrid-Score', {
            'fusion_alpha': f_alpha, 'ease_lambda': lam,
            'rp3_beta': rp3_b, 'rp3_topK': rp3_tk,
        }, results, t)

    # ==========================================================
    # 4/6  Hybrid-Matrix  (72 experiments)
    # ==========================================================
    print("\n" + "=" * 75)
    print("  4/6  Hybrid-Matrix")
    print("=" * 75)

    for lam, rp3_b, rp3_tk, f_alpha in itertools.product(
        h_lambdas, h_rp3_betas, h_rp3_topKs, h_fusion_alphas
    ):
        t0 = time.time()
        h = HybridEASE_RP3beta()
        h.fit(train, method='matrix', fusion_alpha=f_alpha,
              ease_lambda=lam, rp3_alpha=1.0, rp3_beta=rp3_b,
              rp3_topK=rp3_tk)
        t = time.time() - t0
        results = evaluate(h, train, test_positive, k=k)
        log_result('Hybrid-Matrix', {
            'fusion_alpha': f_alpha, 'ease_lambda': lam,
            'rp3_beta': rp3_b, 'rp3_topK': rp3_tk,
        }, results, t)

    # ==========================================================
    # 5/6  GraphReg  (15 experiments)
    # ==========================================================
    print("\n" + "=" * 75)
    print("  5/6  Graph-Regularized EASE (||B-W||^2)")
    print("=" * 75)

    gr_lambdas = [100, 200, 500]
    gr_gammas = [0.01, 0.05, 0.1, 0.3, 0.5]

    for lam, gamma in itertools.product(gr_lambdas, gr_gammas):
        t0 = time.time()
        h = HybridEASE_RP3beta()
        h.fit(train, method='graph_reg', ease_lambda=lam,
              rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=200,
              graph_reg_gamma=gamma)
        t = time.time() - t0
        results = evaluate(h, train, test_positive, k=k)
        log_result('Hybrid-GraphReg', {
            'gamma': gamma, 'ease_lambda': lam,
            'rp3_beta': 0.6, 'rp3_topK': 200,
        }, results, t)
    '''

    # ==========================================================
    # 6/6  Laplacian -- FOCUSED SWEEP  (~38 experiments)
    #
    # Key insight from overnight run:
    #   - Matrix hybrid best was NDCG=0.1238 with beta=0.3
    #   - Previous Laplacian only tested beta=0.6
    #   - beta=0.3 untested and likely better
    #   - gamma still climbing at 50, need higher values
    #   - lambda barely matters at high gamma
    # ==========================================================
    print("=" * 75)
    print("  LAPLACIAN-REGULARIZED EASE -- FOCUSED SWEEP")
    print("=" * 75)

    # Core grid: 2 lambda x 2 beta x 8 gamma = 32 experiments
    lap_lambdas = [100, 200]
    lap_rp3_betas = [0.3, 0.6]
    lap_gammas = [20, 50, 75, 100, 150, 200, 300, 500]

    for lam, rp3_b, gamma in itertools.product(
        lap_lambdas, lap_rp3_betas, lap_gammas
    ):
        t0 = time.time()
        h = HybridEASE_RP3beta()
        h.fit(train, method='laplacian', ease_lambda=lam,
              rp3_alpha=1.0, rp3_beta=rp3_b, rp3_topK=200,
              graph_reg_gamma=gamma)
        t = time.time() - t0
        results = evaluate(h, train, test_positive, k=k)
        log_result('Hybrid-Laplacian', {
            'gamma': gamma, 'ease_lambda': lam,
            'rp3_beta': rp3_b, 'rp3_topK': 200,
        }, results, t)

    # Extra: topK=300 with best beta candidates (6 experiments)
    print("\n  -- topK=300 variants --")
    for rp3_b, gamma in itertools.product(
        [0.3, 0.6], [50, 100, 200]
    ):
        t0 = time.time()
        h = HybridEASE_RP3beta()
        h.fit(train, method='laplacian', ease_lambda=100,
              rp3_alpha=1.0, rp3_beta=rp3_b, rp3_topK=300,
              graph_reg_gamma=gamma)
        t = time.time() - t0
        results = evaluate(h, train, test_positive, k=k)
        log_result('Hybrid-Laplacian', {
            'gamma': gamma, 'ease_lambda': 100,
            'rp3_beta': rp3_b, 'rp3_topK': 300,
        }, results, t)

    # ==========================================================
    # Save & Summary
    # ==========================================================
    df = pd.DataFrame(all_results)
    outfile = f'results_laplacian_{dataset}_{timestamp}.csv'
    df.to_csv(outfile, index=False)

    total_time = time.time() - t_global

    print("\n" + "=" * 75)
    print(f"  DONE -- {len(all_results)} experiments in {total_time/60:.1f} minutes")
    print(f"  Results saved to: {outfile}")
    print("=" * 75)

    # Top results
    show = ['params', 'NDCG@k', 'MAP@k', 'HitRate@k', 'MRR@k',
            'Coverage', 'Gini', 'Novelty']

    print("\n  Top 10 by NDCG:")
    print(df.sort_values('NDCG@k', ascending=False)[show].head(10).to_string(index=False))

    print("\n  Top 5 by MAP:")
    print(df.sort_values('MAP@k', ascending=False)[show].head(5).to_string(index=False))

    print("\n  Top 5 by HitRate:")
    print(df.sort_values('HitRate@k', ascending=False)[show].head(5).to_string(index=False))

    # Reference points to beat
    print("\n" + "=" * 75)
    print("  REFERENCE (from overnight run):")
    print("  Best EASE:          NDCG=0.1163 (lambda=1000)")
    print("  Best RP3beta:       NDCG=0.1163 (beta=0.6, topK=300)")
    print("  Best Score hybrid:  NDCG=0.1227 (lambda=200, beta=0.3, topK=200, alpha=0.3)")
    print("  Best Matrix hybrid: NDCG=0.1238 (lambda=200, beta=0.3, topK=200, alpha=0.8)")
    print("  Previous Laplacian: NDCG=0.1200 (lambda=100, beta=0.6, topK=200, gamma=50)")
    print("=" * 75)

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Laplacian Focused Sweep')
    parser.add_argument('--dataset', default='ml-1m', choices=['ml-small', 'ml-1m'])
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    run_full_sweep(dataset=args.dataset, k=args.k)