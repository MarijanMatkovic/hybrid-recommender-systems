"""
Full-baseline sweep — EASE, RP3beta, Hybrid-Score, Hybrid-Matrix,
Hybrid-GraphReg, Laplacian-EASE.

This is the credibility anchor for the thesis: the Laplacian results in
``experiments/`` must be compared against the adjacent methods
(Hybrid-Score and Hybrid-Matrix, Steck's closest published baselines)
and against the ``||B − W||^2`` graph-regularised EASEr. Prior to this
restoration the only place those numbers were referenced was an
unreproducible prose comment at the bottom of the file. Now each family
produces a per-model CSV and (for every Laplacian-style row) a companion
per-bucket CSV, with paired Wilcoxon-vs-EASE tests on every non-EASE row.

Primary evaluation protocol
---------------------------
We run under two protocols and report both:

  * ``--protocol=primary``  (default):   random 80/20 per user × 5 seeds.
    Headline numbers for the thesis.
  * ``--protocol=temporal`` (secondary): Steck-style temporal split,
    single deterministic run. Kept for comparability with prior EASE
    papers. Faster (single fit).

The primary protocol is what the thesis uses for significance tests;
the temporal protocol is reported as a sanity check. Run both with:

    python main.py --dataset ml-1m --protocol primary
    python main.py --dataset ml-1m --protocol temporal

Each invocation writes to ``results/baselines/<protocol>/``.

Under ``--protocol primary`` the Wilcoxon tests and NDCG confidence
intervals are computed across seeds using per-seed result pools (same
pattern as ``experiments/edlae_multiseed.py``); under
``--protocol temporal`` they are the single-split signed-rank test.
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import evaluate_at_ks
from models import EASE, RP3beta, HybridEASE_RP3beta

from experiments._shared import (
    PRIMARY_SPLIT_SEEDS,
    bucket_rows,
    bucketed_metrics_at_ks,
    ensure_results_dir,
    item_popularity_buckets,
    load_dataset,
    metric_cols_at_ks,
    wilcoxon_blank_columns,
    wilcoxon_columns,
    wilcoxon_paired,
    wilcoxon_vs_baseline,
)


class _StandaloneWrapper:
    """Shim so ``evaluate_at_ks`` sees the ``.ease + .pred`` interface it
    expects, even when the caller is handing it a freshly computed
    prediction matrix that isn't wrapped in a Hybrid model."""

    def __init__(self, ease_model, pred_matrix):
        self.ease = ease_model
        self.pred = pred_matrix


# =====================================================================
# Per-family runners (single split -- called once under 'temporal' and
# once per seed under 'primary').
# =====================================================================

def _fit_eval_ease(train, test_positive, lam, ks, k, implicit=True):
    t0 = time.time()
    ease = EASE()
    B, X = ease.fit(train, lambda_=lam, implicit=implicit)
    dt = time.time() - t0
    wrapped = _StandaloneWrapper(ease, ease.pred)
    res = evaluate_at_ks(wrapped, train, test_positive, ks=ks)
    return res, wrapped, dt, B, X


def _fit_eval_rp3(train, test_positive, base_ease, base_X,
                  alpha, beta, topK, ks, k, implicit=True):
    t0 = time.time()
    rp3 = RP3beta()
    W = rp3.fit(base_X, alpha=alpha, beta=beta, topK=topK,
                implicit=implicit)
    pred_rp3 = base_X.dot(W).toarray()
    dt = time.time() - t0
    wrapped = _StandaloneWrapper(base_ease, pred_rp3)
    res = evaluate_at_ks(wrapped, train, test_positive, ks=ks)
    return res, wrapped, dt


def _fit_eval_hybrid(train, test_positive, method, lam, rp3_beta,
                     rp3_topK, fusion_alpha, ks, k,
                     graph_reg_gamma=0.0):
    t0 = time.time()
    h = HybridEASE_RP3beta()
    kw = dict(method=method, fusion_alpha=fusion_alpha,
              ease_lambda=lam, rp3_alpha=1.0,
              rp3_beta=rp3_beta, rp3_topK=rp3_topK)
    if method == 'graph_reg':
        kw['graph_reg_gamma'] = graph_reg_gamma
    h.fit(train, **kw)
    dt = time.time() - t0
    res = evaluate_at_ks(h, train, test_positive, ks=ks)
    return res, h, dt


def _fit_eval_laplacian(train, test_positive, lam, rp3_beta, rp3_topK,
                        gamma, ks, k, normalise='sym',
                        graph_source='rp3beta'):
    t0 = time.time()
    h = HybridEASE_RP3beta()
    h.fit(train, method='laplacian',
          ease_lambda=lam, rp3_alpha=1.0,
          rp3_beta=rp3_beta, rp3_topK=rp3_topK,
          graph_reg_gamma=gamma,
          graph_source=graph_source,
          laplacian_normalise=normalise)
    dt = time.time() - t0
    res = evaluate_at_ks(h, train, test_positive, ks=ks)
    return res, h, dt


# =====================================================================
# Grid definitions -- single source of truth.
# =====================================================================

def _default_grids(dataset):
    """Return the (lambda, beta, topK, etc.) grids per family.

    Smaller grids on ml-small so the ci smoke-run is cheap, wider on
    ml-1m where the full sweep is the thesis contribution.
    """
    if dataset == 'ml-1m':
        return {
            'ease_lambdas':      [100, 200, 500, 1000],
            # RP3beta grid kept modest so the full baseline job still
            # finishes in the overnight slot.
            'rp3_betas':         [0.3, 0.6],
            'rp3_topKs':         [200, 500],
            # Hybrid-Score / Hybrid-Matrix share this fusion grid.
            'hyb_lambdas':       [200, 500],
            'hyb_rp3_betas':     [0.3, 0.6],
            'hyb_rp3_topKs':     [200],
            'hyb_fusion_alphas': [0.3, 0.5, 0.7],
            # GraphReg (||B - W||^2): small gammas since W is scaled to
            # match diag(G).
            'gr_lambdas':        [200, 500],
            'gr_gammas':         [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
            # Laplacian: anchors at the sweet spot (rp3beta-sym γ~30-100).
            'lap_lambdas':       [500],
            'lap_rp3_betas':     [0.3, 0.6],
            'lap_rp3_topKs':     [200],
            'lap_gammas':        [1, 3, 10, 30, 50, 75, 100],
        }
    # ml-small -- small grids for quick smoke runs.
    return {
        'ease_lambdas':      [50, 200, 500],
        'rp3_betas':         [0.3, 0.6],
        'rp3_topKs':         [200],
        'hyb_lambdas':       [200],
        'hyb_rp3_betas':     [0.3, 0.6],
        'hyb_rp3_topKs':     [200],
        'hyb_fusion_alphas': [0.3, 0.5, 0.7],
        'gr_lambdas':        [200],
        'gr_gammas':         [0.01, 0.1, 0.5],
        'lap_lambdas':       [200],
        'lap_rp3_betas':     [0.3, 0.6],
        'lap_rp3_topKs':     [200],
        'lap_gammas':        [1, 3, 10, 30, 75],
    }


# =====================================================================
# Single-split driver -- returns per-family rows (+ companion bucket
# rows). Gets invoked once under 'temporal' and once per seed under
# 'primary'.
# =====================================================================

def _run_single_split(train, test_positive, dataset, k, ks, grids,
                      seed=None):
    """Run one complete baseline sweep on the given train/test split.

    Returns a dict of lists: ``{family: [row, ...]}`` and
    ``{family: [bucket_row, ...]}``. The split is fully fitted for every
    config so each family's rows share a consistent baseline EASE.
    """
    bucket_of = item_popularity_buckets(train, n_buckets=5)
    family_rows = {}   # family -> list of aggregate rows
    family_buckets = {}  # family -> list of bucket rows
    family_per_user = {}  # family -> {config_key -> (ndcg, user_ids)}

    def _log(family, count, cfg, res, dt):
        line = (f"  [{family}:{count:>3d}] "
                f"NDCG={res[f'NDCG@{k}']:.4f} "
                f"Cov={res[f'Coverage@{k}']:.4f} "
                f"Gini={res[f'Gini@{k}']:.4f} "
                f"Nov={res[f'Novelty@{k}']:.1f} "
                f"| {dt:.1f}s "
                f"| {cfg}")
        print(line)

    seed_meta = {'seed': seed} if seed is not None else {}

    # -----------------------------------------------------------------
    # 1) EASE baseline sweep.
    #    The best-NDCG EASE is memorised as the Wilcoxon reference used
    #    by every other family.
    # -----------------------------------------------------------------
    print("\n=== 1/6 EASE ===")
    family_rows['EASE'] = []
    family_buckets['EASE'] = []
    family_per_user['EASE'] = {}
    ease_cache = {}
    best_ease = {'NDCG': -1.0, 'lam': None, 'res': None, 'model': None}
    for i, lam in enumerate(grids['ease_lambdas']):
        res, wrapped, dt, B, X = _fit_eval_ease(
            train, test_positive, lam, ks, k)
        ease_cache[lam] = (wrapped, B, X)
        _log('EASE', i + 1, {'lambda': lam}, res, dt)
        row = {'model': 'EASE', 'lambda': lam,
               'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        # EASE is self-reference -> blank Wilcoxon columns.
        row.update(wilcoxon_blank_columns('wilcoxon_vs_EASE'))
        family_rows['EASE'].append(row)
        buckets = bucketed_metrics_at_ks(
            wrapped, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['EASE'].extend(
            bucket_rows({'model': 'EASE', 'lambda': lam, **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['EASE'][lam] = (
            res['per_user_ndcg'][k], res['per_user_ids'])
        if res[f'NDCG@{k}'] > best_ease['NDCG']:
            best_ease = {
                'NDCG': res[f'NDCG@{k}'], 'lam': lam,
                'res': res, 'model': wrapped}

    print(f"  best EASE NDCG@{k}={best_ease['NDCG']:.4f} "
          f"at lambda={best_ease['lam']}")
    res_best_ease = best_ease['res']

    # -----------------------------------------------------------------
    # 2) RP3beta sweep (pure graph model, no EASE mixing).
    # -----------------------------------------------------------------
    print("\n=== 2/6 RP3beta ===")
    family_rows['RP3beta'] = []
    family_buckets['RP3beta'] = []
    family_per_user['RP3beta'] = {}
    # Re-use the best-lambda EASE to expose .user_enc / .item_enc.
    anchor_ease, _, anchor_X = ease_cache[best_ease['lam']]
    i = 0
    for beta, topK in itertools.product(grids['rp3_betas'],
                                         grids['rp3_topKs']):
        i += 1
        res, wrapped, dt = _fit_eval_rp3(
            train, test_positive, anchor_ease.ease, anchor_X,
            alpha=1.0, beta=beta, topK=topK, ks=ks, k=k)
        _log('RP3', i, {'rp3_beta': beta, 'rp3_topK': topK}, res, dt)
        w = wilcoxon_vs_baseline(res_best_ease, res, k=k)
        row = {'model': 'RP3beta', 'rp3_beta': beta, 'rp3_topK': topK,
               'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
        family_rows['RP3beta'].append(row)
        buckets = bucketed_metrics_at_ks(
            wrapped, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['RP3beta'].extend(
            bucket_rows({'model': 'RP3beta', 'rp3_beta': beta,
                         'rp3_topK': topK, **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['RP3beta'][(beta, topK)] = (
            res['per_user_ndcg'][k], res['per_user_ids'])

    # -----------------------------------------------------------------
    # 3) Hybrid-Score  (score-level α-blend).
    # -----------------------------------------------------------------
    print("\n=== 3/6 Hybrid-Score ===")
    family_rows['Hybrid-Score'] = []
    family_buckets['Hybrid-Score'] = []
    family_per_user['Hybrid-Score'] = {}
    i = 0
    for lam, rp3_b, rp3_tk, fa in itertools.product(
        grids['hyb_lambdas'], grids['hyb_rp3_betas'],
        grids['hyb_rp3_topKs'], grids['hyb_fusion_alphas'],
    ):
        i += 1
        res, h, dt = _fit_eval_hybrid(
            train, test_positive, 'score', lam, rp3_b, rp3_tk, fa,
            ks=ks, k=k)
        _log('HScore', i,
             {'fusion_alpha': fa, 'lambda': lam, 'beta': rp3_b,
              'topK': rp3_tk}, res, dt)
        w = wilcoxon_vs_baseline(res_best_ease, res, k=k)
        row = {'model': 'Hybrid-Score', 'fusion_alpha': fa,
               'ease_lambda': lam, 'rp3_beta': rp3_b,
               'rp3_topK': rp3_tk, 'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
        family_rows['Hybrid-Score'].append(row)
        buckets = bucketed_metrics_at_ks(
            h, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['Hybrid-Score'].extend(
            bucket_rows({'model': 'Hybrid-Score',
                         'fusion_alpha': fa, 'ease_lambda': lam,
                         'rp3_beta': rp3_b, 'rp3_topK': rp3_tk,
                         **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['Hybrid-Score'][(fa, lam, rp3_b, rp3_tk)] = (
            res['per_user_ndcg'][k], res['per_user_ids'])

    # -----------------------------------------------------------------
    # 4) Hybrid-Matrix (item-item matrix-level blend).
    # -----------------------------------------------------------------
    print("\n=== 4/6 Hybrid-Matrix ===")
    family_rows['Hybrid-Matrix'] = []
    family_buckets['Hybrid-Matrix'] = []
    family_per_user['Hybrid-Matrix'] = {}
    i = 0
    for lam, rp3_b, rp3_tk, fa in itertools.product(
        grids['hyb_lambdas'], grids['hyb_rp3_betas'],
        grids['hyb_rp3_topKs'], grids['hyb_fusion_alphas'],
    ):
        i += 1
        res, h, dt = _fit_eval_hybrid(
            train, test_positive, 'matrix', lam, rp3_b, rp3_tk, fa,
            ks=ks, k=k)
        _log('HMatrix', i,
             {'fusion_alpha': fa, 'lambda': lam, 'beta': rp3_b,
              'topK': rp3_tk}, res, dt)
        w = wilcoxon_vs_baseline(res_best_ease, res, k=k)
        row = {'model': 'Hybrid-Matrix', 'fusion_alpha': fa,
               'ease_lambda': lam, 'rp3_beta': rp3_b,
               'rp3_topK': rp3_tk, 'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
        family_rows['Hybrid-Matrix'].append(row)
        buckets = bucketed_metrics_at_ks(
            h, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['Hybrid-Matrix'].extend(
            bucket_rows({'model': 'Hybrid-Matrix',
                         'fusion_alpha': fa, 'ease_lambda': lam,
                         'rp3_beta': rp3_b, 'rp3_topK': rp3_tk,
                         **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['Hybrid-Matrix'][(fa, lam, rp3_b, rp3_tk)] = (
            res['per_user_ndcg'][k], res['per_user_ids'])

    # -----------------------------------------------------------------
    # 5) GraphReg (||B - W||^2).
    # -----------------------------------------------------------------
    print("\n=== 5/6 Hybrid-GraphReg ===")
    family_rows['Hybrid-GraphReg'] = []
    family_buckets['Hybrid-GraphReg'] = []
    family_per_user['Hybrid-GraphReg'] = {}
    i = 0
    for lam, gamma in itertools.product(grids['gr_lambdas'],
                                         grids['gr_gammas']):
        i += 1
        res, h, dt = _fit_eval_hybrid(
            train, test_positive, 'graph_reg', lam, 0.6, 200, 0.5,
            ks=ks, k=k, graph_reg_gamma=gamma)
        _log('GraphReg', i,
             {'lambda': lam, 'gamma': gamma}, res, dt)
        w = wilcoxon_vs_baseline(res_best_ease, res, k=k)
        row = {'model': 'Hybrid-GraphReg',
               'ease_lambda': lam, 'rp3_beta': 0.6, 'rp3_topK': 200,
               'gamma': gamma, 'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
        family_rows['Hybrid-GraphReg'].append(row)
        buckets = bucketed_metrics_at_ks(
            h, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['Hybrid-GraphReg'].extend(
            bucket_rows({'model': 'Hybrid-GraphReg',
                         'ease_lambda': lam, 'rp3_beta': 0.6,
                         'rp3_topK': 200, 'gamma': gamma,
                         **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['Hybrid-GraphReg'][(lam, gamma)] = (
            res['per_user_ndcg'][k], res['per_user_ids'])

    # -----------------------------------------------------------------
    # 6) Laplacian-EASE (tr(B^T L B) regulariser, sym Laplacian).
    # -----------------------------------------------------------------
    print("\n=== 6/6 Laplacian-EASE (sym, rp3beta graph) ===")
    family_rows['Laplacian-EASE'] = []
    family_buckets['Laplacian-EASE'] = []
    family_per_user['Laplacian-EASE'] = {}
    i = 0
    for lam, rp3_b, rp3_tk, gamma in itertools.product(
        grids['lap_lambdas'], grids['lap_rp3_betas'],
        grids['lap_rp3_topKs'], grids['lap_gammas'],
    ):
        i += 1
        res, h, dt = _fit_eval_laplacian(
            train, test_positive, lam, rp3_b, rp3_tk, gamma, ks, k,
            normalise='sym', graph_source='rp3beta')
        _log('Lap', i,
             {'lambda': lam, 'beta': rp3_b, 'topK': rp3_tk,
              'gamma': gamma}, res, dt)
        w = wilcoxon_vs_baseline(res_best_ease, res, k=k)
        row = {'model': 'Laplacian-EASE',
               'ease_lambda': lam, 'rp3_beta': rp3_b,
               'rp3_topK': rp3_tk, 'gamma': gamma,
               'normalise': 'sym', 'graph_source': 'rp3beta',
               'train_time_s': dt, **seed_meta}
        row.update(metric_cols_at_ks(res, ks, primary_k=k))
        row.update(wilcoxon_columns('wilcoxon_vs_EASE', w))
        family_rows['Laplacian-EASE'].append(row)
        buckets = bucketed_metrics_at_ks(
            h, train, test_positive, ks=ks, bucket_of=bucket_of)
        family_buckets['Laplacian-EASE'].extend(
            bucket_rows({'model': 'Laplacian-EASE',
                         'ease_lambda': lam, 'rp3_beta': rp3_b,
                         'rp3_topK': rp3_tk, 'gamma': gamma,
                         'normalise': 'sym',
                         'graph_source': 'rp3beta',
                         **seed_meta},
                        buckets, ks, n_buckets=5))
        family_per_user['Laplacian-EASE'][
            (lam, rp3_b, rp3_tk, gamma)] = (
            res['per_user_ndcg'][k], res['per_user_ids'])

    return family_rows, family_buckets, family_per_user, res_best_ease


def _summarise_over_seeds(rows):
    """For the primary protocol: aggregate mean/std/count per config
    over the per-seed rows.

    Config / hyperparam columns are the *grouping* keys -- this is how
    we collapse 5 seeds of ``lambda=200`` down to one row. Metric columns
    (anything with ``@`` like ``NDCG@10``) and per-seed stats
    (``train_time_s``, ``wilcoxon_*``) are aggregated. A naive
    numeric/non-numeric heuristic would be wrong because numeric
    hyperparams (``ease_lambda``, ``fusion_alpha``, ``gamma``...) are
    floats, so they'd all be aggregated and every config in a family
    would collapse into a single group.
    """
    df = pd.DataFrame(rows)
    if df.empty or 'seed' not in df.columns:
        return df
    metric_cols = [c for c in df.columns
                   if '@' in c
                   or c == 'train_time_s'
                   or c.startswith('wilcoxon_')]
    key_cols = [c for c in df.columns
                if c not in metric_cols and c != 'seed']
    if not key_cols or not metric_cols:
        return df
    agg = {m: ['mean', 'std', 'count'] for m in metric_cols}
    out = df.groupby(key_cols, dropna=False).agg(agg)
    out.columns = [f'{m}_{stat}' for m, stat in out.columns]
    return out.reset_index()


def _summarise_buckets_over_seeds(rows, ks):
    """Same as ``_summarise_over_seeds`` but for the per-bucket CSV,
    aggregating within (config, bucket)."""
    df = pd.DataFrame(rows)
    if df.empty or 'seed' not in df.columns:
        return df
    metric_cols = [f'{m}@{k}' for k in ks
                   for m in ('NDCG', 'Recall', 'HitRate')]
    metric_cols += ['n_users']
    metric_cols = [c for c in metric_cols if c in df.columns]
    key_cols = [c for c in df.columns
                if c not in metric_cols and c != 'seed']
    agg = {m: ['mean', 'std', 'count'] for m in metric_cols}
    out = df.groupby(key_cols, dropna=False).agg(agg)
    out.columns = [f'{m}_{stat}' for m, stat in out.columns]
    return out.reset_index()


# =====================================================================
# Orchestration
# =====================================================================

def _run(dataset, k, protocol, out_root, seeds=None):
    ks = (k, max(k, 20))
    grids = _default_grids(dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_results_dir(f'baselines/{protocol}',
                                 root=out_root)

    all_family_rows = {}     # family -> list across seeds
    all_family_buckets = {}  # family -> list across seeds
    per_seed_per_user = {}   # seed -> {family -> {config_key -> (ndcg, ids)}}

    if protocol == 'temporal':
        print(f"\nTemporal split (single deterministic run) on {dataset}")
        train, test_pos, _ = load_dataset(dataset, split_mode='temporal')
        frs, fbs, fpu, _ = _run_single_split(
            train, test_pos, dataset, k, ks, grids, seed=None)
        for fam, rows in frs.items():
            all_family_rows[fam] = rows
            all_family_buckets[fam] = fbs[fam]
        seeds_used = []
    else:
        seeds_used = (list(seeds) if seeds is not None
                      else list(PRIMARY_SPLIT_SEEDS))
        print(f"\nPrimary protocol: random 80/20 × {len(seeds_used)} "
              f"seeds on {dataset} (seeds={seeds_used})")
        for seed in seeds_used:
            print(f"\n{'#' * 60}\n# seed = {seed}\n{'#' * 60}")
            train, test_pos, _ = load_dataset(
                dataset, split_mode='random', split_seed=seed)
            frs, fbs, fpu, _ = _run_single_split(
                train, test_pos, dataset, k, ks, grids, seed=seed)
            for fam, rows in frs.items():
                all_family_rows.setdefault(fam, []).extend(rows)
                all_family_buckets.setdefault(fam, []).extend(fbs[fam])
            per_seed_per_user[seed] = fpu

    # -----------------------------------------------------------------
    # Write per-family CSVs. Under 'primary' we write both the raw
    # per-seed rows and a seed-aggregated summary.
    # -----------------------------------------------------------------
    for fam, rows in all_family_rows.items():
        fam_safe = fam.lower().replace('-', '_').replace(' ', '_')
        stem = f'{fam_safe}_{dataset}_{protocol}_{timestamp}'
        path = out_dir / f'{stem}.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  wrote {path}")

        if all_family_buckets.get(fam):
            bpath = out_dir / f'{stem}_buckets.csv'
            pd.DataFrame(all_family_buckets[fam]).to_csv(
                bpath, index=False)
            print(f"  wrote {bpath}")

        if protocol == 'primary':
            summary = _summarise_over_seeds(rows)
            spath = out_dir / f'{stem}_summary.csv'
            summary.to_csv(spath, index=False)
            print(f"  wrote {spath}")
            bsummary = _summarise_buckets_over_seeds(
                all_family_buckets[fam], ks)
            bspath = out_dir / f'{stem}_buckets_summary.csv'
            bsummary.to_csv(bspath, index=False)
            print(f"  wrote {bspath}")

    # -----------------------------------------------------------------
    # Cross-family leaderboard + primary-protocol pooled Wilcoxon
    # (for every non-EASE family's best row, test pooled-per-user NDCG
    # against the pooled EASE best row).
    # -----------------------------------------------------------------
    leaderboard = []
    for fam, rows in all_family_rows.items():
        df = pd.DataFrame(rows)
        if df.empty or f'NDCG@{k}' not in df.columns:
            continue
        if protocol == 'primary':
            # Find the config with the highest *mean* NDCG across seeds,
            # then report mean ± std. Grouping keys = everything that's
            # not a per-seed metric.
            df_summary = _summarise_over_seeds(rows)
            if df_summary.empty or f'NDCG@{k}_mean' not in df_summary.columns:
                continue
            best = df_summary.sort_values(
                f'NDCG@{k}_mean', ascending=False).iloc[0]
            leaderboard.append({
                'family': fam,
                f'NDCG@{k}_mean': best[f'NDCG@{k}_mean'],
                f'NDCG@{k}_std':  best.get(f'NDCG@{k}_std', np.nan),
                f'Coverage@{k}_mean': best.get(f'Coverage@{k}_mean', np.nan),
                f'Gini@{k}_mean':     best.get(f'Gini@{k}_mean', np.nan),
                f'Novelty@{k}_mean':  best.get(f'Novelty@{k}_mean', np.nan),
                'seeds': len(seeds_used),
            })
        else:
            best = df.sort_values(f'NDCG@{k}',
                                  ascending=False).iloc[0]
            leaderboard.append({
                'family': fam,
                f'NDCG@{k}': best[f'NDCG@{k}'],
                f'Coverage@{k}': best.get(f'Coverage@{k}', np.nan),
                f'Gini@{k}':     best.get(f'Gini@{k}', np.nan),
                f'Novelty@{k}':  best.get(f'Novelty@{k}', np.nan),
                'seeds': 1,
            })
    lb_df = pd.DataFrame(leaderboard)
    lb_path = out_dir / f'leaderboard_{dataset}_{protocol}_{timestamp}.csv'
    lb_df.to_csv(lb_path, index=False)
    print(f"\nLeaderboard written to {lb_path}:")
    try:
        with pd.option_context('display.max_columns', None,
                               'display.width', 160):
            print(lb_df.to_string(index=False))
    except Exception:  # pragma: no cover
        print(lb_df.to_string(index=False))

    # -----------------------------------------------------------------
    # Primary-protocol pooled Wilcoxon: for each non-EASE family take
    # the best-mean-NDCG config and test its pooled per-user NDCG
    # against the pooled EASE per-user NDCG (users prefixed by seed).
    # -----------------------------------------------------------------
    if protocol == 'primary' and per_seed_per_user:
        print("\nPooled-Wilcoxon (best-of-family vs best-of-EASE) "
              "across seeds:")
        # Best EASE config first.
        ease_df = pd.DataFrame(all_family_rows.get('EASE', []))
        if not ease_df.empty:
            ease_summary = _summarise_over_seeds(
                all_family_rows['EASE'])
            ease_best_lam = ease_summary.sort_values(
                f'NDCG@{k}_mean', ascending=False).iloc[0]['lambda']
            ease_pool_vals, ease_pool_ids = [], []
            for seed in seeds_used:
                vals, ids = per_seed_per_user[seed]['EASE'][
                    ease_best_lam]
                ease_pool_vals.append(np.asarray(vals, dtype=float))
                ease_pool_ids.extend((seed, u) for u in ids)
            ease_pool = (np.concatenate(ease_pool_vals),
                         ease_pool_ids)

            wilcoxon_rows = []
            for fam, rows in all_family_rows.items():
                if fam == 'EASE' or not rows:
                    continue
                summary = _summarise_over_seeds(rows)
                if f'NDCG@{k}_mean' not in summary.columns:
                    continue
                best = summary.sort_values(
                    f'NDCG@{k}_mean', ascending=False).iloc[0]
                # Reconstruct the config key used in per_seed_per_user.
                # For every non-EASE family the config key is the tuple
                # (p1, p2, ...) of the per-family hyper-params. Rather
                # than hard-code, find the matching seed entry by
                # comparing NDCG values (robust across family-specific
                # key shapes).
                target_ndcg = best[f'NDCG@{k}_mean']
                pool_vals, pool_ids = [], []
                matched = 0
                for seed in seeds_used:
                    by_cfg = per_seed_per_user[seed].get(fam, {})
                    # Find the config whose mean across seeds equals the
                    # summary's best NDCG (within tol). Simple approach:
                    # match by the per-seed NDCG row. We iterate all
                    # rows for this (fam, seed) to find the closest.
                    seed_rows = [r for r in rows
                                 if r.get('seed') == seed]
                    chosen_cfg = None
                    chosen_row = None
                    best_delta = np.inf
                    for r in seed_rows:
                        # Build the config tuple the same way the
                        # family stores it in per_user_store.
                        cfg = _cfg_key_for_family(fam, r)
                        if cfg not in by_cfg:
                            continue
                        # Select the row whose summary-key values match
                        # ``best``. That's easier via the key columns
                        # we already stored; compare them directly.
                        if _row_matches_summary(r, best):
                            chosen_cfg = cfg
                            chosen_row = r
                            break
                    if chosen_cfg is not None:
                        vals, ids = by_cfg[chosen_cfg]
                        pool_vals.append(np.asarray(vals, dtype=float))
                        pool_ids.extend((seed, u) for u in ids)
                        matched += 1
                if matched == len(seeds_used) and pool_vals:
                    pool = (np.concatenate(pool_vals), pool_ids)
                    w = wilcoxon_paired(ease_pool, pool)
                    wilcoxon_rows.append({
                        'family': fam,
                        'wilcoxon_vs_EASE_stat':         w[0],
                        'wilcoxon_vs_EASE_p':            w[1],
                        'wilcoxon_vs_EASE_n_pairs':      w[2],
                        'wilcoxon_vs_EASE_sign':         w[3],
                        'wilcoxon_vs_EASE_mean_diff':    w[4],
                        'wilcoxon_vs_EASE_median_diff':  w[5],
                    })
                    sign_str = (f"{w[3]:+d}" if w[3] != 0 else ' 0')
                    print(f"  {fam:<18s} "
                          f"p={w[1]:.2e}[{sign_str}]  "
                          f"mean_diff={w[4]:+.4f}")
            if wilcoxon_rows:
                wpath = (out_dir /
                         f'wilcoxon_{dataset}_{protocol}_{timestamp}.csv')
                pd.DataFrame(wilcoxon_rows).to_csv(wpath, index=False)
                print(f"  wrote {wpath}")

    return out_dir


def _cfg_key_for_family(fam, row):
    """Reconstruct the per_user_store key that each family used when
    it stashed per-user NDCG during a single split.

    The keys match the ones used in ``_run_single_split`` above."""
    if fam == 'EASE':
        return row.get('lambda')
    if fam == 'RP3beta':
        return (row.get('rp3_beta'), row.get('rp3_topK'))
    if fam in ('Hybrid-Score', 'Hybrid-Matrix'):
        return (row.get('fusion_alpha'), row.get('ease_lambda'),
                row.get('rp3_beta'), row.get('rp3_topK'))
    if fam == 'Hybrid-GraphReg':
        return (row.get('ease_lambda'), row.get('gamma'))
    if fam == 'Laplacian-EASE':
        return (row.get('ease_lambda'), row.get('rp3_beta'),
                row.get('rp3_topK'), row.get('gamma'))
    return None


def _row_matches_summary(row, summary_row):
    """Return True iff the per-seed row's key columns match the
    summary row. Keys = all non-metric, non-seed columns."""
    skip = {'seed', 'train_time_s'}
    for key, val in summary_row.items():
        if key.endswith('_mean') or key.endswith('_std') or \
           key.endswith('_count'):
            continue
        if key in skip:
            continue
        if key in row:
            lhs = row[key]
            if pd.isna(lhs) and pd.isna(val):
                continue
            if lhs != val:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Full-baseline sweep (EASE, RP3beta, Hybrid-*, '
                    'GraphReg, Laplacian-EASE)')
    parser.add_argument('--dataset', default='ml-1m',
                        choices=['ml-small', 'ml-1m'])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--protocol', default='primary',
                        choices=['primary', 'temporal'],
                        help='primary: random 80/20 × 5 seeds (default); '
                             'temporal: single deterministic temporal split.')
    parser.add_argument('--out_root', default='results')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated list of split seeds '
                             'to use with --protocol primary '
                             '(default: 0,1,2,3,4). Ignored for '
                             'temporal protocol. Use --seeds 0,1 for '
                             'a quick smoke-run.')
    args = parser.parse_args()
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')
                 if s.strip()]
    _run(args.dataset, args.k, args.protocol, Path(args.out_root),
         seeds=seeds)


if __name__ == '__main__':
    main()
