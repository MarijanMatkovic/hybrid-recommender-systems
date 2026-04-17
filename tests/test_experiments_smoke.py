"""
Smoke tests for the experiment scripts.

We monkey-patch ``experiments._shared.load_dataset`` to return a synthetic
dataset so the experiments don't depend on MovieLens being present. Each
script is called with a minimal hyperparameter grid and we assert:
    - the script runs to completion
    - a CSV is written
    - a PNG is written
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_tiny_interactions


def _fake_load_dataset(dataset='whatever', split_mode='temporal',
                       split_seed=0, test_ratio=0.2):
    """Return (train, test_positive, threshold) compatible with the real
    loader, built from synthetic data.

    Accepts the same kwargs as the production ``load_dataset`` so tests
    that set ``split_mode='random'`` still route through the fake
    loader. ``split_seed`` is used to vary the synthetic data seed so
    multi-seed experiments see different splits.
    """
    df, _ = _make_tiny_interactions(n_users=80, n_items=40, density=0.1,
                                    seed=1 + int(split_seed))

    if split_mode == 'random':
        rng = np.random.default_rng(int(split_seed))
        train_dfs, test_dfs = [], []
        for _, g in df.groupby('user_id', sort=True):
            n = len(g)
            n_test = max(1, int(n * test_ratio))
            perm = rng.permutation(n)
            test_dfs.append(g.iloc[perm[:n_test]])
            train_dfs.append(g.iloc[perm[n_test:]])
    else:
        df = df.sort_values('timestamp')
        train_dfs, test_dfs = [], []
        for _, g in df.groupby('user_id'):
            n_test = max(1, int(len(g) * test_ratio))
            train_dfs.append(g.iloc[:-n_test])
            test_dfs.append(g.iloc[-n_test:])
    train = pd.concat(train_dfs).reset_index(drop=True)
    test = pd.concat(test_dfs).reset_index(drop=True)
    test_positive = test[test['rating'] >= 3.5].copy()
    return train, test_positive, 3.5


@pytest.fixture
def patched_loader(monkeypatch):
    import experiments._shared as shared
    monkeypatch.setattr(shared, 'load_dataset', _fake_load_dataset)
    # Also patch the loaders that experiment modules imported earlier
    for modname in ('experiments.gamma_sensitivity',
                    'experiments.graph_source_ablation',
                    'experiments.head_tail_analysis',
                    'experiments.slim_experiments',
                    'experiments.edlae_experiments',
                    'experiments.edlae_multiseed'):
        mod = __import__(modname, fromlist=['load_dataset'])
        if hasattr(mod, 'load_dataset'):
            monkeypatch.setattr(mod, 'load_dataset', _fake_load_dataset)


def test_gamma_sensitivity_smoke(tmp_path, patched_loader):
    from experiments import gamma_sensitivity
    out_dir = tmp_path / 'gamma_sensitivity'
    df = gamma_sensitivity.run(
        dataset='synth', k=5,
        gammas=[0.1, 1.0, 10.0],
        lambdas=[50], rp3_betas=[0.3],
        rp3_topK=10,
        out_dir=out_dir,
    )
    assert len(df) == 3
    assert (out_dir / 'gamma_sensitivity_synth.csv').exists()
    assert (out_dir / 'gamma_sensitivity_synth.png').exists()


def test_graph_source_ablation_smoke(tmp_path, patched_loader):
    from experiments import graph_source_ablation
    out_dir = tmp_path / 'graph_source_ablation'
    df = graph_source_ablation.run(
        dataset='synth', k=5,
        ease_lambda=50,
        gammas=[1.0, 10.0],
        topK=10,
        sources=['rp3beta', 'binary'],
        out_dir=out_dir,
    )
    assert len(df) == 4      # 2 sources x 2 gammas
    assert (out_dir / 'graph_source_ablation_synth.csv').exists()
    assert (out_dir / 'graph_source_ablation_synth.png').exists()


def test_head_tail_analysis_smoke(tmp_path, patched_loader):
    from experiments import head_tail_analysis
    out_dir = tmp_path / 'head_tail'
    df = head_tail_analysis.run(
        dataset='synth', k=5,
        ease_lambda=50, gamma=5.0,
        rp3_beta=0.6, rp3_topK=10,
        graph_source='rp3beta',
        n_buckets=3,
        out_dir=out_dir,
    )
    assert len(df) > 0
    assert (out_dir / 'head_tail_synth.csv').exists()
    assert (out_dir / 'head_tail_synth.png').exists()


def test_edlae_experiments_smoke(tmp_path, patched_loader):
    from experiments import edlae_experiments
    out_dir = tmp_path / 'edlae'
    df = edlae_experiments.run(
        dataset='synth', k=5,
        lambda_=50, dropouts=[0.3],
        gammas=[1.0, 10.0], graph_source='rp3beta',
        rp3_beta=0.6, topK=10,
        out_dir=out_dir,
    )
    assert len(df) > 0
    assert (out_dir / 'edlae_synth.csv').exists()
    assert (out_dir / 'edlae_synth.png').exists()


def test_slim_experiments_smoke(tmp_path, patched_loader):
    from experiments import slim_experiments
    out_dir = tmp_path / 'slim'
    df = slim_experiments.run(
        dataset='synth', k=5,
        l1_reg=1e-3, beta=1e-2,
        gammas=[1.0], graph_source='rp3beta',
        rp3_beta=0.6, topK=10,
        n_iter=20,
        out_dir=out_dir,
    )
    assert len(df) > 0
    assert (out_dir / 'slim_synth.csv').exists()
    assert (out_dir / 'slim_synth.png').exists()


def test_head_tail_analysis_gamma_sweep(tmp_path, patched_loader):
    """Gamma-sweep mode: emits the per-bucket gain-vs-gamma plot and
    includes a gamma=0 baseline row for every bucket."""
    from experiments import head_tail_analysis
    out_dir = tmp_path / 'head_tail_sweep'
    df = head_tail_analysis.run(
        dataset='synth', k=5,
        ease_lambda=50,
        rp3_beta=0.6, rp3_topK=10,
        graph_source='rp3beta',
        n_buckets=3,
        gammas=[0.0, 5.0, 50.0],
        out_dir=out_dir,
    )
    # 4 buckets (overall + 3) × 3 gammas = 12 rows (ndcg only)
    assert len(df) == 12
    # Gamma-sweep CSV/PNG should have the distinct filename.
    assert (out_dir / 'head_tail_gamma_sweep_synth.csv').exists()
    assert (out_dir / 'head_tail_gamma_sweep_synth.png').exists()
    # Every bucket has a gamma=0 row with abs_gain==0 (sanity anchor).
    zero_rows = df[df['gamma'] == 0.0]
    assert (zero_rows['abs_gain'] == 0.0).all()


def test_head_tail_analysis_user_activity_mode(tmp_path, patched_loader):
    """user_activity bucket mode: each user contributes to exactly one
    bucket and n_users sums to the total evaluated users."""
    from experiments import head_tail_analysis
    out_dir = tmp_path / 'head_tail_user'
    df = head_tail_analysis.run(
        dataset='synth', k=5,
        ease_lambda=50, gamma=5.0,
        rp3_beta=0.6, rp3_topK=10,
        graph_source='rp3beta',
        n_buckets=3, bucket_by='user_activity',
        out_dir=out_dir,
    )
    assert len(df) > 0
    # Output files carry the bucket_by suffix.
    assert (out_dir / 'head_tail_synth_user_activity.csv').exists()
    assert (out_dir / 'head_tail_synth_user_activity.png').exists()
    # In user_activity mode, n_users summed across the 3 buckets should
    # equal the 'overall' user count (each user contributes exactly once).
    ndcg = df[df['metric'] == 'ndcg']
    overall_n = int(ndcg[ndcg['bucket'] == 'overall']['n_users'].iloc[0])
    bucket_n = int(ndcg[ndcg['bucket'] != 'overall']['n_users'].sum())
    assert bucket_n == overall_n


def test_edlae_multiseed_smoke(tmp_path, patched_loader):
    from experiments import edlae_multiseed
    out_dir = tmp_path / 'edlae_multiseed'
    df, summary = edlae_multiseed.run(
        dataset='synth', k=5,
        lambda_=50, dropout=0.3, gamma=5.0,
        graph_source='rp3beta', rp3_beta=0.6, topK=10,
        ks=(5, 10),
        n_seeds=2,
        out_dir=out_dir,
    )
    # 3 models (EASE, EDLAE, EDLAE-Laplacian) × 2 seeds = 6 rows.
    assert len(df) == 6
    # Summary: one row per (model, config) combination.
    assert len(summary) == 3
    assert (out_dir / 'edlae_multiseed_synth.csv').exists()
    assert (out_dir / 'edlae_multiseed_synth_summary.csv').exists()
    assert (out_dir / 'edlae_multiseed_synth.png').exists()
    # Each summary row should have both NDCG@5 and NDCG@10 mean/std columns.
    for kk in (5, 10):
        for stat in ('mean', 'std'):
            assert f'NDCG@{kk}_{stat}' in summary.columns
