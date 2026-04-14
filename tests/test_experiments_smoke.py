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


def _fake_load_dataset(dataset='whatever'):
    """Return (train, test_positive, threshold) compatible with the real
    loader, built from synthetic data."""
    df, _ = _make_tiny_interactions(n_users=80, n_items=40, density=0.1,
                                    seed=1)
    # Temporal split
    df = df.sort_values('timestamp')
    train_dfs, test_dfs = [], []
    for _, g in df.groupby('user_id'):
        n_test = max(1, int(len(g) * 0.2))
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
                    'experiments.edlae_experiments'):
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
