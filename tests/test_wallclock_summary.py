"""
Unit tests for ``experiments.wallclock_summary``.

We build a fake ``results/`` tree with handcrafted CSVs covering the
three shape-categories the aggregator sees in practice:

  * rows with an explicit ``model`` column (slim_, edlae_, multiseed)
  * rows WITHOUT a ``model`` column in an experiment that implies one
    (``gamma_sensitivity`` → Laplacian-EASE)
  * a CSV that carries no ``train_time_s`` column at all (must be
    silently skipped)

The goal is to lock down the grouping semantics so the thesis-facing
wall-clock table stays trustworthy as more experiments are added.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.wallclock_summary import (
    collect,
    format_markdown,
    summarise,
)


@pytest.fixture
def fake_results(tmp_path):
    root = tmp_path / 'results'
    # Case 1: explicit ``model`` column.
    (root / 'slim').mkdir(parents=True)
    pd.DataFrame([
        {'dataset': 'ml-small', 'model': 'SLIM',
         'gamma': 0.0, 'train_time_s': 100.0},
        {'dataset': 'ml-small', 'model': 'EASE',
         'gamma': 0.0, 'train_time_s': 15.0},
        {'dataset': 'ml-small', 'model': 'SLIM-Laplacian',
         'gamma': 1.0, 'train_time_s': 80.0},
        {'dataset': 'ml-small', 'model': 'SLIM-Laplacian',
         'gamma': 3.0, 'train_time_s': 82.0},
    ]).to_csv(root / 'slim' / 'slim_ml-small.csv', index=False)

    # Case 2: no ``model`` column, folder implies Laplacian-EASE.
    (root / 'gamma_sensitivity').mkdir()
    pd.DataFrame([
        {'dataset': 'ml-small', 'lambda': 200, 'gamma': 1.0,
         'train_time_s': 17.0},
        {'dataset': 'ml-small', 'lambda': 200, 'gamma': 3.0,
         'train_time_s': 17.5},
    ]).to_csv(root / 'gamma_sensitivity' /
              'gamma_sensitivity_ml-small.csv', index=False)

    # Case 3: CSV with no ``train_time_s`` column (must be skipped).
    (root / 'edlae_multiseed').mkdir()
    pd.DataFrame([
        {'dataset': 'synth', 'model': 'EASE',
         'ndcg_mean': 0.1, 'ndcg_std': 0.01},
    ]).to_csv(root / 'edlae_multiseed' /
              'edlae_multiseed_synth_summary.csv', index=False)

    return root


def test_collect_covers_both_shape_categories(fake_results):
    long_df = collect(fake_results)
    # 4 rows from slim + 2 from gamma_sensitivity; the no-train-time
    # summary CSV must be silently dropped.
    assert len(long_df) == 6
    models = set(long_df['model'])
    assert models == {'SLIM', 'EASE', 'SLIM-Laplacian', 'Laplacian-EASE'}


def test_collect_skips_csvs_without_train_time(fake_results):
    long_df = collect(fake_results)
    # No row should have come from the summary-only CSV.
    assert 'edlae_multiseed_synth_summary.csv' not in set(
        long_df['source_csv'])


def test_summarise_groups_and_sorts(fake_results):
    long_df = collect(fake_results)
    summary = summarise(long_df)
    # One row per (dataset, model).
    assert len(summary) == 4
    # Within a dataset, slowest model comes first.
    ml_small = summary[summary['dataset'] == 'ml-small']
    assert ml_small['model'].iloc[0] == 'SLIM'
    # SLIM-Laplacian: median of [80, 82] = 81.0
    lap = ml_small[ml_small['model'] == 'SLIM-Laplacian'].iloc[0]
    assert lap['median_s'] == pytest.approx(81.0)
    assert lap['min_s'] == 80.0
    assert lap['max_s'] == 82.0
    assert lap['n_fits'] == 2


def test_summarise_empty_frame_returns_empty_summary():
    empty = pd.DataFrame(columns=['dataset', 'model', 'train_time_s'])
    out = summarise(empty)
    assert out.empty
    assert list(out.columns) == ['dataset', 'model', 'n_fits',
                                 'median_s', 'min_s', 'max_s']


def test_format_markdown_renders_header_and_rows(fake_results):
    long_df = collect(fake_results)
    summary = summarise(long_df)
    md = format_markdown(summary)
    assert md.startswith('| Dataset | Model |')
    # Header divider row must use right-align markers for numeric cols.
    assert '| ---:|' in md
    # Every data row should land in the output: header + divider + N rows.
    assert md.count('\n') == 2 + len(summary)
    # And each model's name appears exactly once in the table body.
    for model in summary['model']:
        assert f'| {model} |' in md


def test_format_markdown_handles_empty():
    out = format_markdown(pd.DataFrame(columns=['dataset', 'model',
                                                'n_fits', 'median_s',
                                                'min_s', 'max_s']))
    assert 'No training events' in out
