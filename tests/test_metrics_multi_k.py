"""
Unit tests for ``evaluate_at_ks`` (multi-k ranking evaluation) and
``random_train_test_split`` (reproducible per-user random split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import (
    random_train_test_split,
    temporal_train_test_split,
)
from evaluation.metrics import evaluate, evaluate_at_ks
from models.ease import EASE


# ---------------------------------------------------------------------------
# random_train_test_split
# ---------------------------------------------------------------------------

def test_random_split_reproducible(tiny_interactions):
    """Same seed => identical split."""
    tr1, te1 = random_train_test_split(tiny_interactions,
                                       test_ratio=0.2, seed=7)
    tr2, te2 = random_train_test_split(tiny_interactions,
                                       test_ratio=0.2, seed=7)
    # Compare sorted tuples so row-order differences don't cause false
    # negatives.
    def _sig(df):
        return tuple(sorted(zip(df['user_id'], df['item_id'])))
    assert _sig(tr1) == _sig(tr2)
    assert _sig(te1) == _sig(te2)


def test_random_split_seeds_differ(tiny_interactions):
    """Different seeds => (usually) different test set. Given 500+
    interactions we should always see at least one element differ."""
    _, te_a = random_train_test_split(tiny_interactions,
                                      test_ratio=0.2, seed=1)
    _, te_b = random_train_test_split(tiny_interactions,
                                      test_ratio=0.2, seed=2)
    def _sig(df):
        return set(zip(df['user_id'], df['item_id']))
    assert _sig(te_a) != _sig(te_b)


def test_random_split_disjoint_and_complete(tiny_interactions):
    """Train ∪ Test == full; Train ∩ Test == ∅."""
    tr, te = random_train_test_split(tiny_interactions,
                                     test_ratio=0.2, seed=0)
    full = set(zip(tiny_interactions['user_id'],
                   tiny_interactions['item_id'],
                   tiny_interactions['timestamp']))
    tr_set = set(zip(tr['user_id'], tr['item_id'], tr['timestamp']))
    te_set = set(zip(te['user_id'], te['item_id'], te['timestamp']))
    assert tr_set.isdisjoint(te_set)
    assert tr_set | te_set == full


def test_random_split_every_user_has_test(tiny_interactions):
    """Contract: every user has at least one test interaction."""
    _, te = random_train_test_split(tiny_interactions,
                                    test_ratio=0.2, seed=0)
    test_users = set(te['user_id'])
    all_users = set(tiny_interactions['user_id'])
    assert test_users == all_users


# ---------------------------------------------------------------------------
# evaluate_at_ks
# ---------------------------------------------------------------------------

class _Wrap:
    """Tiny adapter so ``evaluate`` / ``evaluate_at_ks`` see the
    (model.ease, model.pred) interface they expect."""
    def __init__(self, m):
        self.ease = m
        self.pred = m.pred


@pytest.fixture(scope='module')
def fitted_ease_small():
    """Small EASE fit on the synthetic data, cached per module."""
    from tests.conftest import _make_tiny_interactions
    df, _ = _make_tiny_interactions(n_users=60, n_items=40, density=0.1,
                                    seed=3)
    df = df.sort_values('timestamp')
    train_dfs, test_dfs = [], []
    for _, g in df.groupby('user_id'):
        n_test = max(1, int(len(g) * 0.2))
        train_dfs.append(g.iloc[:-n_test])
        test_dfs.append(g.iloc[-n_test:])
    train = pd.concat(train_dfs).reset_index(drop=True)
    test = pd.concat(test_dfs).reset_index(drop=True)
    test_positive = test[test['rating'] >= 3.0].copy()

    m = EASE()
    m.fit(train, lambda_=50.0, implicit=True)
    return m, train, test_positive


def test_evaluate_at_ks_matches_single_k(fitted_ease_small):
    """NDCG@k from evaluate_at_ks must equal NDCG@k from evaluate."""
    m, train, test = fitted_ease_small
    res_single = evaluate(_Wrap(m), train, test, k=10)
    res_multi  = evaluate_at_ks(_Wrap(m), train, test, ks=(5, 10, 20))

    # k=10 results should agree exactly (same ranking, same metric defs).
    for metric in ('NDCG', 'MAP', 'HitRate', 'Recall',
                   'Precision', 'MRR'):
        assert res_single[f'{metric}@k'] == pytest.approx(
            res_multi[f'{metric}@10'], abs=1e-12), (
                f"{metric}@10 mismatch: single={res_single[f'{metric}@k']} "
                f"multi={res_multi[f'{metric}@10']}")


def test_evaluate_at_ks_emits_all_ks(fitted_ease_small):
    """Every requested k should be represented in the output keys."""
    m, train, test = fitted_ease_small
    res = evaluate_at_ks(_Wrap(m), train, test, ks=(5, 10, 20))
    for k in (5, 10, 20):
        for metric in ('NDCG', 'MAP', 'HitRate', 'Recall',
                       'Precision', 'MRR', 'Coverage', 'Gini', 'Novelty'):
            assert f'{metric}@{k}' in res, (
                f"Missing {metric}@{k} in evaluate_at_ks result")


def test_evaluate_at_ks_recall_monotone_in_k(fitted_ease_small):
    """Recall@k is monotonically non-decreasing in k. A well-known
    sanity check: if we recommend more, we can't find fewer relevant
    items."""
    m, train, test = fitted_ease_small
    res = evaluate_at_ks(_Wrap(m), train, test, ks=(5, 10, 20))
    assert res['Recall@5'] <= res['Recall@10'] + 1e-12
    assert res['Recall@10'] <= res['Recall@20'] + 1e-12


def test_evaluate_at_ks_per_user_ndcg_shape(fitted_ease_small):
    """per_user_ndcg arrays match n_users_evaluated for every k."""
    m, train, test = fitted_ease_small
    res = evaluate_at_ks(_Wrap(m), train, test, ks=(5, 10))
    n = res['n_users_evaluated']
    for k in (5, 10):
        assert res['per_user_ndcg'][k].shape == (n,)
    assert len(res['per_user_ids']) == n


def test_evaluate_at_ks_ks_deduped_and_sorted(fitted_ease_small):
    """Passing duplicate or unsorted ks should not error; internal
    dedupe + sort should yield a canonical key set."""
    m, train, test = fitted_ease_small
    res = evaluate_at_ks(_Wrap(m), train, test, ks=(20, 10, 10, 5))
    assert res['ks'] == [5, 10, 20]
    assert f'NDCG@5' in res and f'NDCG@10' in res and f'NDCG@20' in res


def test_temporal_split_still_works(tiny_interactions):
    """Sanity: the existing temporal split hasn't regressed after the
    refactor."""
    tr, te = temporal_train_test_split(tiny_interactions, test_ratio=0.2)
    assert len(tr) + len(te) == len(tiny_interactions)
    assert set(te['user_id']) == set(tiny_interactions['user_id'])
