"""
Unit tests for head/tail bucket helpers and the Wilcoxon pairing logic.

These tests do NOT train a full model -- they exercise the bucketing
math directly on handcrafted inputs so the correctness guarantees are
unambiguous (bucket size balance, label ordering, user/item axis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.head_tail_analysis import (
    _item_popularity_buckets,
    _user_activity_buckets,
    _quantile_labels,
    _wilcoxon_paired,
)


def _make_skewed(n_items=15, seed=0):
    """Build interactions where item popularity is strictly monotonic
    (item 0 = most popular, item n-1 = least)."""
    rows = []
    # item i appears (n_items - i) times across distinct users
    uid = 0
    for i in range(n_items):
        for _ in range(n_items - i):
            rows.append({'user_id': uid, 'item_id': i,
                         'rating': 5.0, 'timestamp': uid})
            uid += 1
    return pd.DataFrame(rows)


def _make_user_skewed(n_users=15, seed=0):
    """Build interactions where user activity is strictly monotonic
    (user 0 has most interactions, user n-1 has fewest)."""
    rows = []
    t = 0
    for u in range(n_users):
        for i in range(n_users - u):
            rows.append({'user_id': u, 'item_id': i,
                         'rating': 5.0, 'timestamp': t})
            t += 1
    return pd.DataFrame(rows)


def test_quantile_labels_three_vs_five():
    assert _quantile_labels(3) == ('head', 'torso', 'tail')
    assert _quantile_labels(5) == ('q1', 'q2', 'q3', 'q4', 'q5')


def test_item_popularity_head_is_most_popular():
    df = _make_skewed(n_items=9)  # 9 items -> 3 per bucket at n=3
    bucket_of = _item_popularity_buckets(df, n_buckets=3,
                                         labels=('head', 'torso', 'tail'))
    # head = items 0..2 (most popular), tail = 6..8 (least)
    assert set(i for i, b in bucket_of.items() if b == 'head') == {0, 1, 2}
    assert set(i for i, b in bucket_of.items() if b == 'tail') == {6, 7, 8}


def test_item_popularity_balanced_buckets():
    df = _make_skewed(n_items=12)  # 4 per bucket
    bucket_of = _item_popularity_buckets(df, n_buckets=3)
    counts = pd.Series(list(bucket_of.values())).value_counts()
    assert counts.to_dict() == {'head': 4, 'torso': 4, 'tail': 4}


def test_item_popularity_five_quintiles():
    df = _make_skewed(n_items=10)  # 2 per quintile
    labels = _quantile_labels(5)
    bucket_of = _item_popularity_buckets(df, n_buckets=5, labels=labels)
    counts = pd.Series(list(bucket_of.values())).value_counts()
    for lbl in labels:
        assert counts[lbl] == 2


def test_user_activity_head_is_most_active():
    df = _make_user_skewed(n_users=9)
    bucket_of = _user_activity_buckets(df, n_buckets=3,
                                       labels=('head', 'torso', 'tail'))
    # user 0..2 are the most active, user 6..8 the least
    assert set(u for u, b in bucket_of.items() if b == 'head') == {0, 1, 2}
    assert set(u for u, b in bucket_of.items() if b == 'tail') == {6, 7, 8}


def test_user_activity_each_user_exactly_one_bucket():
    df = _make_user_skewed(n_users=12)
    bucket_of = _user_activity_buckets(df, n_buckets=3)
    users = df['user_id'].unique()
    assert set(bucket_of.keys()) == set(users)
    # Every user appears once in exactly one bucket.
    assert len(bucket_of) == len(users)


# ---------------------------------------------------------------------------
# Wilcoxon pairing logic
# ---------------------------------------------------------------------------

def test_wilcoxon_paired_identical_series_returns_p_equals_1():
    """If baseline == model, diff is all zeros and we should report
    p=1 (no evidence)."""
    users = list(range(20))
    vals = np.linspace(0.1, 0.5, 20)
    base = (vals.copy(), users)
    same = (vals.copy(), users)
    stat, p, n = _wilcoxon_paired(base, same)
    assert n == 20
    assert p == 1.0


def test_wilcoxon_paired_significant_when_strictly_larger():
    """Uniformly better model: Wilcoxon should give a small p-value."""
    rng = np.random.default_rng(0)
    users = list(range(60))
    base_vals = rng.uniform(0.1, 0.5, size=60)
    mod_vals = base_vals + 0.05  # strictly better for every user
    stat, p, n = _wilcoxon_paired((base_vals, users), (mod_vals, users))
    assert n == 60
    assert p < 0.01


def test_wilcoxon_paired_intersects_on_user_id():
    """Only overlapping users are used for the paired test; the rest
    are dropped cleanly."""
    # Only users {10, 11, 12, 13} appear in both.
    base = (np.array([0.1, 0.2, 0.3, 0.4]), [1, 2, 10, 11])
    mod  = (np.array([0.9, 0.8, 0.7, 0.6]), [10, 11, 12, 13])
    stat, p, n = _wilcoxon_paired(base, mod)
    assert n == 2   # common = {10, 11}


def test_wilcoxon_paired_returns_nan_when_no_overlap():
    """No common users -> nan p-value, n_pairs=0."""
    base = (np.array([0.1, 0.2]), [1, 2])
    mod  = (np.array([0.3, 0.4]), [3, 4])
    stat, p, n = _wilcoxon_paired(base, mod)
    assert n == 0
    assert np.isnan(p)
