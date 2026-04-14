"""
Shared pytest fixtures.

We generate a tiny synthetic interaction dataset so unit tests stay
fast (milliseconds) and independent of the MovieLens download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


def _make_tiny_interactions(n_users=50, n_items=30, density=0.08, seed=0):
    """Produce a small dense-ish random interaction matrix and the
    corresponding DataFrame with the schema used by the loaders
    (user_id, item_id, rating, timestamp).
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density

    # Guarantee every user has >= 3 interactions and every item >= 3 --
    # otherwise the ``min_interactions`` filter in the real loader would
    # drop rows and the tests would become flaky across seeds.
    for u in range(n_users):
        if mask[u].sum() < 3:
            idx = rng.choice(n_items, size=3, replace=False)
            mask[u, idx] = True
    for i in range(n_items):
        if mask[:, i].sum() < 3:
            idx = rng.choice(n_users, size=3, replace=False)
            mask[idx, i] = True

    users, items = np.where(mask)
    ratings = rng.choice([3.0, 3.5, 4.0, 4.5, 5.0], size=len(users))
    # timestamps: fake monotonic
    timestamps = np.arange(len(users))

    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings,
        'timestamp': timestamps,
    })
    return df, mask


@pytest.fixture(scope='session')
def tiny_interactions():
    df, _ = _make_tiny_interactions()
    return df


@pytest.fixture(scope='session')
def tiny_split(tiny_interactions):
    """Temporal 80/20 split of the synthetic data."""
    ratings = tiny_interactions.sort_values('timestamp')
    train_dfs, test_dfs = [], []
    for _, g in ratings.groupby('user_id'):
        n_test = max(1, int(len(g) * 0.2))
        train_dfs.append(g.iloc[:-n_test])
        test_dfs.append(g.iloc[-n_test:])
    train = pd.concat(train_dfs).reset_index(drop=True)
    test = pd.concat(test_dfs).reset_index(drop=True)
    return train, test


@pytest.fixture(scope='session')
def tiny_X(tiny_interactions):
    """Sparse interaction matrix for direct model tests."""
    df = tiny_interactions
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    X = csr_matrix(
        (np.ones(len(df)), (df['user_id'], df['item_id'])),
        shape=(n_users, n_items),
    )
    return X


@pytest.fixture(scope='session')
def tiny_train_X(tiny_split):
    """Training-set interaction matrix aligned with the items in the
    train split. Needed whenever the test constructs a Laplacian from a
    graph model trained on the same frame the model-under-test will see.
    """
    train, _ = tiny_split
    # Use the train DataFrame to rebuild X with the full original item
    # range -- item IDs missing from the train set become zero columns,
    # which matches the behaviour of the production fit routines that
    # encode with LabelEncoder and keep a compact item index.
    users = train['user_id'].to_numpy()
    items = train['item_id'].to_numpy()
    # Compact item encoding to match the model code path
    from sklearn.preprocessing import LabelEncoder
    u_enc = LabelEncoder().fit(users)
    i_enc = LabelEncoder().fit(items)
    u = u_enc.transform(users)
    i = i_enc.transform(items)
    X = csr_matrix((np.ones(len(u)), (u, i)))
    return X
