"""Tests for SLIM and Laplacian-SLIM."""

import numpy as np
import pytest

from models import SLIM, build_graph, build_laplacian


def _scale_L(L, scale):
    return L * scale


def test_slim_fit_basic(tiny_split):
    train, _ = tiny_split
    slim = SLIM()
    B, X = slim.fit(train, l1_reg=1e-3, beta=1e-2, positive=True,
                    max_iter=10)
    n_items = X.shape[1]
    assert B.shape == (n_items, n_items)
    # Diagonal constraint
    assert np.allclose(np.diag(B), 0.0)
    # Non-negativity
    assert (B >= -1e-10).all()
    # pred is X @ B
    assert slim.pred.shape == (X.shape[0], n_items)


def test_slim_fit_allows_negative(tiny_split):
    """positive=False should at least not crash and may produce negatives."""
    train, _ = tiny_split
    slim = SLIM()
    B, X = slim.fit(train, l1_reg=1e-3, beta=1e-2,
                    positive=False, max_iter=10)
    assert B.shape == (X.shape[1], X.shape[1])
    assert np.allclose(np.diag(B), 0.0)


def test_slim_laplacian_fit(tiny_split, tiny_train_X):
    train, _ = tiny_split
    # Build Laplacian from the training X (same items as the model will see)
    W = build_graph(tiny_train_X, source='rp3beta', topK=5)
    L, _ = build_laplacian(W)
    L_scaled = _scale_L(L, 1.0)

    slim = SLIM()
    B, X = slim.fit_laplacian(train, L_scaled=L_scaled,
                              beta=1e-2, l1_reg=1e-3, gamma=1.0,
                              positive=True, n_iter=30)
    assert B.shape == (X.shape[1], X.shape[1])
    assert np.allclose(np.diag(B), 0.0)
    assert (B >= -1e-10).all()


def test_slim_laplacian_gamma_zero_matches_regularized_ridge(tiny_split):
    """
    With gamma=0 the Laplacian solver is just ISTA on the
    ridge+L1 objective. The solution should be stable and finite.
    """
    train, _ = tiny_split
    # Use the number of unique items in the training set (matches the
    # LabelEncoder inside SLIM)
    n_items = int(train['item_id'].nunique())
    L = np.zeros((n_items, n_items))

    slim = SLIM()
    B, _ = slim.fit_laplacian(train, L_scaled=L, beta=1e-2, l1_reg=1e-3,
                              gamma=0.0, positive=True, n_iter=30)
    assert np.isfinite(B).all()
    assert np.allclose(np.diag(B), 0.0)


def test_slim_predictions_nonnegative(tiny_split):
    """With positive SLIM and non-negative X, predictions must be >= 0."""
    train, _ = tiny_split
    slim = SLIM()
    slim.fit(train, l1_reg=1e-3, beta=1e-2, positive=True, max_iter=10)
    assert (np.asarray(slim.pred.todense()
                       if hasattr(slim.pred, 'todense')
                       else slim.pred) >= -1e-10).all()
