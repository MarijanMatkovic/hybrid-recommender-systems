"""Tests for EDLAE and Laplacian-EDLAE."""

import numpy as np

from models import EDLAE, build_graph, build_laplacian


def test_edlae_basic(tiny_split):
    train, _ = tiny_split
    model = EDLAE()
    B, X = model.fit(train, lambda_=50.0, dropout=0.3)
    assert B.shape == (X.shape[1], X.shape[1])
    assert np.allclose(np.diag(B), 0.0)
    assert np.isfinite(B).all()


def test_edlae_dropout_zero_reduces_to_ease_like(tiny_split):
    """
    With dropout=0 the EDLAE formulation collapses to vanilla EASE:
        G_tilde = G, and the system becomes (G + lambda I) B = G
        with diag(B) = 0 -- the canonical EASE normal equation.
    The resulting B should match classic EASE (up to numerical noise).
    """
    from models import EASE

    train, _ = tiny_split

    # Compare to EASE
    ease = EASE()
    B_ease, X_ease = ease.fit(train, lambda_=50.0, implicit=True)

    edlae = EDLAE()
    B_edlae, X_edlae = edlae.fit(train, lambda_=50.0, dropout=0.0)

    # Same shape
    assert B_ease.shape == B_edlae.shape
    # The two solutions should agree closely
    assert np.allclose(B_ease, B_edlae, atol=1e-8)


def test_edlae_laplacian_changes_solution(tiny_split, tiny_train_X):
    train, _ = tiny_split
    W = build_graph(tiny_train_X, source='rp3beta', topK=5)
    L, _ = build_laplacian(W)

    # Without Laplacian
    m0 = EDLAE()
    B0, _ = m0.fit(train, lambda_=50.0, dropout=0.3)

    # With Laplacian
    m1 = EDLAE()
    B1, _ = m1.fit_laplacian(train, L_scaled=L,
                             lambda_=50.0, dropout=0.3, gamma=1.0)

    assert B1.shape == B0.shape
    # Solutions must differ
    assert np.abs(B1 - B0).sum() > 1e-6
    # Diagonal still zero
    assert np.allclose(np.diag(B1), 0.0)


def test_edlae_laplacian_gamma_zero_matches_edlae(tiny_split, tiny_train_X):
    train, _ = tiny_split
    W = build_graph(tiny_train_X, source='rp3beta', topK=5)
    L, _ = build_laplacian(W)

    m_nolap = EDLAE()
    B_nolap, _ = m_nolap.fit(train, lambda_=50.0, dropout=0.3)

    m_lap = EDLAE()
    B_lap, _ = m_lap.fit_laplacian(train, L_scaled=L,
                                   lambda_=50.0, dropout=0.3, gamma=0.0)

    assert np.allclose(B_nolap, B_lap, atol=1e-8)
