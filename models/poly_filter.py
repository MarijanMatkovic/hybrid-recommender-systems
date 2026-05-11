"""
Polynomial graph-filter pre-processing of the user-item interaction
matrix X (item 7 of the closed-form roadmap).

Computes ``X_tilde = X @ h(A_tilde)`` for a normalised item-item
adjacency A_tilde and a polynomial filter h(.). Then plain EASE is run
on X_tilde.

Two filters are available:

  * Turbo-CF (Park et al. SIGIR 2024):
        h(A) = (alpha I + (1-alpha) A_tilde)^K        K = 2 or 3
    Sharp low-pass; alpha controls how strongly we mix in the
    smoothed signal.

  * Chebyshev plateau approximation (ChebyCF, Kim et al. SIGIR 2025):
        h(A) ~= sum_{j=0..K-1} (1/K) A_tilde^j
    Cheaper but less peaky than Turbo-CF.

The pre-processing is a sparse matmul chain (no eigendecomposition),
so it stays scalable for Netflix.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def normalised_item_adjacency(X, eps: float = 1e-8, drop_diag: bool = True):
    """Symmetrically-normalised item-item adjacency:

        A_tilde = D^{-1/2} (X^T X) D^{-1/2}

    where D = diag(X^T X) (or its row-sum equivalent after diag drop).
    Sparse-in / sparse-out. Diagonal of XᵀX is dropped first so the
    filter does not over-amplify self-loops.
    """
    G = X.T @ X
    if sps.issparse(G):
        G = G.tolil()
        if drop_diag:
            G.setdiag(0)
        G = G.tocsr()
    else:
        G = np.array(G, dtype=float)
        if drop_diag:
            np.fill_diagonal(G, 0.0)

    if sps.issparse(G):
        d = np.asarray(G.sum(axis=1)).ravel()
    else:
        d = G.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)

    if sps.issparse(G):
        D = sps.diags(d_inv_sqrt)
        return (D @ G @ D).tocsr()
    return (G * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def turbo_cf(X, alpha: float = 0.7, K: int = 2):
    """Turbo-CF low-pass: X @ (alpha I + (1-alpha) A_tilde)^K.

    Recommended ranges: alpha in [0.5, 0.9], K in {2, 3}. K too high
    over-smooths and washes out individual user histories.
    """
    A = normalised_item_adjacency(X)
    if sps.issparse(A):
        I = sps.eye(A.shape[0], format='csr')
        H = (alpha * I + (1.0 - alpha) * A).tocsr()
        # X @ H @ H @ ... (K times)
        Y = X
        for _ in range(K):
            Y = (Y @ H).tocsr() if sps.issparse(Y) else Y @ H
        return Y
    else:
        I = np.eye(A.shape[0])
        H = alpha * I + (1.0 - alpha) * A
        Y = X
        for _ in range(K):
            Y = Y @ H
        return Y


def chebyshev_lowpass(X, K: int = 4):
    """Chebyshev-style plateau low-pass:

        X_tilde = sum_{j=0..K-1} (1/K) * X @ A_tilde^j

    Implemented as cumulative power iteration so we never materialise
    A^j explicitly (X is the moving accumulator). K=4 is the
    ChebyCF default; K=2 acts more like a denoiser.
    """
    A = normalised_item_adjacency(X)
    Y_acc = None
    Y_step = X
    for j in range(K):
        contrib = Y_step / K
        Y_acc = contrib if Y_acc is None else (Y_acc + contrib)
        if j < K - 1:
            Y_step = (Y_step @ A)
    return Y_acc


FILTERS = {
    'turbo_cf':   turbo_cf,
    'chebyshev':  chebyshev_lowpass,
    'identity':   lambda X, **kw: X,
}


def apply_filter(X, name: str, **kw):
    """Apply a registered filter by name."""
    if name not in FILTERS:
        raise ValueError(
            f"Unknown filter '{name}'. Available: {sorted(FILTERS)}")
    return FILTERS[name](X, **kw)
