"""
Graph source abstraction for Laplacian regularization.

Given a user-item interaction matrix X, produces an item-item similarity
matrix W from one of several sources:
    - 'rp3beta'  : 3-step random walk with popularity dampening
    - 'p3alpha'  : 3-step random walk without popularity dampening
    - 'itemknn'  : cosine similarity top-K
    - 'binary'   : binary co-occurrence (1 if item pair co-occurred >0 times)

All sources share the same top-K sparsification and return a CSR sparse
(n_items x n_items) matrix with zero diagonal and no row/column
L1-normalisation (the Laplacian builder handles symmetrisation).
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix

from .rp3beta import RP3beta
from .p3alpha import P3alpha
from .itemknn import ItemKNN


VALID_SOURCES = ('rp3beta', 'p3alpha', 'itemknn', 'binary')


def build_graph(X, source='rp3beta', topK=200,
                rp3_alpha=1.0, rp3_beta=0.6,
                p3_alpha=1.0,
                itemknn_shrink=0.0,
                implicit=True):
    """
    Build an item-item similarity matrix from one of several sources.

    Parameters
    ----------
    X : csr_matrix, shape (n_users, n_items)
    source : str
        One of {'rp3beta', 'p3alpha', 'itemknn', 'binary'}.
    topK : int
        Top-K neighbours per item (applied to all sources except 'binary',
        which uses unbounded co-occurrence). For 'binary' topK is still
        enforced to keep the Laplacian manageable.
    rp3_alpha, rp3_beta : float
        RP3beta parameters.
    p3_alpha : float
        P3alpha parameter.
    itemknn_shrink : float
        ItemKNN shrinkage term.

    Returns
    -------
    W : csr_matrix, shape (n_items, n_items)
        Item-item similarity with zero diagonal. Not normalised.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown graph source: {source}. "
                         f"Valid: {VALID_SOURCES}")

    X = csr_matrix(X)

    if source == 'rp3beta':
        model = RP3beta()
        W = model.fit(X, alpha=rp3_alpha, beta=rp3_beta, topK=topK,
                      implicit=implicit, normalize_similarity=False)

    elif source == 'p3alpha':
        model = P3alpha()
        W = model.fit(X, alpha=p3_alpha, topK=topK,
                      implicit=implicit, normalize_similarity=False)

    elif source == 'itemknn':
        model = ItemKNN()
        W = model.fit(X, topK=topK, shrink=itemknn_shrink,
                      implicit=implicit, normalize_similarity=False)

    elif source == 'binary':
        W = _binary_cooccurrence(X, topK=topK, implicit=implicit)

    return W


def _binary_cooccurrence(X, topK=200, implicit=True):
    """
    Binary co-occurrence graph: W[i, j] = 1 iff items i and j were
    co-interacted by at least one user. Then top-K is applied by
    co-occurrence count (ties broken arbitrarily).
    """
    if implicit:
        X = (X > 0).astype(np.float32)

    # Co-occurrence counts
    G = X.T.dot(X)            # sparse (n_items x n_items)
    G = G.tolil()
    G.setdiag(0)
    G = G.tocsr()

    # Top-K by count per row
    n_rows = G.shape[0]
    rows, cols, data = [], [], []
    for i in range(n_rows):
        row = G.getrow(i)
        if row.nnz == 0:
            continue
        if row.nnz <= topK:
            cols_i = row.indices
        else:
            top = np.argpartition(row.data, -topK)[-topK:]
            cols_i = row.indices[top]
        # All retained entries have value 1.0 (binary)
        rows.extend([i] * len(cols_i))
        cols.extend(cols_i)
        data.extend([1.0] * len(cols_i))

    W = csr_matrix((data, (rows, cols)), shape=G.shape)
    return W


def build_laplacian(W, normalise='none'):
    """
    Build a graph Laplacian from a similarity matrix W.

    The matrix is first symmetrised:  W_sym = (W + W^T) / 2

    Two variants are supported:

    - ``normalise='none'`` (combinatorial / unnormalised Laplacian):
          L = D - W_sym
      where D = diag(d_i),  d_i = sum_j W_sym[i, j].
      Eigenvalues lie in [0, 2 * max(d_i)] and are dominated by the
      most-popular items. This is the "default" Laplacian used in our
      original experiments.

    - ``normalise='sym'`` (symmetric normalised Laplacian):
          L = I - D^{-1/2} W_sym D^{-1/2}
      Eigenvalues lie in [0, 2] regardless of the degree distribution,
      so high-degree (popular) items no longer over-contribute to the
      regulariser. This is the textbook fix for the head-bias / tail-
      collapse problem and is what spectral clustering uses.

    Returns
    -------
    L : np.ndarray (dense, n_items x n_items)
    degrees : np.ndarray (n_items,)
        The raw row-sums of W_sym (not affected by ``normalise``).
        Useful for downstream scaling and diagnostics.
    """
    if normalise not in ('none', 'sym'):
        raise ValueError(
            f"Unknown normalise={normalise!r}; expected 'none' or 'sym'.")

    if sps.issparse(W):
        W_dense = W.toarray()
    else:
        W_dense = np.asarray(W)

    W_sym = (W_dense + W_dense.T) / 2.0
    degrees = W_sym.sum(axis=1)

    if normalise == 'none':
        L = np.diag(degrees) - W_sym
    else:  # 'sym'
        # Avoid division by zero for isolated items (degree == 0).
        d_safe = np.maximum(degrees, 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)
        # L_sym = I - D^{-1/2} W_sym D^{-1/2}
        # Compute the normalised similarity by broadcasting (no diag matmul).
        W_norm = W_sym * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        L = np.eye(W_dense.shape[0], dtype=W_norm.dtype) - W_norm
        # Isolated nodes (d_i = 0) should not contribute -- their row/col
        # of L should be all zeros, not 1 on the diagonal.
        isolated = degrees < 1e-12
        if isolated.any():
            L[isolated, :] = 0.0
            L[:, isolated] = 0.0

    return L, degrees


# Spectral filter types supported by build_spectral_filter.
VALID_FILTER_TYPES = ('laplacian', 'L2', 'heat', 'regularized')


def build_spectral_filter(W, filter_type='laplacian', t=1.0,
                          normalise='sym'):
    """
    Build a spectral regularization matrix K from a similarity matrix W.

    All variants share the same eigenvectors as the graph Laplacian L;
    they differ only in how the eigenvalues are transformed:

    - ``'laplacian'`` : K = L               (identity transform, λ → λ)
      Standard Laplacian penalty; same as ``build_laplacian``.

    - ``'L2'``        : K = L @ L            (λ → λ²)
      Squared Laplacian; penalises high-frequency components more
      aggressively.  Items whose signal lies in high-frequency eigenmodes
      are pushed harder toward their graph neighbours.

    - ``'heat'``      : K = I - exp(-t·L)   (λ → 1 - exp(-tλ))
      Derived from the heat diffusion kernel.  For small tλ, behaves
      like tL; for large tλ, saturates to 1.  Connects to the BSPM /
      GF-CF lineage.  ``t`` controls the diffusion time (larger t =
      stronger smoothing cutoff).

    - ``'regularized'``: K = I - (I + t·L)^{-1}  (λ → tλ/(1 + tλ))
      Regularised Laplacian.  Low-pass: the penalty on frequency λ
      saturates to 1 as λ → ∞, giving equal treatment to very
      high-frequency components.  ``t`` controls the crossover.

    All returned matrices K are symmetric PSD and share L's null space
    (the all-ones vector), so tr(B^T K B) = 0 iff B is graph-smooth.

    Parameters
    ----------
    W : csr_matrix or np.ndarray, shape (n_items, n_items)
    filter_type : str
        One of ``VALID_FILTER_TYPES``.
    t : float
        Diffusion/regularization scale parameter.  Ignored for
        ``'laplacian'`` and ``'L2'``.
    normalise : str
        Laplacian normalisation passed to ``build_laplacian``.
        Use ``'sym'`` (default) so eigenvalues lie in [0, 2].

    Returns
    -------
    K : np.ndarray, shape (n_items, n_items)
        Spectral regularization matrix (dense, symmetric PSD).
    degrees : np.ndarray, shape (n_items,)
        Raw degree vector from ``build_laplacian`` (for downstream
        scaling checks).
    """
    if filter_type not in VALID_FILTER_TYPES:
        raise ValueError(
            f"Unknown filter_type={filter_type!r}; "
            f"valid: {VALID_FILTER_TYPES}")

    L, degrees = build_laplacian(W, normalise=normalise)

    if filter_type == 'laplacian':
        return L, degrees

    # Eigendecomposition (symmetric → use eigh for speed + stability).
    # L is real symmetric, so all eigenvalues are real and >= 0.
    vals, vecs = np.linalg.eigh(L)   # vals in ascending order
    # Clip tiny negatives from floating-point noise.
    vals = np.maximum(vals, 0.0)

    if filter_type == 'L2':
        fvals = vals ** 2
    elif filter_type == 'heat':
        # K = I - exp(-tL)  so that f(0)=0 (graph-constant functions get 0)
        fvals = 1.0 - np.exp(-t * vals)
    elif filter_type == 'regularized':
        # K = I - (I + tL)^{-1}  so f(λ) = tλ/(1+tλ)
        fvals = (t * vals) / (1.0 + t * vals)

    K = (vecs * fvals[np.newaxis, :]) @ vecs.T
    # Symmetrize to cancel floating-point asymmetry from eigenvec products.
    K = (K + K.T) * 0.5
    return K, degrees
