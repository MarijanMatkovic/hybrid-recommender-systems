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


def build_laplacian(W):
    """
    Build an unnormalised graph Laplacian from a similarity matrix W.

    Symmetrises W (W_sym = (W + W^T) / 2), builds L = D - W_sym.

    Returns
    -------
    L : np.ndarray (dense, n_items x n_items)
        The dense Laplacian. Callers needing memory efficiency should
        process W directly; for our thesis-scale datasets dense L is fine.
    """
    if sps.issparse(W):
        W_dense = W.toarray()
    else:
        W_dense = np.asarray(W)

    W_sym = (W_dense + W_dense.T) / 2.0
    degrees = W_sym.sum(axis=1)
    L = np.diag(degrees) - W_sym
    return L, degrees
