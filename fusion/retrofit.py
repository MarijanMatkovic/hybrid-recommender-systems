"""
Faruqui et al. (NAACL 2015) retrofitting of pre-trained vectors.

Original (in NLP): retrofit word vectors q^_i so they are close to
their pre-trained anchors AND smooth over a synonym/lexicon graph:

    q_i^{(t+1)} = (alpha * q_hat_i + sum_j beta_ij q_j^{(t)})
                / (alpha + sum_j beta_ij)

Equivalent to one Jacobi sweep of the linear system
``(alpha I + L_neighbours) Q = alpha Q_hat`` for an unnormalised
Laplacian-like operator built from the neighbour weights.

Cross-domain application (roadmap item 11): treat the *columns of an
EASE B matrix* as item embeddings, with a content/genre similarity
matrix providing the smoothing graph. Cheap (5-10 sweeps), no SGD,
closed-form fixed point. Strong narrative hook: same operator that
retrofits word vectors retrofits item embeddings.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def retrofit(B, neighbours, alpha: float = 1.0, n_iter: int = 10,
             include_self: bool = False, restore_diag_zero: bool = True):
    """Retrofit the columns of B to be smooth over a neighbour graph.

    Parameters
    ----------
    B          : (d, n_items) ndarray  -- columns are item embeddings.
                 For EASE B (n_items x n_items), each column is the
                 set of weights from the rest of the catalogue to a
                 single target item.
    neighbours : (n_items, n_items) similarity matrix (beta_ij). Will
                 be densified internally if sparse.
    alpha      : anchor weight on the original B. Larger alpha = less
                 smoothing.
    n_iter     : Jacobi sweeps (5-10 is typical; converges quickly).
    include_self : if False (default) the diagonal of ``neighbours`` is
                 zeroed before summing, so beta_ii does not influence
                 the update.
    restore_diag_zero : zero the diagonal of the result; if B is an
                 EASE B-matrix this preserves the "no self-recommend"
                 constraint.

    Returns
    -------
    Q : (d, n_items) ndarray   -- retrofitted columns.
    """
    if sps.issparse(neighbours):
        N = neighbours.toarray()
    else:
        N = np.asarray(neighbours, dtype=float)
    if not include_self:
        N = N - np.diag(np.diag(N))

    row_sum = N.sum(axis=1)
    denom = alpha + row_sum
    # Guard against zero rows (isolated items)
    denom = np.where(denom > 0, denom, 1.0)

    Q = np.asarray(B, dtype=float).copy()
    Nt = N.T  # since Q @ N^T is the column-aware contraction
    for _ in range(int(n_iter)):
        Q_new = (alpha * B + Q @ Nt) / denom[np.newaxis, :]
        Q = Q_new

    if restore_diag_zero and Q.shape[0] == Q.shape[1]:
        np.fill_diagonal(Q, 0.0)
    return Q
