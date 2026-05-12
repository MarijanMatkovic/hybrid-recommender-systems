"""
Safe truncated-SVD helper.

scipy.sparse.linalg.svds (ARPACK) has been segfaulting on the SRCE
Supek cluster's Cray-Python 3.11 / LibSci build whenever k >= ~128.
This affects both SVD-AE (k in {64, 128, 256}) and GF-CF (k=256).

The cluster has perfectly adequate dense LAPACK; for ml-1m (n_items ~=
3.4k) a full dense SVD of a 6040 x 3369 matrix takes ~5 sec and uses
~250 MB. So we just prefer the dense path on small matrices and fall
back to ARPACK only when the matrix is too large for dense (Netflix,
17.7k items).

Threshold of 10000 along the smaller dimension keeps ml-1m / ml-small
on the dense path and routes Netflix through ARPACK.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import svds


DENSE_SVD_THRESHOLD = 10000


def truncated_svd_safe(X, k, prefer_dense=True):
    """Top-k singular values / vectors of X.

    Returns (U_k, s_k, Vt_k) with singular values in descending order:
        U_k  shape (n_rows, k)
        s_k  shape (k,)
        Vt_k shape (k, n_cols)

    Always uses a stable LAPACK path on small matrices. On large
    matrices falls back to scipy.sparse.linalg.svds (ARPACK).

    Parameters
    ----------
    X : (n_rows, n_cols) sparse or dense array. Will be densified for
        the dense path.
    k : int -- truncation rank.
    prefer_dense : bool -- if True (default), use dense SVD whenever
        min(X.shape) < DENSE_SVD_THRESHOLD. Set False to force ARPACK.
    """
    n_rows, n_cols = X.shape
    n_min = min(n_rows, n_cols)
    kk = max(1, min(int(k), n_min - 1))

    if prefer_dense and n_min < DENSE_SVD_THRESHOLD:
        X_dense = X.toarray() if sps.issparse(X) else np.asarray(X)
        X_dense = X_dense.astype(np.float64, copy=False)
        # Full thin SVD: U (m, n_min), s (n_min,), Vt (n_min, n)
        U, s, Vt = np.linalg.svd(X_dense, full_matrices=False)
        return U[:, :kk], s[:kk], Vt[:kk, :]

    # ARPACK sparse path (used only for Netflix-scale data)
    U, s, Vt = svds(X.astype(np.float64), k=kk)
    # svds returns ascending; reorder descending
    order = np.argsort(-s)
    return U[:, order], s[order], Vt[order, :]
