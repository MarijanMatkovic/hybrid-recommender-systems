"""
SVD-AE: Closed-form low-rank linear autoencoder via truncated SVD.
Hong et al., IJCAI 2024 (arXiv:2405.04746).

    R_tilde ~= U_k Sigma_k V_k^T              (truncated SVD)
    B = V_k diag( sigma_k^2 / (sigma_k^2 + lambda) ) V_k^T   (then diag=0)

This is ~387x faster than LightGCN on standard benchmarks while
achieving competitive NDCG. With ``filter_name`` set, the SVD is
applied to a polynomial-filtered ``X @ h(A_tilde)`` instead of plain X
-- giving "graph-aware low-rank EASE" with no extra eigendecomposition
beyond the SVD itself (roadmap item 13).

Memory profile (Netflix):
    Plain SVD of 429k x 17.7k with k=256 is feasible with scipy
    ``svds`` (~few minutes, ~few GB).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred
from .poly_filter import apply_filter
from .svd_utils import truncated_svd_safe


class SVDAE:
    """Low-rank linear autoencoder via truncated SVD.

    Parameters (fit)
    ----------------
    k          : truncation rank (default 128; range 64 to 512)
    lambdas    : L2 strength inside the eigenvalue shrinkage
    filter_name: optional polynomial filter applied to X before SVD
                 (one of: 'turbo_cf', 'chebyshev', or None)
    filter_kw  : kwargs for the filter (alpha, K, ...)
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, k: int = 128, lambdas: float = 10.0,
            filter_name=None, filter_kw=None, implicit: bool = True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X

        # Optional graph-filter pre-processing of X
        X_target = X
        if filter_name is not None:
            X_target = apply_filter(X, filter_name, **(filter_kw or {}))
            if not sps.issparse(X_target):
                X_target = csr_matrix(X_target)

        # Truncated SVD via safe helper -- uses dense LAPACK on small
        # data (ml-1m, ml-small) to avoid the ARPACK segfault observed
        # on the SRCE cluster's Cray-Python build for k >= 128. On
        # Netflix-scale matrices the helper falls back to svds.
        U, s, Vt = truncated_svd_safe(X_target, k=k)
        V_k = Vt.T  # (n_items, k)
        kk = V_k.shape[1]

        s2 = s ** 2
        shrink = s2 / (s2 + lambdas)
        # B = V_k diag(shrink) V_k^T  -- dense (n_items, n_items)
        B = (V_k * shrink[np.newaxis, :]) @ V_k.T
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.k_rank = kk
        self.singular_values = s
        self.V_k = V_k
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
