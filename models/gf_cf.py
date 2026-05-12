"""
GF-CF: Graph Filter Collaborative Filtering (Shen et al., CIKM 2021,
"How Powerful is Graph Convolution for Recommendation?").

Closed-form recommender via:

    R_tilde = D_u^{-1/2} R D_i^{-1/2}              (degree-normalised)
    R_tilde = U_k S_k V_k^T                       (truncated SVD)
    pred    = R @ ((1 - alpha) * (R^T R)_norm + alpha * V_k V_k^T)

The first term is the "linear" filter (item-item neighbours) and the
second is the "ideal low-pass" projector. alpha trades off between
them.

Used by the cascade pipeline (roadmap item 20) as a fast candidate
generator before EASE re-ranking.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred
from .svd_utils import truncated_svd_safe


class GFCF:
    """Graph-Filter CF (Shen et al. 2021), closed-form."""

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, k: int = 256, alpha: float = 0.3,
            implicit: bool = True, eps: float = 1e-8):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        R = csr_matrix((values, (users, items)))
        self.X = R
        n_users, n_items = R.shape

        # Degree-normalised R
        du = np.asarray(R.sum(axis=1)).ravel() + eps
        di = np.asarray(R.sum(axis=0)).ravel() + eps
        Du_inv_sq = diags(1.0 / np.sqrt(du))
        Di_inv_sq = diags(1.0 / np.sqrt(di))
        R_tilde = (Du_inv_sq @ R @ Di_inv_sq).astype(np.float64)

        # Truncated SVD via safe helper -- dense LAPACK on small data
        # to avoid the ARPACK segfault observed on the cluster for
        # k >= 128 (which is the typical GF-CF rank).
        _, _, Vt = truncated_svd_safe(R_tilde, k=k)
        V_k = Vt.T  # (n_items, k)
        kk = V_k.shape[1]

        # Linear / item-item filter: D_i^{-1/2} (R^T R) D_i^{-1/2}
        # (We rescale here so its diagonal mean matches the SVD term.)
        RtR = (R.T @ R).toarray().astype(np.float64)
        # Normalise by item degree
        di_inv_sq = 1.0 / np.sqrt(di)
        RtR = (RtR * di_inv_sq[:, None]) * di_inv_sq[None, :]

        # Match scale of low-pass to RtR diag
        lowpass = V_k @ V_k.T
        rtr_mean = float(np.mean(np.diag(RtR))) + eps
        lp_mean = float(np.mean(np.diag(lowpass))) + eps
        lowpass = lowpass * (rtr_mean / lp_mean)

        B = (1.0 - alpha) * RtR + alpha * lowpass
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.V_k = V_k
        self.alpha_gfcf = alpha
        self.pred = make_pred(R, B)
        self.ease = self
        return B, R
