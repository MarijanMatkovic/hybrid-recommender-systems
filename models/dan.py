"""
DAN: Degree-Aware Normalisation for linear recommenders.
Park et al., SIGIR 2025 (arXiv:2504.05805) -- "Why is Normalization
Necessary for Linear Recommenders?" (roadmap item 12).

Variants of degree normalisation applied to the EASE Gram matrix that
preserve the closed-form eigenstructure:

    G_alpha = D^{-alpha} (X^T X) D^{-(1-alpha)}

with alpha in [0, 1]:
  * alpha = 0   -> column-only normalisation D^0 G D^{-1}
  * alpha = 0.5 -> symmetric (sqrt-degree) normalisation
  * alpha = 1.0 -> row-only normalisation D^{-1} G D^{0}

Then standard EASE closed form on G_alpha:

    B = (G_alpha + lambda I)^{-1} G_alpha   with diag(B) = 0.

This is a direct competitor in the closed-form linear-AE family;
benchmarking against it is mandatory before claiming the Laplacian
penalty is novel.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


class DAN_EASE:
    """EASE with degree-aware normalisation of the Gram matrix.

    Parameters
    ----------
    alpha : float in [0, 1]
        Degree-normalisation exponent for the LEFT factor; the right
        factor uses 1 - alpha. alpha = 0.5 gives the symmetric (D^{-1/2}
        G D^{-1/2}) normalisation that's the canonical baseline.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, alpha: float = 0.5,
            implicit: bool = True, eps: float = 1e-8):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n = X.shape[1]

        G = X.T.dot(X).toarray()
        d = np.diag(G).copy() + eps
        # G_alpha = D^{-alpha} G D^{-(1-alpha)}
        d_a = d ** (-alpha)
        d_b = d ** (-(1.0 - alpha))
        G_norm = (G * d_a[:, None]) * d_b[None, :]

        A = G_norm + lambdas * np.eye(n)
        P = np.linalg.inv(A)
        # Steck diag-zero Lagrangian
        B_unc = P @ G_norm
        diag_P = np.diag(P)
        diag_B = np.diag(B_unc)
        mu = diag_B / diag_P
        B = B_unc - P * mu[np.newaxis, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.alpha = alpha
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
