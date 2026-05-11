"""
Heat-diffusion / PPR pre-smoothed Gram matrix EASE (roadmap item 14).

Replaces the EASE Gram

    G = X^T X         (additive Laplacian regularisation lives here)

with the spectrally-shaped

    G_tilde = h(L_tilde)^T  G  h(L_tilde)     (multiplicative shaping)

where ``h`` is a chosen graph-spectral filter:

    'heat' :  h(L_tilde) = expm(-t L_tilde)                 (heat kernel)
    'ppr'  :  h(L_tilde) = (1 - alpha) (I - alpha A_tilde)^{-1}  (PPR)

L_tilde = I - A_tilde is the normalised Laplacian; A_tilde is the
symmetrically-normalised item-item adjacency. Multiplicative spectral
shaping is *mathematically distinct* from additive Laplacian
regularisation (mult != add inside the inverse), so this composes with
Lap-EASE rather than being redundant.

Heat-kernel construction uses a top-k truncated eigendecomposition.
PPR uses one dense inversion.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred
from .poly_filter import normalised_item_adjacency


def _heat_kernel(L, t: float, top_k: int):
    """exp(-t L) via top-k truncated eigendecomposition. L is dense."""
    n = L.shape[0]
    kk = min(int(top_k), n)
    w, V = eigh(L, subset_by_index=(0, kk - 1))
    h_diag = np.exp(-t * w)
    return (V * h_diag[np.newaxis, :]) @ V.T


def _ppr_kernel(A_dense, alpha: float):
    """(1 - alpha) (I - alpha A)^{-1}. Dense inversion."""
    n = A_dense.shape[0]
    return (1.0 - alpha) * np.linalg.inv(np.eye(n) - alpha * A_dense)


class HeatGramEASE:
    """EASE on the multiplicatively-shaped Gram G_tilde."""

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, filter_kind: str = 'heat',
            t_heat: float = 1.0, ppr_alpha: float = 0.5,
            top_k_eig: int = 256, implicit: bool = True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n = X.shape[1]

        # Build normalised item-item adjacency, then L_tilde
        A = normalised_item_adjacency(X)
        A_dense = A.toarray() if sps.issparse(A) else np.asarray(A)
        L_tilde = np.eye(n) - A_dense

        if filter_kind == 'heat':
            H = _heat_kernel(L_tilde, t=t_heat, top_k=top_k_eig)
        elif filter_kind == 'ppr':
            H = _ppr_kernel(A_dense, alpha=ppr_alpha)
        else:
            raise ValueError(f"Unknown filter_kind={filter_kind!r}; "
                             "expected 'heat' or 'ppr'.")

        G = X.T.dot(X).toarray()
        G_tilde = H.T @ G @ H

        A_mat = G_tilde + lambdas * np.eye(n)
        P = np.linalg.inv(A_mat)
        B_unc = P @ G_tilde
        diag_P = np.diag(P)
        diag_B = np.diag(B_unc)
        mu = diag_B / diag_P
        B = B_unc - P * mu[np.newaxis, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.filter_kind = filter_kind
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
