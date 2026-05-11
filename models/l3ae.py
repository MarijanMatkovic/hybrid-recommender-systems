"""
L^3AE: LLM-Enhanced Linear Autoencoders.
Moon, Park & Lee, CIKM 2025 (arXiv:2508.13500) (roadmap item 10).

Closed-form template (their Eq. 4-7):

    A = X^T X + alpha S^2 + lambda I
    B = A^{-1} (X^T X + alpha S^2)     with diag-zero Lagrangian

where ``S`` is an item-item *semantic* similarity matrix. The original
paper builds S from LLM-embedding cosine similarity. For MovieLens we
don't have LLM embeddings shipped with the dataset, so we use the
genre-cosine similarity (cosine of item rows in the binary genre
indicator matrix). This keeps the experiment self-contained and
preserves the structural identity to a Laplacian regulariser: alpha S^2
plays the role of gamma L.

The thesis frames Lap-EASE as the *spectral-graph instantiation* of
this same closed-form template (replacing semantic S with graph L or
a Smola-Kondor kernel), so L^3AE is the mandatory closest-competitor
baseline.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


def build_genre_cosine_similarity(T):
    """Item-item cosine similarity from the (n_genres, n_items) tag
    matrix ``T``. Returns a dense (n_items, n_items) ndarray.

    The item rows of ``T^T`` are normalised to unit length, then we
    take ``F_n F_n^T``.
    """
    Td = T.toarray() if sps.issparse(T) else np.asarray(T)
    F = Td.T  # (n_items, n_genres)
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    Fn = F / norms
    return Fn @ Fn.T


class L3AE:
    """LLM-Enhanced Linear Autoencoder (closed-form)."""

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, S, lambdas: float = 500.0, alpha_S: float = 1.0,
            implicit: bool = True):
        """
        Parameters
        ----------
        S        : (n_items, n_items) item-item semantic similarity
        alpha_S  : weight of the semantic regularisation
        """
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n = X.shape[1]

        if S.shape != (n, n):
            raise ValueError(
                f"S must be ({n},{n}); got {S.shape}")
        S_dense = S.toarray() if sps.issparse(S) else np.asarray(S, dtype=float)

        G = X.T.dot(X).toarray()
        S2 = S_dense @ S_dense
        # Scale S^2 so its diagonal mean roughly matches G (analogous
        # to how Lap-EASE rescales L). This makes alpha_S directly
        # comparable to gamma in Lap-EASE.
        s2_mean = np.mean(np.diag(S2))
        g_mean = np.mean(np.diag(G))
        if s2_mean > 0:
            S2 = S2 * (g_mean / s2_mean)

        RHS = G + alpha_S * S2
        A = RHS + lambdas * np.eye(n)
        P = np.linalg.inv(A)
        B_unc = P @ RHS
        diag_P = np.diag(P)
        diag_B = np.diag(B_unc)
        mu = diag_B / diag_P
        B = B_unc - P * mu[np.newaxis, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.S = S_dense
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
