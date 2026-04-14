"""
EDLAE: Edge-Dropout Linear AutoEncoder (Steck, 2020 --
"Autoencoders That Don't Overfit Towards the Identity", NeurIPS 2020).

Under edge-level dropout on the input X with probability p, the expected
reconstruction loss (after taking expectation over the random binary mask
M) is:

    E[ ||M*X - (M*X) B||_F^2 ]
        = (1-p)^2 ||X - X B||_F^2
        + p(1-p) tr(diag(G) * B^T B - 2 * diag(G) * diag(B))

where G = X^T X. Together with the standard L2 penalty this yields the
closed-form Gram modification:

    G_tilde   = (1 - p) * G           +   p * diag(diag(G))         ... (*)
    (G_tilde + lambda * I) B = G_tilde
    s.t. diag(B) = 0   (via Lagrange multipliers).

Equation (*) is the form used in this implementation: popular items get
additional diagonal regularization, which breaks the trivial identity
solution and in practice generalises better than vanilla EASE.

This file also provides a Laplacian-regularized EDLAE that combines the
dropout denoising with a graph-Laplacian smoothness term:

    min_B  ||X - X B||_F^2
         + lambda * ||B||_F^2
         + (dropout stuff, folded into G_tilde)
         + gamma * tr(B^T L B)
    s.t. diag(B) = 0
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class EDLAE:
    """
    EDLAE recommender. Wraps a LabelEncoder pair like EASE so the standard
    evaluation code (``evaluate``) can be used unchanged.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def _build_X(self, df, implicit):
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )
        X = csr_matrix((values, (users, items)))
        self.X = X
        return X

    def fit(self, df, lambda_=500.0, dropout=0.5, implicit=True):
        """
        Fit EDLAE via closed-form solution with edge-dropout denoising.

        Parameters
        ----------
        lambda_ : float
            Standard L2 penalty.
        dropout : float in [0, 1)
            Edge dropout probability. dropout=0 reduces to vanilla EASE.
        """
        X = self._build_X(df, implicit)
        n_items = X.shape[1]

        G = X.T.dot(X).toarray()
        diag_G = np.diag(G).copy()

        # Dropout-denoised Gram: (1-p) G + p diag(G)
        p = float(dropout)
        G_tilde = (1.0 - p) * G
        G_tilde[np.diag_indices(n_items)] = (
            (1.0 - p) * diag_G + p * diag_G
        )  # == diag_G; keeping the decomposition explicit for clarity

        # Modified normal equation: (G_tilde + lambda I) B = G_tilde
        A = G_tilde.copy()
        A[np.diag_indices(n_items)] += lambda_
        P = np.linalg.inv(A)

        B_unconstrained = P.dot(G_tilde)

        # Diagonal constraint (column-wise Lagrange multipliers)
        diag_P = np.diag(P)
        diag_Bu = np.diag(B_unconstrained)
        mu = diag_Bu / diag_P
        B = B_unconstrained - P * mu[np.newaxis, :]
        B[np.diag_indices(n_items)] = 0.0

        self.B = B
        self.pred = X.dot(B)
        return B, X

    def fit_laplacian(self, df, L_scaled, lambda_=500.0, dropout=0.5,
                      gamma=1.0, implicit=True):
        """
        EDLAE with additional Laplacian regularization.

        Normal equation:
            (G_tilde + lambda I + gamma L) B = G_tilde
            diag(B) = 0 via Lagrange multipliers.

        Parameters
        ----------
        L_scaled : np.ndarray (n_items x n_items)
            Pre-built, pre-scaled graph Laplacian.
        lambda_ : float
        dropout : float in [0, 1)
        gamma : float
            Laplacian regularization strength (applies to L_scaled).
        """
        X = self._build_X(df, implicit)
        n_items = X.shape[1]

        G = X.T.dot(X).toarray()
        diag_G = np.diag(G).copy()

        p = float(dropout)
        G_tilde = (1.0 - p) * G
        G_tilde[np.diag_indices(n_items)] = diag_G  # denoised diag

        A = G_tilde.copy()
        A[np.diag_indices(n_items)] += lambda_
        A += gamma * L_scaled

        P = np.linalg.inv(A)
        B_unconstrained = P.dot(G_tilde)

        diag_P = np.diag(P)
        diag_Bu = np.diag(B_unconstrained)
        mu = diag_Bu / diag_P
        B = B_unconstrained - P * mu[np.newaxis, :]
        B[np.diag_indices(n_items)] = 0.0

        self.B = B
        self.pred = X.dot(B)
        return B, X

    def predict_for_user(self, user_idx, watched_set, score_vector,
                         candidate_items, k):
        """Generate top-k predictions for a single user."""
        candidates = [item for item in candidate_items
                      if item not in watched_set]
        pred = np.take(score_vector, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame({
            "user_id": [user_idx] * len(res),
            "item_id": np.take(candidates, res),
            "score": np.take(pred, res),
        }).sort_values('score', ascending=False)
        return r
