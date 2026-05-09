"""
Graph-Shrunk EASE (GS-EASE).

Novel extension of Laplacian-EASE where the regularisation target is the
item-item graph W rather than zero:

    min_B  ||X - XB||^2_F  +  λ||B||^2_F  +  γ·tr((B - W)^T L (B - W))
           s.t.  diag(B) = 0

The closed-form unconstrained solution is:

    A = G + λI + γL          (same inverse as Lap-EASE)
    R = G + γ · L @ W        (W appears only in the RHS)
    B_unc = A^{-1} R

Lagrangian diagonal correction (μ_j = B_unc_jj / A^{-1}_jj):

    B_ij = B_unc_ij - P_ij · (B_unc_jj / P_jj),   diag(B) = 0

where P = A^{-1}.

Limiting cases
--------------
* γ → 0 :  A → G + λI, R → G  →  standard EASE (exact).
* W → 0 :  R → G              →  Lap-EASE RHS, but NOT identical to the
              Lap-EASE diagonal correction (different mu_j). The two share
              the same A^{-1} but differ in the unconstrained solution;
              numerical equality holds only when γ = 0.

The caller is responsible for:
  - Pre-building L (via ``models.build_laplacian``) and scaling it.
  - Pre-building W (via ``models.build_graph``) and converting to dense.
  - Matching the Laplacian scaling used for Lap-EASE so γ sweeps are
    comparable between the two models.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


class GraphShrunkEASE:
    """
    Graph-Shrunk EASE: closes the gap between EASE, Lap-EASE, and
    hybrid score fusion by pulling B toward the item graph W via the
    Laplacian term.

    Usage
    -----
    model = GraphShrunkEASE()
    B, X = model.fit(train_df, L_scaled=L, W_dense=W, lambda_=500, gamma=1.0)
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, L_scaled: np.ndarray, W_dense: np.ndarray,
            lambda_: float = 500.0, gamma: float = 1.0,
            implicit: bool = True):
        """
        Fit GS-EASE.

        Parameters
        ----------
        df : pd.DataFrame
            Training interactions with columns ['user_id', 'item_id'].
        L_scaled : np.ndarray, shape (n_items, n_items)
            Pre-built, pre-scaled graph Laplacian (dense).  Use the same
            scaling as for Lap-EASE so gamma values are comparable.
        W_dense : np.ndarray, shape (n_items, n_items)
            Item-item similarity target matrix (dense).  Typically the
            same W used to build L.
        lambda_ : float
            EASE ridge parameter.
        gamma : float
            Laplacian regularisation strength.

        Returns
        -------
        B : np.ndarray, shape (n_items, n_items)
        X : csr_matrix, shape (n_users, n_items)
        """
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=np.float32)
                  if implicit
                  else (df['rating'].to_numpy() / df['rating'].max()).astype(np.float32))

        X = csr_matrix((values, (users, items)))
        self.X = X
        n_items = X.shape[1]

        G = X.T.dot(X).toarray()
        diag_idx = np.diag_indices(n_items)

        # A = G + λI + γL  (same system matrix as Lap-EASE)
        A = G + gamma * L_scaled
        A[diag_idx] += lambda_

        P = np.linalg.inv(A)   # A^{-1}, shape (n_items, n_items)

        # R = G + γ · L @ W  (GS-EASE right-hand side)
        LW = L_scaled @ W_dense
        R = G + gamma * LW

        # Unconstrained solution
        Q = P @ R   # shape (n_items, n_items)

        # Lagrangian diagonal correction: B_ij = Q_ij - P_ij * (Q_jj / P_jj)
        mu = np.diag(Q) / np.diag(P)   # shape (n_items,)
        B = Q - P * mu[np.newaxis, :]
        B[diag_idx] = 0.0              # clean up floating-point residue

        self.B = B
        self.pred = make_pred(X, B)
        return B, X
