"""
SLIM (Sparse Linear Methods) for top-N recommendation.

Classic SLIM (Ning & Karypis, 2011) objective:

    min_W  ||X - X W||_F^2  +  beta * ||W||_F^2  +  l1_reg * ||W||_1
    s.t.   W >= 0
           diag(W) = 0

We solve it column-by-column with coordinate descent via sklearn's
ElasticNet. This keeps memory bounded since each column solve is a
1D regression problem.

Laplacian-regularized SLIM extends the objective:

    min_W  ||X - X W||_F^2
         + beta * ||W||_F^2
         + l1_reg * ||W||_1
         + gamma * tr(W^T L W)
    s.t.  diag(W) = 0

The Laplacian term couples columns together, so per-column coordinate
descent no longer applies. We instead use a proximal-gradient (ISTA)
solver for the full W matrix.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder


class SLIM:
    """
    SLIM model. Supports optional non-negativity and an ISTA-based
    Laplacian-regularized variant.
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

    def fit(self, df, l1_reg=1e-3, beta=1e-3, implicit=True,
            positive=True, max_iter=50, tol=1e-4, alpha_scale=1.0):
        """
        Standard SLIM via per-item ElasticNet.

        Parameters
        ----------
        df : pd.DataFrame with columns ['user_id', 'item_id', 'rating']
        l1_reg : float
            L1 regularization strength.
        beta : float
            L2 regularization strength.
        positive : bool
            Enforce W >= 0 (original SLIM constraint).
        max_iter : int
            ElasticNet coordinate-descent iterations per column.
        alpha_scale : float
            Multiplier applied to both (l1_reg + beta) -- useful for
            quick sensitivity scans.

        Returns
        -------
        B : np.ndarray (n_items x n_items)
        X : csr_matrix
        """
        X = self._build_X(df, implicit)
        n_items = X.shape[1]

        # sklearn ElasticNet parameters: alpha = (l1 + l2) / n_samples,
        # l1_ratio = l1 / (l1 + l2)
        alpha = (l1_reg + beta) * alpha_scale
        l1_ratio = l1_reg / (l1_reg + beta + 1e-12)

        B = np.zeros((n_items, n_items), dtype=np.float32)

        # Fit per column: W[:, j] regresses X[:, j] on X with X[:, j] removed
        # (we zero the j-th column before fitting to enforce diag(W) = 0)
        X_dense = X.toarray().astype(np.float32)

        for j in range(n_items):
            target = X_dense[:, j].copy()
            # Zero the j-th column in the feature matrix
            x_col_backup = X_dense[:, j].copy()
            X_dense[:, j] = 0.0

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                positive=positive,
                fit_intercept=False,
                copy_X=False,
                precompute=False,
                selection='random',
                max_iter=max_iter,
                tol=tol,
            )
            try:
                model.fit(X_dense, target)
                B[:, j] = model.coef_.astype(np.float32)
            except Exception:
                # Degenerate column -- leave zeros
                pass
            finally:
                X_dense[:, j] = x_col_backup

        # Enforce diagonal = 0 (safety)
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.pred = X.dot(B)
        return B, X

    def fit_laplacian(self, df, L_scaled, beta=1e-3, l1_reg=1e-3,
                      gamma=1.0, implicit=True, positive=True,
                      lr=None, n_iter=200, tol=1e-5, verbose=False):
        """
        SLIM with additional Laplacian regularization.

        Objective:
            min_W 0.5 * ||X - XW||_F^2
                + 0.5 * beta * ||W||_F^2
                + l1_reg * ||W||_1
                + 0.5 * gamma * tr(W^T L W)
            s.t. diag(W) = 0, W >= 0 if positive=True

        Solved via proximal gradient descent (ISTA) on the full W matrix.

        Parameters
        ----------
        L_scaled : np.ndarray (n_items x n_items)
            Pre-built, pre-scaled graph Laplacian. The caller is
            responsible for scaling so that ``gamma * L`` is commensurate
            with ``X^T X``.
        lr : float or None
            Step size. If None we estimate from an upper bound on the
            Lipschitz constant (``||X^T X|| + beta + gamma * ||L||``).
        n_iter : int
            Number of ISTA iterations.

        Returns
        -------
        B : np.ndarray
        X : csr_matrix
        """
        X = self._build_X(df, implicit)
        n_items = X.shape[1]

        G = X.T.dot(X).toarray().astype(np.float32)
        L_scaled = L_scaled.astype(np.float32)

        if lr is None:
            # Conservative Lipschitz estimate via power iteration on the
            # smooth-part Hessian H = G + beta*I + gamma*L
            H = G + beta * np.eye(n_items, dtype=np.float32) + gamma * L_scaled
            L_const = _power_iteration_spectral_norm(H, n_iter=20)
            lr = 1.0 / (L_const + 1e-8)

        B = np.zeros((n_items, n_items), dtype=np.float32)
        prev_obj = np.inf

        for it in range(n_iter):
            # Gradient of smooth part:
            # 0.5 ||X - XB||^2 has grad = -G + GB
            # 0.5 beta ||B||^2  has grad = beta B
            # 0.5 gamma tr(B^T L B) has grad = gamma L B
            GB = G.dot(B)
            grad = -G + GB + beta * B + gamma * L_scaled.dot(B)

            # Gradient step
            B = B - lr * grad

            # Proximal (soft thresholding for L1)
            if l1_reg > 0:
                B = np.sign(B) * np.maximum(np.abs(B) - lr * l1_reg, 0.0)

            # Non-negativity projection
            if positive:
                B = np.maximum(B, 0.0)

            # Enforce zero diagonal
            np.fill_diagonal(B, 0.0)

            if verbose and (it % max(1, n_iter // 10) == 0):
                recon = X.dot(B)
                residual = X - recon
                obj = (0.5 * (residual.multiply(residual)).sum()
                       + 0.5 * beta * (B * B).sum()
                       + l1_reg * np.abs(B).sum()
                       + 0.5 * gamma * np.trace(B.T.dot(L_scaled).dot(B)))
                print(f"  [SLIM-Lap iter {it:>3d}] obj={obj:.4f}")
                if abs(prev_obj - obj) < tol * max(1.0, abs(prev_obj)):
                    break
                prev_obj = obj

        np.fill_diagonal(B, 0.0)

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


def _power_iteration_spectral_norm(M, n_iter=20):
    """Estimate the largest eigenvalue magnitude of symmetric M."""
    n = M.shape[0]
    v = np.random.RandomState(0).randn(n).astype(M.dtype)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        w = M.dot(v)
        nrm = np.linalg.norm(w) + 1e-12
        v = w / nrm
    return float(v.dot(M.dot(v)))
