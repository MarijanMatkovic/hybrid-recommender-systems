"""
CAR / SAR Bayesian-prior EASE (roadmap item 18).

Spatial-statistics priors on the B matrix that interpolate between
"independent ridge" (rho=0) and "fully Laplacian" (rho near 1) via a
single knob rho:

    CAR (Conditional Autoregressive):
        Prec = D - rho * W            with |rho| < 1 / lambda_max(D^-1 W)
        rho = 0   -> diagonal precision (vanilla ridge)
        rho near 1 -> approximately Laplacian D - W (Lap-EASE)

    SAR (Simultaneous Autoregressive):
        Prec = (I - rho * W)^T (I - rho * W)
        Polynomial filter directly; rho = 0 -> identity prior.

We plug Prec into the same closed-form template as Lap-EASE:

    B = (X^T X + lambda I + mu * Prec)^{-1} X^T X        (diag-zero rescale)

Full type-II marginal-likelihood optimisation for (mu, rho) is the
"empirical Bayes" stretch goal flagged in the roadmap; this commit
provides the solver + grid-search interface, which matches the existing
gamma-sweep machinery for Lap-EASE.

Connects MRF-CF (Steck NeurIPS 2019) to the Laplacian work: a graph-
structured precision matrix is precisely a CAR / IGMRF prior on B.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


def car_precision(W, rho: float):
    """CAR precision matrix Prec = D - rho * W with D = diag(|W| row-sums).

    Using |W| in D keeps Prec PSD even when W has small signed entries
    near the rho stability boundary.
    """
    Wd = W.toarray() if sps.issparse(W) else np.asarray(W, dtype=float)
    D = np.diag(np.abs(Wd).sum(axis=1))
    return D - rho * Wd


def sar_precision(W, rho: float):
    """SAR precision Prec = (I - rho W)^T (I - rho W). Always PSD."""
    Wd = W.toarray() if sps.issparse(W) else np.asarray(W, dtype=float)
    n = Wd.shape[0]
    M = np.eye(n) - rho * Wd
    return M.T @ M


class CARSAR_EASE:
    """CAR or SAR Bayesian-prior EASE."""

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, W, lambdas: float = 500.0, mu: float = 10.0,
            rho: float = 0.5, prior: str = 'car', implicit: bool = True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n = X.shape[1]

        if prior == 'car':
            Prec = car_precision(W, rho)
        elif prior == 'sar':
            Prec = sar_precision(W, rho)
        else:
            raise ValueError(
                f"Unknown prior={prior!r}; expected 'car' or 'sar'.")

        # Rescale Prec so mu has comparable meaning to Lap-EASE's gamma
        G = X.T.dot(X).toarray()
        g_mean = float(np.mean(np.diag(G)))
        p_mean = float(np.mean(np.diag(Prec)))
        if p_mean > 0:
            Prec = Prec * (g_mean / p_mean)
        self.Prec = Prec

        A = G + lambdas * np.eye(n) + mu * Prec
        P = np.linalg.inv(A)
        B_unc = P @ G
        diag_P = np.diag(P)
        diag_B = np.diag(B_unc)
        mu_c = diag_B / diag_P
        B = B_unc - P * mu_c[np.newaxis, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.prior = prior
        self.rho = rho
        self.mu = mu
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
