"""
Closed-form Lap-EASE extensions covering items 3, 4, 6 of the closed-form
hybridisation roadmap (compass_artifact_*.md):

  3. GramShrinkEASE       -- Ledoit-Wolf-style shrinkage of the Gram
                              matrix toward a graph similarity target
                              (DUET / L^3AE-style move).
  4. MultiLaplacianEASE   -- weighted sum of multiple Laplacians as M.
  6. MahalanobisShrinkEASE -- generalised GS-EASE: shrinks B toward
                               an arbitrary anchor B0 with metric M.

All three reuse the same closed-form template

    B_unc = (G + lambda I + gamma M)^-1 (G + gamma M B0)
    B     = diag-zero Lagrangian rescale of B_unc

so the heavy lifting is one matrix inversion. Wrapped in the same
``EASE``-compatible interface (``.user_enc``, ``.item_enc``, ``.B``,
``.X``, ``.pred``) so the existing evaluate_at_ks / bucketed_metrics
pipeline works without changes.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from .lazy_pred import make_pred


def _build_X(df, user_enc, item_enc, implicit=True):
    users = user_enc.fit_transform(df['user_id'])
    items = item_enc.fit_transform(df['item_id'])
    values = (np.ones(len(df), dtype=float) if implicit
              else df['rating'].to_numpy() / df['rating'].max())
    return csr_matrix((values, (users, items)))


def _diag_zero_lagrangian(P, RHS):
    """Steck diagonal-zero Lagrangian rescale of a closed-form solution.

    Given B_unc = P @ RHS with P symmetric, returns B such that
    diag(B) = 0 by per-column subtraction of mu_j * P[:, j] where
    mu_j = B_unc[j,j] / P[j,j].
    """
    B_unc = P @ RHS
    diag_P = np.diag(P)
    diag_B = np.diag(B_unc)
    mu = diag_B / diag_P
    B = B_unc - P * mu[np.newaxis, :]
    np.fill_diagonal(B, 0.0)
    return B


def _scale_to_match(target_diag_mean, M):
    """Rescale M so its mean(diag(M)) matches ``target_diag_mean``."""
    m_diag = np.mean(np.diag(M))
    if m_diag <= 0:
        return M
    return M * (target_diag_mean / m_diag)


# ---------------------------------------------------------------------------
# Item 3: graph-aware Gram shrinkage (DUET / Ledoit-Wolf style)
# ---------------------------------------------------------------------------

class GramShrinkEASE:
    """
    EASE on a shrunken Gram matrix:

        G_tilde = (1 - rho) * G + rho * S_graph_scaled
        B       = (G_tilde + lambda I)^-1 G_tilde   with diag(B) = 0.

    S_graph is rescaled so its diagonal mean matches G's, before the
    convex combination.

    This is identical in structure to L^3AE's
    (X^T X + alpha S^2 + lambda I)^-1 (X^T X + alpha S^2) but with
    rho parameterising the *convex* combination (always invertible)
    rather than additive scale.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, rho: float = 0.1,
            S_graph=None, implicit: bool = True):
        X = _build_X(df, self.user_enc, self.item_enc, implicit=implicit)
        self.X = X
        n = X.shape[1]

        G = X.T.dot(X)
        if sps.issparse(G):
            G = G.toarray()
        G = np.asarray(G, dtype=float)

        if S_graph is None:
            G_tilde = G
        else:
            S = S_graph.toarray() if sps.issparse(S_graph) else np.asarray(S_graph)
            S = _scale_to_match(np.mean(np.diag(G)), S)
            G_tilde = (1.0 - rho) * G + rho * S

        A = G_tilde + lambdas * np.eye(n)
        P = np.linalg.inv(A)
        B = _diag_zero_lagrangian(P, G_tilde)
        self.B = B
        self.pred = make_pred(X, B)
        # evaluate_at_ks reads model.ease.user_enc / .item_enc; wire a
        # self-reference so a standalone closed-form model can be used
        # directly without an external wrapper.
        self.ease = self
        return B, X


# ---------------------------------------------------------------------------
# Item 4: multi-graph Laplacian EASE
# ---------------------------------------------------------------------------

class MultiLaplacianEASE:
    """
    Lap-EASE with M = sum_i alpha_i * L_i for multiple item-item
    Laplacians. Reduces to plain Lap-EASE when only one Laplacian is
    supplied.

    Each Laplacian is independently rescaled so its diagonal mean
    matches G's, before the weighted sum -- this mirrors the per-graph
    scaling done in single-Laplacian Lap-EASE so gamma has comparable
    meaning across graph counts.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, gamma: float = 10.0,
            laplacians=(), weights=None, implicit: bool = True):
        if not laplacians:
            raise ValueError("MultiLaplacianEASE requires >=1 Laplacian.")
        if weights is None:
            weights = [1.0 / len(laplacians)] * len(laplacians)
        if len(weights) != len(laplacians):
            raise ValueError("weights and laplacians length mismatch.")

        X = _build_X(df, self.user_enc, self.item_enc, implicit=implicit)
        self.X = X
        n = X.shape[1]

        G = X.T.dot(X).toarray()
        g_diag_mean = float(np.mean(np.diag(G)))

        M = np.zeros((n, n))
        for w, L in zip(weights, laplacians):
            L_dense = L.toarray() if sps.issparse(L) else np.asarray(L)
            M = M + w * _scale_to_match(g_diag_mean, L_dense)
        self.M = M

        A = G + lambdas * np.eye(n) + gamma * M
        P = np.linalg.inv(A)
        B = _diag_zero_lagrangian(P, G)
        self.B = B
        self.pred = make_pred(X, B)
        # evaluate_at_ks reads model.ease.user_enc / .item_enc; wire a
        # self-reference so a standalone closed-form model can be used
        # directly without an external wrapper.
        self.ease = self
        return B, X


# ---------------------------------------------------------------------------
# Item 6: Mahalanobis shrinkage to an arbitrary anchor B0
# ---------------------------------------------------------------------------

class MahalanobisShrinkEASE:
    """
    Generalised GS-EASE:

        B_unc = (G + lambda I + gamma M)^-1 (G + gamma M B0)
        B     = diag-zero rescale of B_unc

    The anchor B0 encodes positive prior information about which
    item-item interactions are likely; the metric M (Mahalanobis-style)
    chooses *how* deviations from B0 are penalised. Setting B0=0
    recovers Lap-EASE; setting M=L and B0=W recovers GS-EASE.

    Convenient anchors (caller's responsibility to construct):
      * W^2 or W^3 (k-hop neighbourhood -- Higher-Order EASE prior)
      * D^-1 W (random-walk transition matrix)
      * exp(-t L) (heat-kernel prior)
      * V_k V_k^T (top-k eigvecs of L_norm; GF-CF low-pass projector)
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, gamma: float = 10.0,
            M=None, B0=None, implicit: bool = True):
        X = _build_X(df, self.user_enc, self.item_enc, implicit=implicit)
        self.X = X
        n = X.shape[1]

        G = X.T.dot(X).toarray()

        if M is None:
            M_arr = np.zeros((n, n))
        else:
            M_arr = M.toarray() if sps.issparse(M) else np.asarray(M, dtype=float)
            M_arr = _scale_to_match(np.mean(np.diag(G)), M_arr)

        if B0 is None:
            B0_arr = np.zeros((n, n))
        else:
            B0_arr = B0.toarray() if sps.issparse(B0) else np.asarray(B0, dtype=float)

        A = G + lambdas * np.eye(n) + gamma * M_arr
        P = np.linalg.inv(A)
        RHS = G + gamma * (M_arr @ B0_arr)
        B = _diag_zero_lagrangian(P, RHS)
        self.B = B
        self.pred = make_pred(X, B)
        # evaluate_at_ks reads model.ease.user_enc / .item_enc; wire a
        # self-reference so a standalone closed-form model can be used
        # directly without an external wrapper.
        self.ease = self
        return B, X
