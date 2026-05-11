"""
Personalised PageRank score-diffusion re-ranker (roadmap item 19).

Takes a user's EASE scores s_u and diffuses them over the item-item
adjacency A_tilde:

    pi^{(t+1)} = (1 - alpha) s_u + alpha A_tilde pi^{(t)}

Converges to pi = (1 - alpha) (I - alpha A_tilde)^{-1} s_u; 5-10 power
iterations are enough in practice (Chung's heat-kernel PageRank, PNAS
2007, gives the closed-form bound).

Heat-kernel variant: pi = exp(-t L) s_u via a 5-term Taylor truncation.

Spreads EASE confidence to graph neighbours, recovering items EASE
under-scores due to sparsity but where the graph gives evidence.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def ppr_diffuse_user(scores_u, A_tilde, alpha: float = 0.5, n_iter: int = 10):
    """Power-iteration PPR on one user's score vector.

    Parameters
    ----------
    scores_u : (n_items,) ndarray
    A_tilde  : (n_items, n_items) sparse or dense adjacency
    alpha    : restart probability (0..1). Larger = more diffusion.
    n_iter   : number of power iterations.

    Returns
    -------
    pi : (n_items,) ndarray of diffused scores.
    """
    s = np.asarray(scores_u, dtype=np.float64).ravel()
    pi = s.copy()
    for _ in range(n_iter):
        if sps.issparse(A_tilde):
            ax = np.asarray(A_tilde @ pi).ravel()
        else:
            ax = A_tilde @ pi
        pi = (1.0 - alpha) * s + alpha * ax
    return pi


def heat_diffuse_user(scores_u, L, t: float = 1.0, n_taylor: int = 5):
    """Heat-kernel diffusion via Taylor truncation:
        exp(-t L) s = sum_{k=0..n_taylor} (-t)^k / k! L^k s
    """
    s = np.asarray(scores_u, dtype=np.float64).ravel()
    out = s.copy()
    term = s.copy()
    for k in range(1, n_taylor + 1):
        if sps.issparse(L):
            term = -(t / k) * np.asarray(L @ term).ravel()
        else:
            term = -(t / k) * (L @ term)
        out = out + term
    return out


def ppr_rerank_full(pred_matrix, A_tilde, alpha: float = 0.5,
                    n_iter: int = 10):
    """Apply PPR diffusion to all user-rows simultaneously.

    Memory: needs dense ``pred_matrix`` (n_users, n_items) -- on
    Netflix this is 61 GB. Use ``ppr_diffuse_user`` per-user there.
    """
    s = np.asarray(pred_matrix, dtype=np.float64)
    pi = s.copy()
    for _ in range(n_iter):
        if sps.issparse(A_tilde):
            # pi @ A_tilde where A_tilde is sparse -- result dense
            ax = np.asarray(pi @ A_tilde.T).astype(np.float64)
        else:
            ax = pi @ A_tilde.T
        pi = (1.0 - alpha) * s + alpha * ax
    return pi
