"""
Output-level fusion of multiple recommender model scores.
Implements roadmap items 1 (RRF + CombMNZ) and 9 (inverse-variance
fusion using Wager-Wang-Liang dropout variance from EDLAE).

All functions take a list of (n_users, n_items) score matrices (or
LazyPred wrappers) and return a single fused score matrix of the same
shape. ``materialise_pred`` is a tiny helper that handles both ndarray
and LazyPred inputs uniformly -- since LazyPred only supports per-row
indexing, fusion that needs the full matrix dense-materialises it.

Memory note
-----------
Unlike Lap-EASE evaluation, fusion fundamentally needs the full pred
matrix in memory (RRF needs per-user ranks, CombMNZ needs per-user
z-scores). For Netflix-scale data this is ~61 GB and requires either
chunked fusion or operating only on per-user top-K candidate lists.
The functions below are sized for ml-1m (165 MB pred); the
``rrf_top_k`` variant is provided for the Netflix candidate-list case.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def materialise_pred(pred):
    """Convert a LazyPred / sparse / ndarray into a dense (n_u, n_i)
    ndarray. Caller is responsible for the memory cost."""
    if isinstance(pred, np.ndarray):
        return pred
    if sps.issparse(pred):
        return np.asarray(pred.toarray())
    # LazyPred: walk per-user rows
    n_users, n_items = pred.shape
    out = np.empty((n_users, n_items), dtype=np.float64)
    for u in range(n_users):
        out[u, :] = pred[u, :]
    return out


# ---------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion (RRF) -- Cormack et al. SIGIR 2009
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(score_matrices, k=60):
    """RRF: fused(u, i) = sum_m 1 / (k + rank_m(u, i)).

    Per-user rank is computed by argsort: rank=1 for the highest score.
    No tuning required; k=60 is the canonical default and the result
    is invariant to score scale (which is the entire point -- our
    EASE / RP3beta / SLIM scores live on different scales).

    Parameters
    ----------
    score_matrices : sequence of (n_users, n_items) arrays
    k              : RRF damping (default 60; Cormack 2009)

    Returns
    -------
    Dense fused (n_users, n_items) ndarray.
    """
    fused = None
    for S in score_matrices:
        S_d = materialise_pred(S)
        # rank by sorting -S so largest gets rank 1
        order = np.argsort(-S_d, axis=1, kind='stable')
        ranks = np.empty_like(order, dtype=np.float64)
        rows = np.arange(S_d.shape[0])[:, None]
        cols = np.arange(S_d.shape[1])[None, :].astype(np.float64) + 1.0
        ranks[rows, order] = cols
        contrib = 1.0 / (k + ranks)
        fused = contrib if fused is None else (fused + contrib)
    return fused


def rrf_top_k(score_per_user_lists, n_items, k=60):
    """RRF for memory-constrained settings: each model emits a sorted
    top-K list per user (item ids in rank order), and we fuse only the
    union of those lists.

    Parameters
    ----------
    score_per_user_lists : list of (n_users, K_m) ndarrays of item indices
    n_items              : total items, for output shape
    k                    : RRF damping

    Returns
    -------
    fused (n_users, n_items) sparse-friendly ndarray; non-candidate
    cells are 0 so argpartition still works downstream.
    """
    n_users = score_per_user_lists[0].shape[0]
    fused = np.zeros((n_users, n_items), dtype=np.float64)
    for ranks in score_per_user_lists:
        for u in range(n_users):
            for r, item in enumerate(ranks[u]):
                fused[u, int(item)] += 1.0 / (k + r + 1)
    return fused


# ---------------------------------------------------------------------------
# CombMNZ with z-score normalisation -- Lee 1997
# ---------------------------------------------------------------------------

def combmnz_zscore(score_matrices):
    """CombMNZ: per-user z-score, sum across models, multiply by the
    number of models that placed the item above zero (consensus boost).

    Lee (1997) showed CombMNZ beats CombSUM on TREC by rewarding items
    that multiple models agree on.
    """
    z_list = []
    for S in score_matrices:
        S_d = materialise_pred(S)
        mu = S_d.mean(axis=1, keepdims=True)
        sd = S_d.std(axis=1, keepdims=True) + 1e-12
        z_list.append((S_d - mu) / sd)
    Z = np.stack(z_list, axis=0)
    score_sum = Z.sum(axis=0)
    n_above = (Z > 0).sum(axis=0).astype(np.float64)
    return score_sum * n_above


# ---------------------------------------------------------------------------
# 9. Inverse-variance fusion + EDLAE dropout variance
# ---------------------------------------------------------------------------

def inverse_variance_fusion(score_matrices, variance_matrices, eps=1e-12):
    """BLUE / Bayesian-model-averaging fusion under Gaussian noise:

        s_fused = (sum_m s_m / sigma_m^2) / (sum_m 1 / sigma_m^2)

    A model with smaller predictive variance pulls more weight toward
    its score on a per-cell basis. When variance is uniform across
    cells this reduces to a simple weighted average.

    Parameters
    ----------
    score_matrices    : list of (n_users, n_items) arrays
    variance_matrices : list of (n_users, n_items) arrays of variances
                         (or scalars / 1D arrays broadcastable to that)
    """
    if len(score_matrices) != len(variance_matrices):
        raise ValueError("score / variance lists must be same length")
    inv_vars = []
    weighted = None
    for S, V in zip(score_matrices, variance_matrices):
        S_d = materialise_pred(S)
        V_d = np.asarray(V, dtype=np.float64) if not isinstance(V, np.ndarray) \
              else V
        iv = 1.0 / (V_d + eps)
        contrib = S_d * iv
        weighted = contrib if weighted is None else (weighted + contrib)
        inv_vars.append(iv)
    inv_var_sum = inv_vars[0]
    for iv in inv_vars[1:]:
        inv_var_sum = inv_var_sum + iv
    return weighted / (inv_var_sum + eps)


def edlae_dropout_variance(X, B, dropout):
    """Closed-form predictive variance from edge dropout
    (Wager, Wang & Liang 2013, derivation specialised to EDLAE).

    For a linear autoencoder pred = X @ B, dropout on the input X with
    rate p induces predictive variance

        Var[pred[u, i]] approximately = p / (1 - p) * sum_j x_{u,j}^2 * B_{j,i}^2

    This connects the dropout-theory chapter directly to a per-cell
    fusion weight (item 9 of the roadmap).

    Parameters
    ----------
    X       : sparse (n_users, n_items) interaction matrix
    B       : dense (n_items, n_items) item-item weight matrix
    dropout : float in [0, 1)

    Returns
    -------
    Variance ndarray (n_users, n_items). Memory-equivalent to a dense
    pred matrix; callers should chunk on large datasets.
    """
    p = float(dropout)
    if p <= 0.0:
        return np.zeros((X.shape[0], B.shape[1]))
    factor = p / (1.0 - p + 1e-12)
    X2 = X.multiply(X) if sps.issparse(X) else X * X
    B2 = B * B
    out = X2.dot(B2)
    if sps.issparse(out):
        out = out.toarray()
    return factor * np.asarray(out)
