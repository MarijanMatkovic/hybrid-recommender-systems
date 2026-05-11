"""
Cascade retrieval: GF-CF top-N candidates, then EASE re-ranks them
(roadmap item 20). GF-CF (Shen et al. CIKM 2021) is fast and good at
recall; EASE is slower per fit but stronger at precision.

Pipeline:
    1. Compute GF-CF score for every (u, i).
    2. For each user, keep top-N candidates from GF-CF.
    3. Use the (already-fitted) EASE pred to rerank only those candidates.

This cuts EASE inference cost by roughly ``n_items / top_n`` when used
on a candidate-set evaluator. The closed-form B is still computed once
on the full data.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def topn_candidates(scores, top_n=500):
    """Per-user top-N candidate indices.

    Returns
    -------
    cand_idx : (n_users, top_n) ndarray  of item indices.
    """
    n_users, n_items = scores.shape
    if top_n >= n_items:
        return np.tile(np.arange(n_items), (n_users, 1))
    if sps.issparse(scores):
        scores = np.asarray(scores.toarray())
    cand = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
    return cand


def cascade_rerank(rerank_pred, candidates):
    """Build a final score matrix where only candidate items have
    their rerank_pred score; everything else is -inf so it never makes
    the top-k.

    Parameters
    ----------
    rerank_pred : (n_users, n_items) full pred matrix (e.g. EASE pred)
    candidates  : (n_users, top_n) of column indices

    Returns
    -------
    masked : (n_users, n_items) dense ndarray.
    """
    n_users, n_items = rerank_pred.shape
    out = np.full((n_users, n_items), -np.inf, dtype=np.float64)
    rows = np.arange(n_users)[:, None]
    if sps.issparse(rerank_pred):
        # Pull only the candidate entries
        rp = np.asarray(rerank_pred[rows.ravel(), candidates.ravel()]).reshape(
            n_users, candidates.shape[1])
    else:
        rp = np.asarray(rerank_pred)[rows, candidates]
    out[rows, candidates] = rp
    return out
