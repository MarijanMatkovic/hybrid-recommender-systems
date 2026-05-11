"""
DPP-MAP greedy re-ranking with score-graph kernel (roadmap item 17).
Chen, Zhang & Zhou, NeurIPS 2018 (arXiv:1709.05135) -- fast greedy MAP
inference with (1 - 1/e) approximation guarantee.

For each user we build a DPP kernel from EASE scores and an item-item
similarity:

    L_ii = s_u(i)^2 + epsilon
    L_ij = s_u(i) * s_u(j) * sim_graph(i, j)

The greedy MAP picks items that maximise the log-determinant of the
principal sub-matrix, balancing per-item quality (large s_u(i)) with
diversity (low pairwise similarity).

Adds a "diversity / popularity-bias" axis to the thesis using exactly
the same item-item graph that powers Lap-EASE.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def dpp_map_greedy(L, k):
    """Greedy MAP for k-DPP via Chen et al. (2018) fast Cholesky-based
    algorithm. O(k^2 N) where N is the kernel size.

    Parameters
    ----------
    L : (N, N) PSD kernel matrix (dense)
    k : number of items to pick

    Returns
    -------
    selected : list of indices in selection order (length <= k)
    """
    N = L.shape[0]
    kk = min(k, N)
    d2 = np.diag(L).astype(np.float64).copy()
    # Storage for Cholesky-like factor columns
    C = np.zeros((N, kk), dtype=np.float64)
    selected = []
    for step in range(kk):
        j = int(np.argmax(d2))
        if d2[j] <= 0:
            break
        e_j = float(np.sqrt(d2[j]))
        if step == 0:
            new_col = L[:, j] / e_j
        else:
            # new_col[i] = (L[i, j] - C[i, :step] @ C[j, :step]) / e_j
            new_col = (L[:, j] - C[:, :step] @ C[j, :step]) / e_j
        C[:, step] = new_col
        d2 = d2 - new_col ** 2
        d2[j] = -np.inf
        selected.append(j)
    return selected


def dpp_rerank_user(scores_u, sim_graph, k=10, top_pool=200, eps=1e-6):
    """Re-rank one user's scores with a DPP-MAP greedy selection.

    Parameters
    ----------
    scores_u   : (n_items,) ndarray of EASE / Lap-EASE scores
    sim_graph  : (n_items, n_items) similarity matrix (sparse or dense)
    k          : number of items to return
    top_pool   : restrict candidates to the top ``top_pool`` items by
                 score before building the kernel (efficiency)

    Returns
    -------
    ranked_items : (k,) ndarray of item indices in DPP-MAP order.
    """
    n_items = len(scores_u)
    pool_size = min(top_pool, n_items)
    pool = np.argpartition(-scores_u, pool_size - 1)[:pool_size]
    pool = pool[np.argsort(-scores_u[pool])]

    s = np.maximum(scores_u[pool], 0.0).astype(np.float64) + eps
    if sps.issparse(sim_graph):
        sub = np.asarray(sim_graph[pool, :][:, pool].toarray())
    else:
        sub = np.asarray(sim_graph)[np.ix_(pool, pool)]

    # Kernel: L[i, i] = s_i^2 + eps; L[i, j] = s_i s_j sim_ij
    L = (s[:, None] * sub) * s[None, :]
    np.fill_diagonal(L, s ** 2)

    selected_local = dpp_map_greedy(L, k)
    return pool[np.asarray(selected_local, dtype=int)]


def dpp_rerank_pred(pred_matrix, sim_graph, k=10, top_pool=200):
    """Apply DPP-MAP re-ranking to every user-row of ``pred_matrix``.

    Returns
    -------
    A (n_users, n_items) score matrix where the top-k items per user
    have descending integer scores k, k-1, ..., 1 and everything else
    is 0 -- so downstream evaluation picks up the DPP order without
    needing a separate API.
    """
    n_users, n_items = pred_matrix.shape
    out = np.zeros((n_users, n_items), dtype=np.float64)
    for u in range(n_users):
        if sps.issparse(pred_matrix):
            su = np.asarray(pred_matrix[u, :].toarray()).ravel()
        else:
            su = np.asarray(pred_matrix[u, :]).ravel()
        order = dpp_rerank_user(su, sim_graph, k=k, top_pool=top_pool)
        # Encode rank: pos 0 -> k, pos 1 -> k-1, ...
        scores = np.arange(len(order), 0, -1)
        out[u, order] = scores
    return out
