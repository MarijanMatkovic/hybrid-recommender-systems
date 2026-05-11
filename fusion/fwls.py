"""
Feature-Weighted Linear Stacking (FWLS) -- roadmap item 16.
Sill, Takács, Mackey & Lin, arXiv:0911.0460 (2009; Netflix Prize 2nd place).

Per-user adaptive ensemble where the per-base-model weight is a linear
function of user meta-features phi(u):

    s_FWLS(u, i) = sum_m sum_f w_{m,f} * phi_f(u) * S_m(u, i)

This is a single ridge regression in the augmented feature space
{phi_f(u) * S_m(u, i)} -- fully closed-form via np.linalg.lstsq /
np.linalg.solve. Meta-features used here:

    f0 : constant 1                (baseline weight)
    f1 : log(1 + |history_u|)      (heavy users get different blend)

The thesis can extend with score-variance per user, graph-centrality,
etc. without changing the solver.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps


def user_meta_features(X):
    """Per-user meta-feature matrix.

    Returns
    -------
    phi : (n_users, n_features) ndarray. f0=1 bias, f1=log(1+|history_u|).
    """
    if sps.issparse(X):
        hist = np.asarray((X != 0).sum(axis=1)).ravel().astype(np.float64)
    else:
        hist = (np.asarray(X) != 0).sum(axis=1).astype(np.float64)
    f0 = np.ones_like(hist)
    f1 = np.log1p(hist)
    return np.column_stack([f0, f1])


def build_augmented_features(base_scores, phi, users, items):
    """Build the FWLS feature matrix for given (user, item) pairs.

    Parameters
    ----------
    base_scores : sequence of (n_users, n_items) arrays
    phi         : (n_users, n_features)
    users, items : array-like of indices (same length)

    Returns
    -------
    Xfwls : (n_pairs, n_models * n_features) ndarray.
    """
    M = len(base_scores)
    F = phi.shape[1]
    n_pairs = len(users)
    Xfwls = np.empty((n_pairs, M * F), dtype=np.float64)
    users = np.asarray(users, dtype=int)
    items = np.asarray(items, dtype=int)
    for m, S in enumerate(base_scores):
        # Per-pair score s_m(u, i)
        if sps.issparse(S):
            s_mi = np.asarray(S[users, items]).ravel()
        else:
            s_mi = np.asarray(S)[users, items]
        # Per-pair feature: phi_f(u) * s_m(u, i)
        for f in range(F):
            Xfwls[:, m * F + f] = phi[users, f] * s_mi
    return Xfwls


def fit_fwls(base_scores, phi, pos_pairs, neg_pairs, lambda_=1.0):
    """Fit FWLS ridge on positive vs negative (u, i) pairs.

    Parameters
    ----------
    base_scores : list of (n_users, n_items) score arrays
    phi         : (n_users, n_features)
    pos_pairs   : array-like of (user, item) -- relevance label = 1
    neg_pairs   : array-like of (user, item) -- relevance label = 0
    lambda_     : ridge regularisation

    Returns
    -------
    weights : (M*F,) ndarray  -- flattened weights w_{m,f}.
    """
    u_p = [p[0] for p in pos_pairs]
    i_p = [p[1] for p in pos_pairs]
    u_n = [p[0] for p in neg_pairs]
    i_n = [p[1] for p in neg_pairs]
    Xp = build_augmented_features(base_scores, phi, u_p, i_p)
    Xn = build_augmented_features(base_scores, phi, u_n, i_n)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(u_p)), np.zeros(len(u_n))])
    A = X.T @ X + lambda_ * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(A, b)


def apply_fwls(base_scores, phi, weights):
    """Apply fitted FWLS weights to compute per-(u,i) fused scores.

    Vectorised: per user u, the effective weight on model m is
        c_m(u) = sum_f weights[m*F + f] * phi[u, f]
    so the fused score becomes
        s_fwls(u, i) = sum_m c_m(u) * S_m(u, i)
    """
    M = len(base_scores)
    F = phi.shape[1]
    fused = None
    for m, S in enumerate(base_scores):
        w_m = weights[m * F:(m + 1) * F]  # (F,)
        coef_per_user = phi @ w_m  # (n_users,)
        S_d = np.asarray(S, dtype=np.float64) if not sps.issparse(S) else None
        if S_d is None:
            # Sparse path: keep S sparse and use multiply
            contrib = (S.multiply(coef_per_user[:, None])).toarray()
        else:
            contrib = S_d * coef_per_user[:, None]
        fused = contrib if fused is None else (fused + contrib)
    return fused


def sample_negatives(train_pos_pairs, n_items, n_per_pos=5, seed=42):
    """Sample uniform-random negative (u, i) pairs from items the user
    has not interacted with in ``train_pos_pairs``.

    Cheap negative sampler for FWLS validation. NOT recommended for
    headline numbers (proper FWLS validation should hold out a fraction
    of training interactions first).
    """
    rng = np.random.default_rng(seed)
    pos_set = {(int(u), int(i)) for u, i in train_pos_pairs}
    per_user = {}
    for u, i in train_pos_pairs:
        per_user.setdefault(int(u), set()).add(int(i))
    negs = []
    for u, _ in train_pos_pairs:
        u = int(u)
        seen = per_user.get(u, set())
        # Draw n_per_pos items uniformly until we get unseen ones
        for _ in range(n_per_pos):
            j = int(rng.integers(0, n_items))
            tries = 0
            while j in seen and tries < 5:
                j = int(rng.integers(0, n_items))
                tries += 1
            negs.append((u, j))
    return negs
