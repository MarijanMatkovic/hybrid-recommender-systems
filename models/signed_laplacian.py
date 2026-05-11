"""
Signed-Laplacian EASE (roadmap item 15).

Kunegis (ICDM 2007 / EuroPar 2010) showed that the *signed Laplacian*

    L_sig = D_bar - W   where D_bar_{ii} = sum_j |W_{ij}|

is positive semi-definite even when W contains negative entries. This
is exactly what we need to extend Lap-EASE to datasets with an explicit
"dislike" signal.

For MovieLens we encode signed item co-occurrence as:

    s(u, i) = +1  if user u rated item i >= pos_thr   (liked)
             -1  if user u rated item i <= neg_thr   (disliked)
              0  otherwise

    W = X_signed^T X_signed   (then drop diag, top-K-by-|.|, symmetrise)

Items co-liked by many users get strongly positive edges; items where
one is liked and the other disliked by many users get negative edges.
This gives a thesis-novel directional modality on top of the standard
positive-only Lap-EASE. No published "signed-Laplacian EASE" exists.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


def build_signed_item_graph(ratings_df, item_enc,
                            pos_thr: float = 4.0,
                            neg_thr: float = 3.0,
                            topK: int = 200):
    """Build a signed item-item co-occurrence matrix from raw ratings.

    Parameters
    ----------
    ratings_df : DataFrame with columns ['user_id', 'item_id', 'rating']
    item_enc   : LabelEncoder fit on the same items used by the EASE X
    pos_thr    : rating >= pos_thr => positive signal (+1)
    neg_thr    : rating <= neg_thr => negative signal (-1)
    topK       : per-row top-K-by-absolute-value pruning (sparsity)

    Returns
    -------
    W_signed : dense (n_items, n_items) ndarray with values in [-1, +1] range
               (after the per-row top-K mask).
    """
    items_in = set(item_enc.classes_)
    df = ratings_df[ratings_df['item_id'].isin(items_in)].copy()
    df['item_idx'] = item_enc.transform(df['item_id'])

    sign = np.zeros(len(df), dtype=float)
    sign[(df['rating'] >= pos_thr).to_numpy()] = +1.0
    sign[(df['rating'] <= neg_thr).to_numpy()] = -1.0
    keep = sign != 0
    df_s = df[keep]
    sign_v = sign[keep]

    user_enc_local = LabelEncoder()
    u_idx = user_enc_local.fit_transform(df_s['user_id'])
    n_items = len(item_enc.classes_)
    n_users = len(user_enc_local.classes_)

    X_signed = csr_matrix(
        (sign_v, (u_idx, df_s['item_idx'].to_numpy())),
        shape=(n_users, n_items))

    W = (X_signed.T @ X_signed).toarray()
    np.fill_diagonal(W, 0.0)

    if topK is not None and 0 < topK < n_items:
        # Per-row top-K by absolute value
        abs_W = np.abs(W)
        keep_idx = np.argpartition(-abs_W, topK, axis=1)[:, :topK]
        mask = np.zeros_like(W, dtype=bool)
        rows = np.arange(n_items)[:, None]
        mask[rows, keep_idx] = True
        W = W * mask

    W = 0.5 * (W + W.T)   # symmetrise
    return W


def signed_laplacian(W):
    """L_sig = D_bar - W with D_bar_ii = sum_j |W_ij|. PSD for any
    real-valued W (Kunegis 2007)."""
    abs_d = np.sum(np.abs(W), axis=1)
    return np.diag(abs_d) - W


class SignedLaplacianEASE:
    """Lap-EASE with signed Laplacian regularisation."""

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, lambdas: float = 500.0, gamma: float = 10.0,
            W_signed=None, implicit: bool = True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df), dtype=float) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n = X.shape[1]

        G = X.T.dot(X).toarray()
        g_diag_mean = float(np.mean(np.diag(G)))

        if W_signed is None:
            L_sig = np.zeros((n, n))
        else:
            L_sig = signed_laplacian(W_signed)
            l_mean = float(np.mean(np.diag(L_sig)))
            if l_mean > 0:
                L_sig = L_sig * (g_diag_mean / l_mean)
        self.L_sig = L_sig

        A = G + lambdas * np.eye(n) + gamma * L_sig
        P = np.linalg.inv(A)
        B_unc = P @ G
        diag_P = np.diag(P)
        diag_B = np.diag(B_unc)
        mu = diag_B / diag_P
        B = B_unc - P * mu[np.newaxis, :]
        np.fill_diagonal(B, 0.0)

        self.B = B
        self.pred = make_pred(X, B)
        self.ease = self
        return B, X
