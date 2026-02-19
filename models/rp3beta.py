import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

class RP3beta:
    """
    RP3beta recommender (Paudel et al., 2016).
    Computes item-item similarity via a 3-step random walk on the user-item
    bipartite graph, with popularity dampening controlled by beta.

    The random walk: item_i -> user -> item_j, with transition probabilities
    normalized by node degrees raised to alpha (users) and beta (items).
    """

    def __init__(self):
        pass

    def fit(self, X, alpha=1.0, beta=0.6, topK=100, implicit=True,
            normalize_similarity=True):
        """
        Fit RP3beta model.

        Parameters
        ----------
        X: csr_matrix, shape (n_users, n_items)
            User-item interaction matrix (same one used by EASE).
        alpha : float
            Controls user-side transition probability weighting.
        beta : float
            Popularity dampening exponent. Higher beta = more penalty on
            popular items. This is the key parameter that differentiates
            RP3beta from P3alpha.
        topK : int
            Keep only top-K entries per item in the similarity matrix
            (sparsification for efficiency).

        Returns
        -------
        W: csr_matrix, shape (n_items, n_items)
            Sparse item-item similarity matrix.
        """
        self.alpha = alpha
        self.beta = beta
        self.topK = topK

        if implicit:
            X = (X > 0).astype(np.float32)

        X = csr_matrix(X)
        n_users, n_items = X.shape

        # --- Step 1: Build transition matrices ---
        # User degree: how many items each user interacted with
        # Item degree (popularity): how many users interacted with each item
        user_degree = np.array(X.sum(axis=1)).flatten()  # shape (n_users,)
        item_degree = np.array(X.sum(axis=0)).flatten()  # shape (n_items,)

        # Avoid division by zero
        user_degree[user_degree == 0] = 1.0
        item_degree[item_degree == 0] = 1.0

        # --- Step 2: Random walk computation ---
        # The 3-step random walk: item -> user -> item
        # P(user | item) ∝ X[user, item] / item_degree[item]^alpha
        # P(item' | user) ∝ X[user, item'] / user_degree[user]
        # Final similarity: W[item, item'] = sum_user P(user|item) * P(item'|user) / item_degree[item']^beta

        # Normalize X by user degree (row-normalize with alpha exponent)
        # Pui = X^T normalized by item degree^alpha (column of X = item)
        # Then multiply by X normalized by user degree

        # Row-normalize X by user degree
        user_degree_inv = np.power(user_degree, -1.0)
        D_user_inv = sps.diags(user_degree_inv)

        # Column-normalize X by item degree^alpha
        item_degree_alpha_inv = np.power(item_degree, -alpha)
        D_item_alpha_inv = sps.diags(item_degree_alpha_inv)

        # Pui (item->user transition): X * D_item_alpha_inv (normalize columns)
        # Piu (user->item transition): D_user_inv * X (normalize rows)

        # W = (X * D_item_alpha_inv)^T * (D_user_inv * X)
        #   = D_item_alpha_inv * X^T * D_user_inv * X
        # Then apply beta dampening on item popularity

        Xt_normalized = X.T.dot(D_user_inv)  # (n_items x n_users)
        W = D_item_alpha_inv.dot(Xt_normalized).dot(X)  # (n_items x n_items)

        # Apply beta dampening: divide each column j by item_degree[j]^beta
        if beta > 0:
            item_degree_beta_inv = np.power(item_degree, -beta)
            D_item_beta_inv = sps.diags(item_degree_beta_inv)
            W = W.dot(D_item_beta_inv)

        # --- Step 3: Zero out self-similarity ---
        W = W.tolil()
        W.setdiag(0)
        W = W.tocsr()

        # --- Step 4: Top-K sparsification ---
        W = self._top_k_sparse(W, topK)

        # --- Step 5: Optionally normalize ---
        if normalize_similarity:
            W = normalize(W, norm='l1', axis=1)

        self.W = W
        return W

    @staticmethod
    def _top_k_sparse(matrix, k):
        """Keep only top-k values per row in a sparse matrix."""
        n_rows = matrix.shape[0]
        rows, cols, data = [], [], []

        for i in range(n_rows):
            row = matrix.getrow(i)
            if row.nnz == 0:
                continue
            if row.nnz <= k:
                # Keep all entries
                cols_i = row.indices
                data_i = row.data
            else:
                # Keep only top-k
                top_idx = np.argpartition(row.data, -k)[-k:]
                cols_i = row.indices[top_idx]
                data_i = row.data[top_idx]

            rows.extend([i] * len(cols_i))
            cols.extend(cols_i)
            data.extend(data_i)

        return csr_matrix((data, (rows, cols)), shape=matrix.shape)