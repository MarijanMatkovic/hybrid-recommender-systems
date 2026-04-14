import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class P3alpha:
    """
    P3alpha recommender (Cooper et al., 2014).

    Item-item similarity computed via a 3-step random walk on the user-item
    bipartite graph, normalized by node degrees raised to alpha.

    This is a special case of RP3beta with beta = 0 (no popularity
    dampening on destination items). The random walk:
        item_i -> user -> item_j

    Transitions:
        P(user | item) = X[user, item] / item_degree[item]^alpha
        P(item' | user) = X[user, item'] / user_degree[user]

    Similarity:
        W[i, j] = sum_user P(user | i) * P(j | user)
    """

    def __init__(self):
        pass

    def fit(self, X, alpha=1.0, topK=100, implicit=True,
            normalize_similarity=True):
        """
        Fit P3alpha model.

        Parameters
        ----------
        X : csr_matrix, shape (n_users, n_items)
        alpha : float
            Exponent applied to item degree in the transition normalisation.
        topK : int
            Per-row sparsification cut-off.
        implicit : bool
            Binarise X before fitting.

        Returns
        -------
        W : csr_matrix, shape (n_items, n_items)
        """
        self.alpha = alpha
        self.topK = topK

        if implicit:
            X = (X > 0).astype(np.float32)

        X = csr_matrix(X)
        n_users, n_items = X.shape

        user_degree = np.array(X.sum(axis=1)).flatten()
        item_degree = np.array(X.sum(axis=0)).flatten()
        user_degree[user_degree == 0] = 1.0
        item_degree[item_degree == 0] = 1.0

        user_degree_inv = np.power(user_degree, -1.0)
        D_user_inv = sps.diags(user_degree_inv)

        item_degree_alpha_inv = np.power(item_degree, -alpha)
        D_item_alpha_inv = sps.diags(item_degree_alpha_inv)

        # W = D_item_alpha_inv * X^T * D_user_inv * X
        Xt_normalized = X.T.dot(D_user_inv)
        W = D_item_alpha_inv.dot(Xt_normalized).dot(X)

        # Zero diagonal
        W = W.tolil()
        W.setdiag(0)
        W = W.tocsr()

        W = self._top_k_sparse(W, topK)

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
                cols_i = row.indices
                data_i = row.data
            else:
                top_idx = np.argpartition(row.data, -k)[-k:]
                cols_i = row.indices[top_idx]
                data_i = row.data[top_idx]

            rows.extend([i] * len(cols_i))
            cols.extend(cols_i)
            data.extend(data_i)

        return csr_matrix((data, (rows, cols)), shape=matrix.shape)
