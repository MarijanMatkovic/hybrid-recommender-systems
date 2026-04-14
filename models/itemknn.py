import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemKNN:
    """
    Item-Item k-nearest-neighbours with cosine similarity.

    Similarity:
        W[i, j] = (x_i . x_j) / (||x_i|| * ||x_j|| + shrink)

    Top-K sparsification is applied per row.
    """

    def __init__(self):
        pass

    def fit(self, X, topK=100, shrink=0.0, implicit=True,
            normalize_similarity=True):
        """
        Fit ItemKNN cosine similarity.

        Parameters
        ----------
        X : csr_matrix, shape (n_users, n_items)
        topK : int
            Keep top-K neighbours per item.
        shrink : float
            Shrinkage term in the denominator (reduces influence of item
            pairs with very few common users).
        implicit : bool
        normalize_similarity : bool
            L1-normalize rows after sparsification (for score-level use).

        Returns
        -------
        W : csr_matrix, shape (n_items, n_items)
        """
        self.topK = topK
        self.shrink = shrink

        if implicit:
            X = (X > 0).astype(np.float32)

        X = csr_matrix(X)
        n_users, n_items = X.shape

        # ||x_i||_2 per item (column)
        item_norms = np.sqrt(np.array(X.multiply(X).sum(axis=0)).flatten())
        item_norms[item_norms == 0] = 1.0

        # Raw dot product: X^T X gives (n_items x n_items) with entries = sum over users of X[u,i]*X[u,j]
        G = X.T.dot(X).toarray()

        # Outer product of norms for cosine denominator
        denom = np.outer(item_norms, item_norms) + shrink
        denom[denom == 0] = 1.0

        W = G / denom

        # Zero the diagonal (no self-similarity)
        np.fill_diagonal(W, 0.0)

        # Top-K per row
        W_sparse = self._top_k_dense(W, topK)

        if normalize_similarity:
            W_sparse = normalize(W_sparse, norm='l1', axis=1)

        self.W = W_sparse
        return W_sparse

    @staticmethod
    def _top_k_dense(W, k):
        """Keep only top-k values per row in a dense matrix, return sparse."""
        n_rows = W.shape[0]
        rows, cols, data = [], [], []

        for i in range(n_rows):
            row = W[i]
            nz_mask = row > 0
            if not nz_mask.any():
                continue
            nz_idx = np.where(nz_mask)[0]
            nz_vals = row[nz_idx]

            if len(nz_vals) <= k:
                cols_i = nz_idx
                data_i = nz_vals
            else:
                top = np.argpartition(nz_vals, -k)[-k:]
                cols_i = nz_idx[top]
                data_i = nz_vals[top]

            rows.extend([i] * len(cols_i))
            cols.extend(cols_i)
            data.extend(data_i)

        return csr_matrix((data, (rows, cols)), shape=W.shape)
