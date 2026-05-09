import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .lazy_pred import make_pred


class EASE:
    """
    Embarrassingly Shallow Autoencoder (Steck, 2019).
    Learns a dense item-item weight matrix B via closed-form L2-regularized
    regression: B = P / (-diag(P)), where P = (X^T X + λI)^{-1}.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.item_enc.fit_transform(df.loc[:, 'item_id'])
        return users, items

    def fit(self, df, lambda_: float = 50, implicit=True):
        """
        Fit EASE model.

        Returns
        -------
        B: np.ndarray, shape (n_items, n_items)
            The item-item weight matrix.
        X: csr_matrix, shape (n_users, n_items)
            The user-item interaction matrix.
        """
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        # Core EASE computation
        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = make_pred(X, B)
        return B, X

    def predict_for_user(self, user_idx, watched_set, score_vector, candidate_items, k):
        """Generate top-k predictions for a single user."""
        candidates = [item for item in candidate_items if item not in watched_set]
        pred = np.take(score_vector, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame({
            "user_id": [user_idx] * len(res),
            "item_id": np.take(candidates, res),
            "score": np.take(pred, res),
        }).sort_values('score', ascending=False)
        return r