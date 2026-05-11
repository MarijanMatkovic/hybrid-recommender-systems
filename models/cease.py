"""
CEASE and Add-EASE -- closed-form linear autoencoders that incorporate
side information (item 2 of the closed-form roadmap).

References
----------
Jeunen, Van Balen & Goethals -- "Closed-Form Models for Collaborative
Filtering with Side-Information", RecSys 2020.

Both models reuse the EASE closed-form solution; the only difference is
how the side-information matrix T is folded in:

    CEASE   : G = X^T X + alpha_T * T^T T  (single weighted ridge)
    Add-EASE: B = alpha * B_X + (1-alpha) * B_T
              where B_X is plain EASE on X
              and   B_T is plain EASE on T (treats tags as virtual users)

For MovieLens, T is the (n_genres, n_items) binary genre indicator
matrix. ``build_genre_tag_matrix`` constructs it from the standard
movies.dat / movies.csv layout.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from .lazy_pred import make_pred


def _ease_solve(G, lambda_):
    """Standard EASE closed-form: B = P / (-diag(P)) with diag(B)=0."""
    n = G.shape[0]
    A = G + lambda_ * np.eye(n)
    P = np.linalg.inv(A)
    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.0)
    return B


class CEASE:
    """
    CEASE: stack tags T (n_tags x n_items) on top of X (n_users x
    n_items) and solve a *single* weighted EASE on the stacked Gram

        G = X^T X + alpha_T * T^T T

    alpha_T = 0 reduces to vanilla EASE; alpha_T -> infty pushes toward
    a tag-only model. Jeunen et al. recommend alpha_T near 1.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def fit(self, df, T, lambda_=500.0, alpha_T=1.0, implicit=True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df)) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n_items = X.shape[1]
        if T.shape[1] != n_items:
            raise ValueError(
                f"T must have {n_items} columns; got T.shape={T.shape}")

        G = X.T.dot(X)
        if sps.issparse(G):
            G = G.toarray()
        TT = T.T.dot(T)
        if sps.issparse(TT):
            TT = TT.toarray()

        G_aug = G + alpha_T * TT
        B = _ease_solve(G_aug, lambda_)
        self.B = B
        self.alpha_T = alpha_T
        self.pred = make_pred(X, B)
        self.ease = self  # so evaluate_at_ks's model.ease.user_enc works
        return B, X


class AddEASE:
    """
    Add-EASE: train two EASEs (one on X, one on T) independently and
    combine with a scalar mixing weight:

        B_X     = EASE(X, lambda_X)
        B_T     = EASE(T, lambda_T)
        B_final = alpha * B_X + (1 - alpha) * B_T

    More expressive than CEASE because the two sources get different
    L2 strengths.
    """

    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.B_X = None
        self.B_T = None

    def fit(self, df, T, lambda_X=500.0, lambda_T=10.0, alpha=0.5,
            implicit=True):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['item_id'])
        values = (np.ones(len(df)) if implicit
                  else df['rating'].to_numpy() / df['rating'].max())
        X = csr_matrix((values, (users, items)))
        self.X = X
        n_items = X.shape[1]
        if T.shape[1] != n_items:
            raise ValueError(
                f"T must have {n_items} columns; got T.shape={T.shape}")

        # B_X via plain EASE
        G = X.T.dot(X).toarray()
        B_X = _ease_solve(G, lambda_X)

        # B_T via plain EASE on tags-as-virtual-users
        TT = T.T.dot(T)
        if sps.issparse(TT):
            TT = TT.toarray()
        B_T = _ease_solve(TT, lambda_T)

        self.B_X = B_X
        self.B_T = B_T
        self.alpha = alpha
        B = alpha * B_X + (1.0 - alpha) * B_T
        np.fill_diagonal(B, 0.0)
        self.B = B
        self.pred = make_pred(X, B)
        self.ease = self  # so evaluate_at_ks's model.ease.user_enc works
        return B, X


# ---------------------------------------------------------------------------
# Helpers: build a (n_genres, n_items) tag matrix from MovieLens metadata.
# ---------------------------------------------------------------------------

def build_genre_tag_matrix(movies_df, item_enc, id_column='movieId'):
    """Construct a binary (n_genres, n_items) genre tag matrix.

    Parameters
    ----------
    movies_df : pd.DataFrame
        Must have columns [id_column, 'genres'] where genres are
        pipe-separated (ML-1M / ML-small style).
    item_enc : sklearn LabelEncoder
        Fit on the same item ids used for the X matrix; only items in
        ``item_enc.classes_`` get a column in T.

    Returns
    -------
    T       : sparse (n_genres, n_items)
    genres  : list of genre names (row labels)
    """
    all_genres = set()
    for g in movies_df['genres'].dropna():
        all_genres.update(g.split('|'))
    if '(no genres listed)' in all_genres:
        all_genres.discard('(no genres listed)')
    genres = sorted(all_genres)
    g_idx = {g: i for i, g in enumerate(genres)}
    item_set = set(item_enc.classes_)

    rows, cols = [], []
    for _, row in movies_df.iterrows():
        movie_id = row[id_column]
        if movie_id not in item_set:
            continue
        col = int(item_enc.transform([movie_id])[0])
        for g in str(row['genres']).split('|'):
            if g in g_idx:
                rows.append(g_idx[g])
                cols.append(col)

    n_genres = len(genres)
    n_items = len(item_enc.classes_)
    if not rows:
        return csr_matrix((n_genres, n_items)), genres
    T = csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)),
        shape=(n_genres, n_items))
    return T, genres
