import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix

from models.ease import EASE
from models.rp3beta import RP3beta
from models.graph_sources import build_graph, build_laplacian


class HybridEASE_RP3beta:
    """
    Graph-Enhanced EASEr: Hybrid recommender combining EASE and RP3beta.

    Implements three fusion strategies:
      - 'Score': Score-level weighted ensemble (Level 1)
      - 'Matrix': Matrix-level fusion before prediction (Level 2)
      - 'Graph_reg': Graph-regularized EASEr objective (Level 3)
      - 'Laplacian': Laplacian-regularized EASEr objective (Level 4)

    For the Laplacian variant the graph source is configurable via
    ``graph_source`` (default 'rp3beta'); supported: rp3beta, p3alpha,
    itemknn, binary.
    """

    def __init__(self):
        self.ease = EASE()
        self.rp3 = RP3beta()

    def fit(self, df, method='matrix',
            # EASE params
            ease_lambda=50, implicit=True,  #ease_lambda=0.5
            # RP3beta params
            rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=100,
            # Hybrid params
            fusion_alpha=0.5,
            # Graph-reg params (Level 3 only)
            graph_reg_gamma=0.0001,   #graph_reg_gamma = 0.1
            # Graph source (Laplacian only)
            graph_source='rp3beta',
            p3_alpha=1.0,
            itemknn_shrink=0.0,
            # Laplacian normalisation (Laplacian only): 'none' or 'sym'
            laplacian_normalise='none'):
        """
        Fit the hybrid model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with columns ['user_id', 'item_id', 'rating' (optional)]
        method : str
            'score' | 'matrix' | 'graph_reg'
        fusion_alpha : float
            Weight for an EASE component. (1 - fusion_alpha) = weight for RP3beta.
            Range [0, 1]. Tuned via validation.
        graph_reg_gamma : float
            Strength of graph regularization (Level 3 only).
            :param method:
            :param df:
            :param graph_reg_gamma:
            :param fusion_alpha:
            :param rp3_topK:
            :param rp3_beta:
            :param rp3_alpha:
            :param implicit:
            :param ease_lambda:
        """
        self.method = method
        self.fusion_alpha = fusion_alpha

        # --- Fit EASE ---
        if method == 'graph_reg':
            B, X = self._fit_graph_regularized_ease(
                df, ease_lambda, implicit,
                rp3_alpha, rp3_beta, rp3_topK,
                graph_reg_gamma
            )
        elif method == 'laplacian':
            B, X = self._fit_laplacian_ease(
                df, ease_lambda, implicit,
                rp3_alpha, rp3_beta, rp3_topK,
                graph_reg_gamma,
                graph_source=graph_source,
                p3_alpha=p3_alpha,
                itemknn_shrink=itemknn_shrink,
                laplacian_normalise=laplacian_normalise,
            )
        else:
            B, X = self.ease.fit(df, lambda_=ease_lambda, implicit=implicit)

        # --- Fit RP3beta on the same interaction matrix ---
        # Skipped for the Laplacian path -- the graph was already built
        # from the configured source inside _fit_laplacian_ease.
        if method != 'laplacian':
            W = self.rp3.fit(
                X, alpha=rp3_alpha, beta=rp3_beta,
                topK=rp3_topK, implicit=implicit
            )
        else:
            # The Laplacian path stores the built graph on
            # ``_fit_laplacian_ease``; fall back to a plain rp3beta fit
            # if the Laplacian branch used a different source and we
            # still need a W for downstream analysis.
            W = getattr(self.rp3, 'W', None)
            if W is None:
                W = self.rp3.fit(
                    X, alpha=rp3_alpha, beta=rp3_beta,
                    topK=rp3_topK, implicit=implicit,
                )

        n_items = X.shape[1]

        # --- Compute predictions based on method ---
        if method == 'score':
            # Level 1: Score-level ensemble
            # pred = α * (X @ B) + (1-α) * (X @ W)
            pred_ease = X.dot(B)
            pred_rp3 = X.dot(W).toarray()

            # Normalize both score matrices to [0, 1] range for a fair combination
            pred_ease_norm = self._min_max_normalize(pred_ease)
            pred_rp3_norm = self._min_max_normalize(pred_rp3)

            self.pred = fusion_alpha * pred_ease_norm + (1 - fusion_alpha) * pred_rp3_norm

        elif method == 'matrix':
            # Level 2: Matrix-level fusion
            # S = α * normalize(B) + (1-α) * normalize(W_dense)
            # Then pred = X @ S

            # Normalize B (L2 row-normalize so scales are comparable)
            B_norm = self._row_normalize(B)
            W_dense = W.toarray()
            W_norm = self._row_normalize(W_dense)

            # Fuse the item-item matrices
            S = fusion_alpha * B_norm + (1 - fusion_alpha) * W_norm

            # Zero diagonal (no self-recommendation)
            np.fill_diagonal(S, 0)

            self.S = S  # Store for analysis/interpretability
            self.pred = X.dot(S)

        elif method == 'graph_reg':
            # Level 3: Already computed B via graph-regularized objective
            # RP3beta was still fitted for analysis purposes
            self.pred = X.dot(B)

        elif method == 'laplacian':
            # Level 4: Already computed B via Laplacian-regularized objective
            self.pred = X.dot(B)

        else:
            raise ValueError(f"Unknown method: {method}")

        self.X = X
        return self

    def _fit_graph_regularized_ease(self, df, lambda_, implicit,
                                     rp3_alpha, rp3_beta, rp3_topK,
                                     gamma):
        """
        Level 3: Graph-Regularized EASE.

        Modified objective:
            min_B ||X - XB||^2_F + λ||B||^2_F + γ||B - W||^2_F
                s.t. diag(B) = 0

        Where W is RP3beta's similarity matrix (converted to dense).

        The graph regularization term γ||B - W||^2_F encourages EASE's
        weight matrix to be close to the graph-based similarities, effectively
        injecting multi-hop structural information into the linear model.

        Closed-form solution:
            The modified Gram matrix becomes:
            G' = X^T X + (λ + γ)I
            And the solution becomes:
            P = (G')^{-1}
            But we need to account for the W term. The full derivation:

            ∂L/∂B = -2 X^T(X - XB) + 2λB + 2γ(B - W) = 0
            X^T X B + λB + γB = X^T X + γW
            (X^T X + (λ+γ)I) B = X^T X + γW
            B_unconstrained = (X^T X + (λ+γ)I)^{-1} (X^T X + γW)

            Then apply the diagonal constraint as in standard EASE.
        """
        # First, we need X to compute RP3beta
        users = self.ease.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.ease.item_enc.fit_transform(df.loc[:, 'item_id'])
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )
        X = csr_matrix((values, (users, items)))
        self.ease.X = X

        # Fit RP3beta to get W
        W_sparse = self.rp3.fit(
            X, alpha=rp3_alpha, beta=rp3_beta,
            topK=rp3_topK, implicit=implicit,
            normalize_similarity=False  # Don't L1-normalize for regularization
        )
        W = W_sparse.toarray()

        # Scale W to be comparable to G_raw's scale in the RHS.
        # RHS = G_raw + γ * W_scaled, so for gamma to have meaningful effect,
        # W_scaled must have entries on the same order as G_raw.
        # Then gamma=0.1 means "10% graph influence" in the right-hand side.
        G_raw = X.T.dot(X).toarray()

        w_nonzero_mean = np.mean(np.abs(W[W != 0])) if np.any(W != 0) else 1.0
        g_nonzero_mean = np.mean(np.abs(G_raw[G_raw != 0])) if np.any(G_raw != 0) else 1.0
        w_scale = g_nonzero_mean / (w_nonzero_mean + 1e-10)
        W_scaled = W * w_scale

        # Modified Gram matrix
        G = G_raw.copy()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += (lambda_ + gamma)

        P = np.linalg.inv(G)

        # Right-hand side: X^T X + γ * W_scaled
        RHS = G_raw + gamma * W_scaled

        # Unconstrained solution
        B_unconstrained = P.dot(RHS)

        # Apply diagonal constraint via Lagrange multipliers.
        # From the Lagrangian derivation:
        #   B_ij = B_unc_ij - P_ij * μ_j   where μ_j = B_unc_jj / P_jj
        # IMPORTANT: the correction is COLUMN-wise (index j), not row-wise.
        # This matches standard EASE's B = P / (-diag(P)) which divides columns.
        diag_P = np.diag(P)
        diag_B_unc = np.diag(B_unconstrained)
        mu = diag_B_unc / diag_P  # shape (n_items,)

        B = B_unconstrained - P * mu[np.newaxis, :]  # broadcast along columns
        B[diagIndices] = 0

        self.ease.B = B
        self.ease.pred = X.dot(B)

        return B, X

    def _fit_laplacian_ease(self, df, lambda_, implicit,
                            rp3_alpha, rp3_beta, rp3_topK,
                            gamma,
                            graph_source='rp3beta',
                            p3_alpha=1.0,
                            itemknn_shrink=0.0,
                            laplacian_normalise='none'):
        """
        Level 4: Graph Laplacian-Regularized EASE.

        Modified objective:
            min_B ||X - XB||^2_F + λ||B||^2_F + γ · tr(B^T L B)
                s.t. diag(B) = 0

        where L is the graph Laplacian built from one of several item-item
        similarity sources (RP3beta, P3alpha, ItemKNN cosine, binary
        co-occurrence). The Laplacian variant is controlled by
        ``laplacian_normalise``:
          - 'none' :  L = D - W_sym  (combinatorial Laplacian)
          - 'sym'  :  L = I - D^{-1/2} W_sym D^{-1/2}  (symmetric
                      normalised; eigenvalues in [0, 2])

        Derivation:
            ∂L/∂B = -2 X^T(X - XB) + 2λB + 2γLB = 0
            (X^T X + λI + γL) B = X^T X

            Let G = X^T X, then:
            P = (G + λI + γL)^{-1}
            B_unconstrained = P · G

            Diagonal constraint (Lagrange multipliers, column-wise):
            μ_j = B_unc_jj / P_jj
            B_ij = B_unc_ij - P_ij · μ_j
            B_jj = 0
        """
        # Build X
        users = self.ease.user_enc.fit_transform(df.loc[:, 'user_id'])
        items = self.ease.item_enc.fit_transform(df.loc[:, 'item_id'])
        values = (
            np.ones(df.shape[0])
            if implicit
            else df['rating'].to_numpy() / df['rating'].max()
        )
        X = csr_matrix((values, (users, items)))
        self.ease.X = X
        n_items = X.shape[1]

        # Build the item-item similarity matrix from the chosen graph source.
        W_sparse = build_graph(
            X, source=graph_source, topK=rp3_topK,
            rp3_alpha=rp3_alpha, rp3_beta=rp3_beta,
            p3_alpha=p3_alpha,
            itemknn_shrink=itemknn_shrink,
            implicit=implicit,
        )
        # Expose the graph on ``self.rp3`` so downstream analysis code
        # (and the fit() dispatcher) can always read ``self.rp3.W`` even
        # when we fit a non-RP3beta graph.
        self.rp3.W = W_sparse
        if graph_source == 'rp3beta':
            self.rp3.alpha = rp3_alpha
            self.rp3.beta = rp3_beta
            self.rp3.topK = rp3_topK

        # Build the (sym or unnormalised) Laplacian via the shared helper
        # so the same code path is used for downstream SLIM/EDLAE models.
        L, degrees = build_laplacian(W_sparse, normalise=laplacian_normalise)

        # Scale L so that mean(diag(L)) ≈ mean(diag(G)). For the sym
        # variant diag(L)=1 on non-isolated items, so the scaling factor
        # collapses to g_diag_mean (much larger). Sweepers should expect
        # the optimal gamma to differ between normalisations.
        G_raw = X.T.dot(X).toarray()
        g_diag_mean = np.mean(np.diag(G_raw))
        l_diag_mean = np.mean(np.diag(L))
        if l_diag_mean > 0:
            L_scaled = L * (g_diag_mean / l_diag_mean)
        else:
            L_scaled = L

        self.L_scaled = L_scaled  # expose for downstream models (SLIM/EDLAE)
        self.laplacian_normalise = laplacian_normalise

        # Modified Gram matrix: G + λI + γL
        G = G_raw.copy()
        diagIndices = np.diag_indices(n_items)
        G[diagIndices] += lambda_
        G += gamma * L_scaled

        P = np.linalg.inv(G)

        B_unconstrained = P.dot(G_raw)

        diag_P = np.diag(P)
        diag_B_unc = np.diag(B_unconstrained)
        mu = diag_B_unc / diag_P

        B = B_unconstrained - P * mu[np.newaxis, :]
        B[diagIndices] = 0

        self.ease.B = B
        self.ease.pred = X.dot(B)

        return B, X

    @staticmethod
    def _min_max_normalize(matrix):
        """Normalize matrix values to [0, 1] range."""
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val == 0:
            return matrix
        return (matrix - min_val) / (max_val - min_val)

    @staticmethod
    def _row_normalize(matrix):
        """L2 row-normalize a dense matrix."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def predict(self, train_df, users, items, k):
        """
        Generate top-k recommendations for given users.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions (to exclude already-seen items).
        users : array-like
            User IDs to generate predictions for.
        items : array-like
            Candidate item IDs.
        k : int
            Number of recommendations per user.
        """
        items_enc = self.ease.item_enc.transform(items)
        dd = train_df.loc[train_df['user_id'].isin(users)].copy()
        dd['ci'] = self.ease.item_enc.transform(dd['item_id'])
        dd['cu'] = self.ease.user_enc.transform(dd['user_id'])

        pred_matrix = self.pred
        if sps.issparse(pred_matrix):
            pred_matrix = pred_matrix.toarray()

        results = []
        for user_enc, group in dd.groupby('cu'):
            watched = set(group['ci'])
            candidates = [item for item in items_enc if item not in watched]
            scores = np.take(pred_matrix[user_enc, :], candidates)
            top_k_idx = np.argpartition(scores, -k)[-k:]
            r = pd.DataFrame({
                "user_id": [user_enc] * len(top_k_idx),
                "item_id": np.take(candidates, top_k_idx),
                "score": np.take(scores, top_k_idx),
            }).sort_values('score', ascending=False)
            results.append(r)

        df = pd.concat(results)
        df['item_id'] = self.ease.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.ease.user_enc.inverse_transform(df['user_id'])
        return df

    def get_item_similarity_matrices(self):
        """
        Return the component matrices for analysis/visualization.
        Useful for interpretability analysis in your thesis.
        """
        result = {
            'ease_B': self.ease.B,
            'rp3_W': self.rp3.W.toarray() if sps.issparse(self.rp3.W) else self.rp3.W,
        }
        if hasattr(self, 'S'):
            result['fused_S'] = self.S
        return result