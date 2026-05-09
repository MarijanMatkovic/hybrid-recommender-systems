"""
LazyPred: defer the dense ``X @ B`` materialisation for the prediction
matrix when (n_users x n_items) is too large to fit in memory.

Background
----------
EASE-family models (EASE, EDLAE, Lap-EASE, GS-EASE, SLIM, Hybrid-Matrix,
Hybrid-GraphReg, Hybrid-Laplacian) all set ``self.pred = X @ B`` after
fitting. ``X @ B`` is a *dense* (n_users, n_items) ndarray because B is
dense. On MovieLens-1M that's 6040 * 3416 * 8 bytes ~= 165 MB (fine).
On Netflix Prize that's 429584 * 17764 * 8 bytes ~= 61 GB, which OOMs
every job on a 64 GB node.

The downstream evaluation (``evaluate_at_ks``, ``bucketed_metrics_at_ks``,
``head_tail_analysis._evaluate_by_bucket``) only ever accesses the pred
matrix one user-row at a time:

    scores = pred[user_idx, :].copy()

So we don't need to materialise it -- we can compute ``X[u] @ B`` on
demand. ``LazyPred`` wraps ``(X, B)`` and exposes the
``pred[user_idx, :]`` access pattern without storing the full product.

Trade-off
---------
For small datasets where the dense product fits comfortably, the eager
matmul (single BLAS call) is slightly faster than per-row computation
(many small BLAS calls in a Python loop). The ``make_pred`` helper
auto-switches: above ``LAZY_PRED_THRESHOLD`` cells we go lazy, below it
we keep the existing eager behaviour.

For Netflix Prize specifically, lazy is also *faster* in practice
because the per-row ``X[u] @ B`` exploits sparsity of X[u] (~200 nnz)
fully, whereas the eager dense product wastes work on zero rows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps


# Threshold (in # of cells) above which `make_pred` returns a LazyPred
# instead of materialising X @ B. 5e8 cells * 8 bytes = 4 GB at float64.
# This keeps ml-1m (~2e7 cells, ~165 MB) on the eager path and Netflix
# Prize (~7.6e9 cells, ~61 GB) on the lazy path.
LAZY_PRED_THRESHOLD = 5e8


class LazyPred:
    """Deferred ``X @ B``. Behaves as a 2D-ndarray-like for per-user-row
    indexing -- the only access pattern used by the evaluation code.

    Notably this class deliberately does **not** define a ``toarray``
    method. The downstream evaluation does
    ``if hasattr(pred, 'toarray'): pred = pred.toarray()`` to convert
    sparse predictions to dense; on Netflix this would force the 61 GB
    materialisation we're trying to avoid. Without ``toarray`` the
    branch is skipped and the loop falls through to per-row indexing,
    which uses ``__getitem__`` below.
    """

    __slots__ = ('X', 'B', 'shape', 'dtype')

    def __init__(self, X, B):
        self.X = X
        self.B = B
        n_users = X.shape[0]
        if hasattr(B, 'ndim') and B.ndim == 2:
            n_items = B.shape[1]
        else:
            # Fallback for 1D or unusual B; not really supported.
            n_items = B.shape[-1]
        self.shape = (int(n_users), int(n_items))
        self.dtype = np.result_type(
            getattr(X, 'dtype', np.float64),
            getattr(B, 'dtype', np.float64))

    def _row(self, idx):
        """Compute one user-row of pred = X @ B without materialising."""
        row_x = self.X[int(idx)]
        if sps.issparse(row_x):
            # Single sparse row -> dense 1D ndarray
            row_x = np.asarray(row_x.toarray()).ravel()
        else:
            row_x = np.asarray(row_x).ravel()
        # Now row_x is (n_items,) dense; B is (n_items, n_items) dense.
        return np.asarray(row_x @ self.B)

    def __getitem__(self, key):
        # Most common pattern: pred[user_idx, :]
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(row, (int, np.integer)):
                full = self._row(int(row))
                if isinstance(col, slice) and (
                    col.start is None and col.stop is None
                    and col.step is None
                ):
                    return full
                return full[col]
        # Single integer key: row access
        if isinstance(key, (int, np.integer)):
            return self._row(int(key))
        raise IndexError(
            f"LazyPred supports [user_idx] or [user_idx, :] indexing only; "
            f"got {key!r}")

    def __array__(self, dtype=None):
        # Last-resort materialisation; avoid on Netflix.
        out = self.X @ self.B
        if sps.issparse(out):
            out = out.toarray()
        out = np.asarray(out)
        return out.astype(dtype) if dtype is not None else out


def make_pred(X, B):
    """Return ``X @ B`` either eagerly (small) or as ``LazyPred`` (big).

    Auto-selects based on the size of the resulting matrix. For small
    datasets (ml-small, ml-1m) the eager path is unchanged; for Netflix
    Prize the lazy wrapper avoids a 61 GB materialisation.

    Parameters
    ----------
    X : sparse user-item matrix (n_users, n_items)
    B : dense item-item matrix (n_items, n_items)

    Returns
    -------
    Either a dense numpy ndarray (eager) or a ``LazyPred`` instance
    (lazy). Both support the ``arr[user_idx, :]`` access pattern that the
    evaluation code relies on.
    """
    n_users = X.shape[0]
    if hasattr(B, 'ndim') and B.ndim == 2:
        n_items = B.shape[1]
    else:
        n_items = B.shape[-1]
    if n_users * n_items > LAZY_PRED_THRESHOLD:
        return LazyPred(X, B)
    out = X @ B
    if sps.issparse(out):
        out = out.toarray()
    return np.asarray(out)
