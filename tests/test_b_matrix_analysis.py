"""
Unit tests for ``experiments.b_matrix_analysis``.

We exercise the three diagnostic helpers in isolation on handcrafted
matrices so the correctness guarantees are unambiguous:

  * ``sparsity_stats``        -- thresholded-mass counting excludes the
                                 diagonal and tracks several thresholds
                                 correctly.
  * ``graph_alignment_stats`` -- perfectly aligned / perfectly misaligned
                                 B vs W give expected extremes; the
                                 per-row Jaccard correctly pulls ranking
                                 overlap out.
  * ``condition_number_stats`` -- on a known diagonal matrix the answer
                                  matches the analytical ratio.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from experiments.b_matrix_analysis import (
    condition_number_stats,
    graph_alignment_stats,
    sparsity_stats,
    _topk_support_per_row,
)


# ---------------------------------------------------------------------------
# sparsity_stats
# ---------------------------------------------------------------------------

def test_sparsity_stats_excludes_diagonal():
    """Giant values on the diagonal must not contaminate the stats."""
    n = 5
    B = np.zeros((n, n))
    # Off-diagonal: one entry at 0.5, rest zero.
    B[0, 1] = 0.5
    # Diagonal full of huge values (should be ignored).
    np.fill_diagonal(B, 1e6)
    sp = sparsity_stats(B, thresholds=(1e-3, 1e-1))
    assert sp['B_off_diag_n_entries'] == n * n - n    # 20 off-diagonal entries
    assert sp['B_max_abs'] == 0.5                      # NOT 1e6
    assert sp['B_l1'] == 0.5
    # Exactly 1 entry above 1e-3 and 1 above 1e-1.
    assert sp['B_frac_above_0.001'] == 1.0 / 20
    assert sp['B_frac_above_0.1'] == 1.0 / 20


def test_sparsity_stats_thresholds_monotone():
    """Fraction above threshold must be non-increasing as threshold grows."""
    rng = np.random.default_rng(0)
    B = rng.standard_normal((8, 8)) * 0.1
    np.fill_diagonal(B, 0)
    sp = sparsity_stats(B, thresholds=(1e-4, 1e-3, 1e-2, 1e-1, 1.0))
    fracs = [sp[f'B_frac_above_{t:g}']
             for t in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    for a, b in zip(fracs, fracs[1:]):
        assert a >= b


# ---------------------------------------------------------------------------
# graph_alignment_stats
# ---------------------------------------------------------------------------

def test_graph_alignment_perfect_agreement():
    """If |B| == W (up to scale), Jaccard and Pearson should be ~1."""
    rng = np.random.default_rng(7)
    n = 12
    W_dense = np.abs(rng.standard_normal((n, n)))
    np.fill_diagonal(W_dense, 0)
    B = 0.3 * W_dense.copy()   # same support and ranking
    out = graph_alignment_stats(B, csr_matrix(W_dense), topK=4)
    # Pearson between |B| and W is exactly 1 (linear with positive coef),
    # modulo float round-off.
    assert out['align_pearson_absB_W'] == pytest.approx(1.0, abs=1e-12)
    # Same top-K per row -> Jaccard = 1.0 on every row.
    assert out['align_jaccard_topK_mean'] == pytest.approx(1.0, abs=1e-12)
    # No zero W entries with this random matrix, so the ratio is 1/1 = inf
    # OR equal means. In practice all entries are W-edges here (dense
    # random > 0), so b_off_edge = 0 and the ratio is inf.
    assert np.isinf(out['align_edge_to_nonedge_ratio']) or \
           out['align_edge_to_nonedge_ratio'] > 1.0


def test_graph_alignment_antagonistic_support():
    """If B's support lives where W is zero, the edge-magnitude ratio
    collapses to 0: B has zero mass on W's edges."""
    n = 12
    W_dense = np.zeros((n, n))
    # W has edges on the upper triangle only (excluding diagonal).
    for i in range(n):
        for j in range(i + 1, n):
            W_dense[i, j] = 1.0
    B = np.zeros((n, n))
    # B has strictly positive (and all-different) mass on the strict
    # lower triangle, zero elsewhere. Using distinct values prevents
    # ties from causing argpartition to fall back on arbitrary columns.
    v = 0.0
    for i in range(n):
        for j in range(i):
            v += 1.0
            B[i, j] = v
    out = graph_alignment_stats(B, csr_matrix(W_dense), topK=3)
    # B has ZERO mass on W-edges -- that's the unambiguous diagnostic.
    assert out['align_mean_absB_on_W_edge'] == 0.0
    # B's mass on non-edges is strictly positive.
    assert out['align_mean_absB_off_W_edge'] > 0.0
    # Edge-to-nonedge ratio is exactly 0 (numerator is zero).
    assert out['align_edge_to_nonedge_ratio'] == 0.0


def test_graph_alignment_accepts_sparse_W():
    """CSR input path must match dense input path."""
    n = 6
    rng = np.random.default_rng(3)
    W_dense = rng.uniform(0, 1, size=(n, n))
    np.fill_diagonal(W_dense, 0)
    B = rng.standard_normal((n, n)) * 0.1
    np.fill_diagonal(B, 0)
    out_dense = graph_alignment_stats(B, W_dense, topK=3)
    out_sparse = graph_alignment_stats(B, csr_matrix(W_dense), topK=3)
    for key in out_dense:
        if isinstance(out_dense[key], float) and np.isnan(out_dense[key]):
            assert np.isnan(out_sparse[key])
        else:
            assert out_dense[key] == out_sparse[key]


def test_topk_support_per_row_never_flags_diagonal():
    """The top-K support must never include the self-loop."""
    n = 8
    M = np.eye(n) * 100.0   # diagonal is HUGE
    M += np.random.default_rng(0).standard_normal((n, n)) * 0.01
    mask = _topk_support_per_row(M, k=2)
    # Diagonal must be all False despite being the largest entry.
    assert not mask.diagonal().any()
    # Exactly K entries per row.
    assert (mask.sum(axis=1) == 2).all()


# ---------------------------------------------------------------------------
# condition_number_stats
# ---------------------------------------------------------------------------

def test_condition_number_matches_analytical_for_diagonal():
    """When G, L are diagonal we can predict cond exactly."""
    n = 5
    G_raw = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
    L = np.diag([0.0, 0.5, 1.0, 2.0, 3.0])
    lam = 0.5
    gamma = 2.0
    out = condition_number_stats(G_raw, lam=lam, gamma=gamma, L_scaled=L)
    # Diagonal of A = 1 + 0.5 + 2 * [0, 0.5, 1, 2, 3] = [1.5, 2.5, 3.5, 5.5, 7.5]
    expected_cond = 7.5 / 1.5
    assert abs(out['cond_number'] - expected_cond) < 1e-9
    assert abs(out['eig_max'] - 7.5) < 1e-9
    assert abs(out['eig_min'] - 1.5) < 1e-9


def test_condition_number_no_laplacian_path():
    """Passing ``L_scaled=None`` or ``gamma=0`` must give the EASE-only
    condition number."""
    G_raw = np.diag([2.0, 3.0, 4.0])
    lam = 1.0
    out_a = condition_number_stats(G_raw, lam=lam, gamma=0.0,
                                   L_scaled=np.eye(3))
    out_b = condition_number_stats(G_raw, lam=lam, gamma=10.0,
                                   L_scaled=None)
    # Both paths collapse to cond([3, 4, 5]) = 5/3.
    expected = 5.0 / 3.0
    assert abs(out_a['cond_number'] - expected) < 1e-9
    assert abs(out_b['cond_number'] - expected) < 1e-9
