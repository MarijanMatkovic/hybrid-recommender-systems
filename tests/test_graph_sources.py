"""
Tests for the graph-source abstraction and the individual graph models
(P3alpha, ItemKNN, RP3beta in its graph-source role, binary co-occurrence).
"""

import numpy as np
import pytest
import scipy.sparse as sps

from models import P3alpha, ItemKNN, build_graph, build_laplacian
from models.graph_sources import VALID_SOURCES


def test_valid_sources_registered():
    assert set(VALID_SOURCES) == {'rp3beta', 'p3alpha',
                                  'itemknn', 'binary'}


@pytest.mark.parametrize('source', list(VALID_SOURCES))
def test_build_graph_each_source(tiny_X, source):
    W = build_graph(tiny_X, source=source, topK=10)
    assert sps.issparse(W)
    assert W.shape == (tiny_X.shape[1], tiny_X.shape[1])
    # Zero diagonal
    assert np.allclose(W.diagonal(), 0.0)
    # Respects topK per row
    for i in range(W.shape[0]):
        assert W.getrow(i).nnz <= 10


def test_p3alpha_shapes(tiny_X):
    model = P3alpha()
    W = model.fit(tiny_X, alpha=1.0, topK=5)
    assert W.shape == (tiny_X.shape[1], tiny_X.shape[1])
    assert np.allclose(W.diagonal(), 0.0)


def test_itemknn_cosine_properties(tiny_X):
    model = ItemKNN()
    W = model.fit(tiny_X, topK=5, shrink=0.0,
                  normalize_similarity=False)
    # Cosine similarities are in [0, 1] for non-negative inputs
    data = W.data
    assert data.min() >= -1e-6
    assert data.max() <= 1.0 + 1e-6


def test_binary_graph_values_are_one(tiny_X):
    W = build_graph(tiny_X, source='binary', topK=5)
    assert W.nnz > 0
    assert np.allclose(np.unique(W.data), [1.0])


def test_build_laplacian_psd(tiny_X):
    """Graph Laplacian must be symmetric positive-semidefinite."""
    W = build_graph(tiny_X, source='rp3beta', topK=10)
    L, degrees = build_laplacian(W)
    # Symmetric
    assert np.allclose(L, L.T)
    # PSD: all eigenvalues >= 0 up to numerical noise
    eigs = np.linalg.eigvalsh(L)
    assert eigs.min() >= -1e-6
    # Degrees >= 0
    assert (degrees >= -1e-12).all()


def test_rp3beta_vs_p3alpha_differ(tiny_X):
    """With beta > 0 RP3beta should differ from P3alpha on this data."""
    W_rp3 = build_graph(tiny_X, source='rp3beta', topK=10, rp3_beta=0.6)
    W_p3 = build_graph(tiny_X, source='p3alpha', topK=10)
    assert W_rp3.shape == W_p3.shape
    # Ensure the matrices aren't coincidentally identical
    diff = (W_rp3 - W_p3)
    assert np.abs(diff.data).sum() > 1e-8


def test_build_graph_unknown_source_raises(tiny_X):
    with pytest.raises(ValueError):
        build_graph(tiny_X, source='nonexistent')
