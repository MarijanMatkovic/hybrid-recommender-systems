"""
Tests that the extended HybridEASE_RP3beta accepts every graph source
for the Laplacian method and continues to produce sane predictions.
"""

import numpy as np
import pytest

from models import HybridEASE_RP3beta
from models.graph_sources import VALID_SOURCES


@pytest.mark.parametrize('source', list(VALID_SOURCES))
def test_hybrid_laplacian_graph_sources(tiny_split, source):
    train, _ = tiny_split
    model = HybridEASE_RP3beta()
    model.fit(train, method='laplacian',
              ease_lambda=50.0, rp3_alpha=1.0, rp3_beta=0.6,
              rp3_topK=20, graph_reg_gamma=1.0,
              graph_source=source)
    # B matrix sanity
    B = model.ease.B
    assert np.allclose(np.diag(B), 0.0)
    assert np.isfinite(B).all()
    # predictions non-empty
    assert model.pred.shape[0] == model.ease.X.shape[0]


def test_hybrid_laplacian_gamma_zero_matches_ease(tiny_split):
    """gamma=0 → Laplacian-EASE reduces to plain EASE."""
    from models import EASE

    train, _ = tiny_split

    ease = EASE()
    B_ease, _ = ease.fit(train, lambda_=50.0, implicit=True)

    h = HybridEASE_RP3beta()
    h.fit(train, method='laplacian', ease_lambda=50.0,
          rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
          graph_reg_gamma=0.0, graph_source='rp3beta')

    assert np.allclose(B_ease, h.ease.B, atol=1e-7)


def test_hybrid_laplacian_rp3beta_vs_binary_differ(tiny_split):
    """Different graph sources should lead to different B matrices."""
    train, _ = tiny_split

    h_rp3 = HybridEASE_RP3beta()
    h_rp3.fit(train, method='laplacian', ease_lambda=50.0,
              rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
              graph_reg_gamma=5.0, graph_source='rp3beta')

    h_bin = HybridEASE_RP3beta()
    h_bin.fit(train, method='laplacian', ease_lambda=50.0,
              rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
              graph_reg_gamma=5.0, graph_source='binary')

    assert np.abs(h_rp3.ease.B - h_bin.ease.B).sum() > 1e-6


def test_hybrid_invalid_graph_source_raises(tiny_split):
    train, _ = tiny_split
    model = HybridEASE_RP3beta()
    with pytest.raises(ValueError):
        model.fit(train, method='laplacian', ease_lambda=50.0,
                  rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
                  graph_reg_gamma=1.0, graph_source='unknown')


def test_hybrid_laplacian_sym_variant_runs(tiny_split):
    """End-to-end smoke test: laplacian_normalise='sym' should train
    without error and produce a finite B with zero diagonal.
    """
    train, _ = tiny_split
    model = HybridEASE_RP3beta()
    model.fit(train, method='laplacian', ease_lambda=50.0,
              rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
              graph_reg_gamma=5.0, graph_source='rp3beta',
              laplacian_normalise='sym')
    B = model.ease.B
    assert np.isfinite(B).all()
    assert np.allclose(np.diag(B), 0.0)


def test_hybrid_laplacian_sym_vs_none_differ(tiny_split):
    """With the same graph and gamma, 'sym' and 'none' normalisation
    should yield different solutions (the whole point of the variant).
    """
    train, _ = tiny_split
    common = dict(method='laplacian', ease_lambda=50.0,
                  rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
                  graph_reg_gamma=5.0, graph_source='rp3beta')

    h_none = HybridEASE_RP3beta()
    h_none.fit(train, laplacian_normalise='none', **common)

    h_sym = HybridEASE_RP3beta()
    h_sym.fit(train, laplacian_normalise='sym', **common)

    assert np.abs(h_none.ease.B - h_sym.ease.B).sum() > 1e-6


def test_hybrid_laplacian_invalid_normalise_raises(tiny_split):
    train, _ = tiny_split
    model = HybridEASE_RP3beta()
    with pytest.raises(ValueError):
        model.fit(train, method='laplacian', ease_lambda=50.0,
                  rp3_alpha=1.0, rp3_beta=0.6, rp3_topK=20,
                  graph_reg_gamma=1.0, graph_source='rp3beta',
                  laplacian_normalise='row')
