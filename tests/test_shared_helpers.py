"""
Unit tests for helpers in ``experiments._shared`` that are consumed by
multiple experiment scripts.

Focus: ``wilcoxon_vs_baseline`` — the convenience wrapper over
``wilcoxon_paired`` that takes two ``evaluate_at_ks``-shaped result dicts
and runs the paired signed-rank test on their per-user NDCG@k arrays.
Because it's called unconditionally inside every experiment loop it must
return a safe ``(nan, nan, 0)`` sentinel rather than raise whenever the
inputs don't line up.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments._shared import wilcoxon_paired, wilcoxon_vs_baseline


def _mk_result(k_to_vals, user_ids):
    """Build a minimal dict shaped like what ``evaluate_at_ks`` returns."""
    return {
        'per_user_ndcg': {k: np.asarray(v, dtype=float)
                          for k, v in k_to_vals.items()},
        'per_user_ids': list(user_ids),
    }


def test_wilcoxon_vs_baseline_happy_path():
    users = list(range(40))
    rng = np.random.default_rng(42)
    base_vals = rng.uniform(0.1, 0.5, size=40)
    mod_vals = base_vals + 0.03  # strictly-better model
    base = _mk_result({10: base_vals}, users)
    mod = _mk_result({10: mod_vals}, users)
    stat, p, n = wilcoxon_vs_baseline(base, mod, k=10)
    assert n == 40
    assert p < 0.01
    assert not np.isnan(stat)


def test_wilcoxon_vs_baseline_missing_k_returns_sentinel():
    base = _mk_result({10: [0.1, 0.2]}, [1, 2])
    mod = _mk_result({20: [0.3, 0.4]}, [1, 2])
    stat, p, n = wilcoxon_vs_baseline(base, mod, k=10)
    # baseline has k=10 but model doesn't.
    assert n == 0
    assert np.isnan(stat)
    assert np.isnan(p)


def test_wilcoxon_vs_baseline_missing_keys_in_dict():
    """If the result dicts don't have the per-user blocks (e.g. caller
    passed a summary-only dict), we silently return the sentinel instead
    of raising."""
    base = {'NDCG@10': 0.1}         # no per_user_* keys
    mod = _mk_result({10: [0.1]}, [1])
    stat, p, n = wilcoxon_vs_baseline(base, mod, k=10)
    assert n == 0
    assert np.isnan(p)


def test_wilcoxon_vs_baseline_agrees_with_paired_helper():
    """Sanity: the wrapper must produce the same numbers as calling
    ``wilcoxon_paired`` directly."""
    users = list(range(25))
    rng = np.random.default_rng(7)
    b = rng.uniform(0, 1, 25)
    m = b + rng.normal(0.02, 0.01, 25)
    base = _mk_result({10: b}, users)
    mod = _mk_result({10: m}, users)
    s1, p1, n1 = wilcoxon_vs_baseline(base, mod, k=10)
    s2, p2, n2 = wilcoxon_paired((b, users), (m, users))
    assert n1 == n2
    assert pytest.approx(p1) == p2
    assert pytest.approx(s1) == s2
