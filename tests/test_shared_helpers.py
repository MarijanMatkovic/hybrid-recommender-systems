"""
Unit tests for helpers in ``experiments._shared`` that are consumed by
multiple experiment scripts.

Focus: ``wilcoxon_vs_baseline`` — the convenience wrapper over
``wilcoxon_paired`` that takes two ``evaluate_at_ks``-shaped result dicts
and runs the paired signed-rank test on their per-user NDCG@k arrays.
Because it's called unconditionally inside every experiment loop it must
return a safe ``(nan, nan, 0, 0, nan, nan)`` sentinel rather than raise
whenever the inputs don't line up.

The helpers return a 6-tuple
``(stat, p, n_pairs, sign, mean_diff, median_diff)``.

Sign semantics: ``sign`` is derived from ``mean(model − baseline)`` (not
the median). Per-user NDCG is sparse (many users score exactly 0), so
the median difference is often 0 even when the Wilcoxon test strongly
rejects the null. Basing ``sign`` on the mean matches the aggregate
NDCG delta reported elsewhere and keeps the sign informative.

``mean_diff`` is the primary magnitude column (and the basis for the
sign); ``median_diff`` is kept for reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments._shared import (
    wilcoxon_blank_columns,
    wilcoxon_columns,
    wilcoxon_paired,
    wilcoxon_vs_baseline,
)


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
    stat, p, n, sign, mean_diff, median_diff = wilcoxon_vs_baseline(
        base, mod, k=10)
    assert n == 40
    assert p < 0.01
    assert not np.isnan(stat)
    # Model beats baseline uniformly, so sign must be +1 and
    # mean_diff ~ +0.03.
    assert sign == 1
    assert mean_diff == pytest.approx(0.03, abs=1e-12)
    assert median_diff == pytest.approx(0.03, abs=1e-12)


def test_wilcoxon_vs_baseline_sign_negative_when_model_loses():
    """Sign must be -1 when the model is uniformly worse — otherwise the
    p-value is ambiguous about direction."""
    users = list(range(30))
    rng = np.random.default_rng(5)
    base_vals = rng.uniform(0.2, 0.6, size=30)
    mod_vals = base_vals - 0.05   # strictly worse
    base = _mk_result({10: base_vals}, users)
    mod = _mk_result({10: mod_vals}, users)
    _, p, n, sign, mean_diff, median_diff = wilcoxon_vs_baseline(
        base, mod, k=10)
    assert n == 30
    assert p < 0.01
    assert sign == -1
    assert mean_diff == pytest.approx(-0.05, abs=1e-12)
    assert median_diff == pytest.approx(-0.05, abs=1e-12)


def test_wilcoxon_vs_baseline_sign_mean_based_not_median_based():
    """The regression this test guards against: sparse NDCG vectors
    (many zeros) often have ``median(diff)=0`` even when the Wilcoxon
    test is highly significant and the mean is clearly nonzero.

    Construct a case where the baseline is all zeros except a handful,
    and the model lifts some users slightly. Median(diff) is 0
    (majority tied at 0), but mean(diff) is clearly positive, so sign
    must be +1 (the old median-based sign would have reported 0 — the
    bug this test prevents)."""
    users = list(range(20))
    base_vals = np.zeros(20)
    mod_vals = np.zeros(20)
    mod_vals[:3] = 0.1   # 3/20 users get a boost, 17/20 tied at 0
    base = _mk_result({10: base_vals}, users)
    mod = _mk_result({10: mod_vals}, users)
    _, _, n, sign, mean_diff, median_diff = wilcoxon_vs_baseline(
        base, mod, k=10)
    assert n == 20
    assert median_diff == 0.0           # old sign semantics -> 0 here
    assert mean_diff == pytest.approx(0.3 / 20, abs=1e-12)
    # New sign semantics: mean is positive -> +1.
    assert sign == 1


def test_wilcoxon_vs_baseline_missing_k_returns_sentinel():
    base = _mk_result({10: [0.1, 0.2]}, [1, 2])
    mod = _mk_result({20: [0.3, 0.4]}, [1, 2])
    stat, p, n, sign, mean_diff, median_diff = wilcoxon_vs_baseline(
        base, mod, k=10)
    # baseline has k=10 but model doesn't.
    assert n == 0
    assert np.isnan(stat)
    assert np.isnan(p)
    assert sign == 0
    assert np.isnan(mean_diff)
    assert np.isnan(median_diff)


def test_wilcoxon_vs_baseline_missing_keys_in_dict():
    """If the result dicts don't have the per-user blocks (e.g. caller
    passed a summary-only dict), we silently return the sentinel instead
    of raising."""
    base = {'NDCG@10': 0.1}         # no per_user_* keys
    mod = _mk_result({10: [0.1]}, [1])
    stat, p, n, sign, mean_diff, median_diff = wilcoxon_vs_baseline(
        base, mod, k=10)
    assert n == 0
    assert np.isnan(p)
    assert sign == 0
    assert np.isnan(mean_diff)
    assert np.isnan(median_diff)


def test_wilcoxon_vs_baseline_agrees_with_paired_helper():
    """Sanity: the wrapper must produce the same numbers as calling
    ``wilcoxon_paired`` directly."""
    users = list(range(25))
    rng = np.random.default_rng(7)
    b = rng.uniform(0, 1, 25)
    m = b + rng.normal(0.02, 0.01, 25)
    base = _mk_result({10: b}, users)
    mod = _mk_result({10: m}, users)
    s1, p1, n1, sign1, md1, med1 = wilcoxon_vs_baseline(base, mod, k=10)
    s2, p2, n2, sign2, md2, med2 = wilcoxon_paired((b, users), (m, users))
    assert n1 == n2
    assert pytest.approx(p1) == p2
    assert pytest.approx(s1) == s2
    assert sign1 == sign2
    assert pytest.approx(md1) == md2
    assert pytest.approx(med1) == med2


def test_wilcoxon_columns_has_stable_schema():
    """``wilcoxon_columns`` / ``wilcoxon_blank_columns`` must always
    return the same 6 keys so every experiment CSV ends up with an
    identical Wilcoxon schema."""
    tup = (0.5, 0.02, 40, 1, 0.04, 0.03)
    cols = wilcoxon_columns('wilcoxon_vs_EASE', tup)
    assert set(cols) == {
        'wilcoxon_vs_EASE_stat',
        'wilcoxon_vs_EASE_p',
        'wilcoxon_vs_EASE_n_pairs',
        'wilcoxon_vs_EASE_sign',
        'wilcoxon_vs_EASE_mean_diff',
        'wilcoxon_vs_EASE_median_diff',
    }
    assert cols['wilcoxon_vs_EASE_sign'] == 1
    assert cols['wilcoxon_vs_EASE_mean_diff'] == 0.04
    assert cols['wilcoxon_vs_EASE_median_diff'] == 0.03

    blank = wilcoxon_blank_columns('wilcoxon_vs_EASE')
    assert set(blank) == set(cols)
    assert blank['wilcoxon_vs_EASE_n_pairs'] == 0
    assert blank['wilcoxon_vs_EASE_sign'] == 0
    assert np.isnan(blank['wilcoxon_vs_EASE_p'])
    assert np.isnan(blank['wilcoxon_vs_EASE_mean_diff'])
    assert np.isnan(blank['wilcoxon_vs_EASE_median_diff'])
