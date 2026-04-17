"""
Shared helpers for the weekly experiment scripts.

Every script accepts ``--dataset`` (``ml-small`` | ``ml-1m``) and writes
results/plots under ``results/<experiment_name>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_WILCOXON = True
except Exception:  # pragma: no cover
    _HAVE_WILCOXON = False

from data import (
    load_movielens,
    load_movielens_1m,
    temporal_train_test_split,
    random_train_test_split,
)


DEFAULT_RESULTS_DIR = Path('results')


def wilcoxon_paired(baseline_per_user, model_per_user):
    """Paired Wilcoxon signed-rank on per-user NDCG, aligned by user_id.

    Parameters
    ----------
    baseline_per_user : tuple[np.ndarray, list]
        ``(values, user_ids)`` for the baseline model (e.g. EASE).
    model_per_user : tuple[np.ndarray, list]
        ``(values, user_ids)`` for the compared model (e.g. Laplacian-EASE).

    The two sets of users are intersected so the test is genuinely
    paired. Returns ``(stat, pvalue, n_pairs)``. ``nan`` p-value with
    ``n_pairs=0`` indicates no overlap; ``p=1.0`` indicates all
    differences are zero (no evidence either way).
    """
    if not _HAVE_WILCOXON:
        return (float('nan'), float('nan'), 0)

    base_vals, base_users = baseline_per_user
    mod_vals,  mod_users  = model_per_user

    base_map = dict(zip(base_users, base_vals))
    mod_map  = dict(zip(mod_users,  mod_vals))
    common = sorted(set(base_map) & set(mod_map))
    if len(common) < 2:
        return (float('nan'), float('nan'), len(common))

    b = np.array([base_map[u] for u in common], dtype=float)
    m = np.array([mod_map[u]  for u in common], dtype=float)
    diff = m - b
    if not np.any(diff != 0):
        return (0.0, 1.0, len(common))
    try:
        stat, p = wilcoxon(m, b, zero_method='wilcox',
                           alternative='two-sided')
    except Exception:  # pragma: no cover
        return (float('nan'), float('nan'), len(common))
    return (float(stat), float(p), int(len(common)))


def wilcoxon_vs_baseline(res_baseline, res_model, k):
    """Convenience wrapper that takes two ``evaluate_at_ks`` results
    and runs ``wilcoxon_paired`` on the per-user NDCG@k arrays.

    Returns ``(stat, pvalue, n_pairs)``. Safe to call with k values
    not present in either result -- returns ``(nan, nan, 0)``.
    """
    if ('per_user_ndcg' not in res_baseline
            or 'per_user_ndcg' not in res_model
            or 'per_user_ids' not in res_baseline
            or 'per_user_ids' not in res_model):
        return (float('nan'), float('nan'), 0)
    b_ndcg = res_baseline['per_user_ndcg'].get(k)
    m_ndcg = res_model['per_user_ndcg'].get(k)
    if b_ndcg is None or m_ndcg is None:
        return (float('nan'), float('nan'), 0)
    return wilcoxon_paired(
        (b_ndcg, res_baseline['per_user_ids']),
        (m_ndcg, res_model['per_user_ids']),
    )


def load_dataset(dataset: str, split_mode: str = 'temporal',
                 split_seed: int = 0, test_ratio: float = 0.2):
    """Load ratings + perform an 80/20 split.

    Parameters
    ----------
    dataset : {'ml-small', 'ml-1m'}
    split_mode : {'temporal', 'random'}
        'temporal' (default, deterministic) uses each user's most recent
        interactions for test, matching the protocol from Steck's EASE
        paper. 'random' picks a per-user random subset using ``split_seed``
        -- used by the multi-seed experiments to report mean ± std across
        several splits.
    split_seed : int
        Only used when ``split_mode='random'``.
    test_ratio : float
        Fraction of each user's interactions in the held-out set.

    Returns
    -------
    train, test_positive, threshold : train df, positive-rating test df,
        and the rating threshold used to filter the test set.
    """
    if dataset == 'ml-1m':
        ratings, _ = load_movielens_1m('data/ml-1m', min_interactions=5)
        threshold = 4.0
    else:
        ratings, _ = load_movielens('data/ml-latest-small',
                                    min_interactions=5)
        threshold = 3.5

    if split_mode == 'temporal':
        train, test = temporal_train_test_split(ratings,
                                                test_ratio=test_ratio)
    elif split_mode == 'random':
        train, test = random_train_test_split(ratings,
                                              test_ratio=test_ratio,
                                              seed=split_seed)
    else:
        raise ValueError(
            f"Unknown split_mode={split_mode!r}; "
            "expected 'temporal' or 'random'.")

    test_positive = test[test['rating'] >= threshold].copy()

    return train, test_positive, threshold


def ensure_results_dir(experiment_name: str,
                      root: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Create ``results/<experiment_name>`` if needed and return the path."""
    out = Path(root) / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out
