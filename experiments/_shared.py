"""
Shared helpers for the weekly experiment scripts.

Every script accepts ``--dataset`` (``ml-small`` | ``ml-1m``) and writes
results/plots under ``results/<experiment_name>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from data import (
    load_movielens,
    load_movielens_1m,
    temporal_train_test_split,
    random_train_test_split,
)


DEFAULT_RESULTS_DIR = Path('results')


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
