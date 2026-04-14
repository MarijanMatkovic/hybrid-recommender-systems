"""
Shared helpers for the weekly experiment scripts.

Every script accepts ``--dataset`` (``ml-small`` | ``ml-1m``) and writes
results/plots under ``results/<experiment_name>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from data import load_movielens, load_movielens_1m, temporal_train_test_split


DEFAULT_RESULTS_DIR = Path('results')


def load_dataset(dataset: str):
    """Load ratings + perform a temporal 80/20 split.

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

    train, test = temporal_train_test_split(ratings, test_ratio=0.2)
    test_positive = test[test['rating'] >= threshold].copy()

    return train, test_positive, threshold


def ensure_results_dir(experiment_name: str,
                      root: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Create ``results/<experiment_name>`` if needed and return the path."""
    out = Path(root) / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out
