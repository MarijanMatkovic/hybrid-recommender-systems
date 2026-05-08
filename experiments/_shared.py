"""
Shared helpers for the weekly experiment scripts.

Every script accepts ``--dataset`` (``ml-small`` | ``ml-1m``) and writes
results/plots under ``results/<experiment_name>/``.

Primary evaluation protocol
---------------------------
The thesis reports two protocols; **the primary one is random 80/20 per
user × 5 seeds** (``split_mode='random'``, seeds 0..4). All headline
numbers, confidence intervals, and statistical tests are reported on the
primary protocol. The secondary protocol — temporal last-item split
(``split_mode='temporal'``) — is reported for comparability with Steck's
EASE paper and is treated as a single-split sanity check. A method that
wins under the temporal protocol but loses under the primary protocol is
reported as *protocol-dependent* rather than as a generic improvement.

Rationale:

* The random protocol has built-in CIs via the 5 seeds, making aggregate
  claims falsifiable.
* Temporal-split-only evaluation can hide distributional wins (e.g. the
  Laplacian helps popular items) by evaluating on a tail that's easier
  to predict for popularity-biased models.
* The multi-seed edlae_multiseed script was what first surfaced that
  the Laplacian regression vanishes on random splits — this is the
  scientifically honest place to anchor headline numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    _HAVE_WILCOXON = True
except Exception:  # pragma: no cover
    _HAVE_WILCOXON = False

from data import (
    load_movielens,
    load_movielens_1m,
    load_netflix_prize,
    temporal_train_test_split,
    random_train_test_split,
)


DEFAULT_RESULTS_DIR = Path('results')

# Canonical per-k metric tuple. ``metric_cols_at_ks`` and any code that
# iterates the standard columns should use this list -- updating in one
# place propagates to every experiment's CSV schema.
STANDARD_METRICS = ('NDCG', 'MAP', 'HitRate', 'Recall',
                    'Coverage', 'Gini', 'Novelty')

# Default seeds used by the primary (random 80/20 × 5 seeds) protocol.
PRIMARY_SPLIT_SEEDS = (0, 1, 2, 3, 4)


def wilcoxon_paired(baseline_per_user, model_per_user):
    """Paired Wilcoxon signed-rank on per-user NDCG, aligned by user_id.

    Parameters
    ----------
    baseline_per_user : tuple[np.ndarray, list]
        ``(values, user_ids)`` for the baseline model (e.g. EASE).
    model_per_user : tuple[np.ndarray, list]
        ``(values, user_ids)`` for the compared model (e.g. Laplacian-EASE).

    The two sets of users are intersected so the test is genuinely
    paired.

    Returns
    -------
    (stat, pvalue, n_pairs, sign, mean_diff, median_diff) : tuple
        ``stat``, ``pvalue``: from ``scipy.stats.wilcoxon``.
        ``n_pairs``: number of overlapping users. ``nan`` p-value with
            ``n_pairs=0`` indicates no overlap; ``p=1.0`` indicates all
            differences are zero (no evidence either way).
        ``sign`` in {-1, 0, +1}: sign of ``mean(model - baseline)``. We
            intentionally use the *mean* (not the median) here because
            per-user NDCG is sparse — many users score 0, so the median
            difference is often exactly 0 even when the Wilcoxon test
            strongly rejects the null. A tiny p-value with ``sign=-1``
            therefore means the model is SIGNIFICANTLY WORSE, not
            better. If the mean is exactly zero we fall back to the sign
            of ``wilcoxon.statistic - expected_W`` as a tie-breaker so
            a significant but mean-neutral asymmetry is still
            directional.
        ``mean_diff``:   ``mean(model - baseline)``   (the signed mean
            gain; the primary magnitude + sign anchor).
        ``median_diff``: ``median(model - baseline)`` (kept for
            reference; robust to outliers but often 0 on sparse NDCG).
    """
    empty = (float('nan'), float('nan'), 0, 0,
             float('nan'), float('nan'))
    if not _HAVE_WILCOXON:
        return empty

    base_vals, base_users = baseline_per_user
    mod_vals,  mod_users  = model_per_user

    base_map = dict(zip(base_users, base_vals))
    mod_map  = dict(zip(mod_users,  mod_vals))
    common = sorted(set(base_map) & set(mod_map))
    if len(common) < 2:
        return (float('nan'), float('nan'), len(common),
                0, float('nan'), float('nan'))

    b = np.array([base_map[u] for u in common], dtype=float)
    m = np.array([mod_map[u]  for u in common], dtype=float)
    diff = m - b
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    if not np.any(diff != 0):
        return (0.0, 1.0, len(common), 0, 0.0, 0.0)
    try:
        stat, p = wilcoxon(m, b, zero_method='wilcox',
                           alternative='two-sided')
    except Exception:  # pragma: no cover
        sign = (int(np.sign(mean_diff))
                if mean_diff != 0.0 else 0)
        return (float('nan'), float('nan'), len(common),
                sign, mean_diff, median_diff)
    # Primary sign from mean. Fall back to the Wilcoxon W-statistic's
    # orientation when the mean is exactly zero (rare; means a
    # distributional asymmetry with no net mean shift, like one big
    # loss offsetting many small wins).
    if mean_diff != 0.0:
        sign = int(np.sign(mean_diff))
    else:
        # scipy.wilcoxon's ``statistic`` is sum of positive ranks. The
        # expected value under the null is n*(n+1)/4.
        n = len(diff)
        expected_W = n * (n + 1) / 4.0
        sign = int(np.sign(float(stat) - expected_W))
    return (float(stat), float(p), int(len(common)),
            sign, mean_diff, median_diff)


def wilcoxon_vs_baseline(res_baseline, res_model, k):
    """Convenience wrapper that takes two ``evaluate_at_ks`` results
    and runs ``wilcoxon_paired`` on the per-user NDCG@k arrays.

    Returns ``(stat, pvalue, n_pairs, sign, mean_diff, median_diff)``.
    Safe to call with k values not present in either result --
    returns ``(nan, nan, 0, 0, nan, nan)``.
    """
    empty = (float('nan'), float('nan'), 0, 0,
             float('nan'), float('nan'))
    if ('per_user_ndcg' not in res_baseline
            or 'per_user_ndcg' not in res_model
            or 'per_user_ids' not in res_baseline
            or 'per_user_ids' not in res_model):
        return empty
    b_ndcg = res_baseline['per_user_ndcg'].get(k)
    m_ndcg = res_model['per_user_ndcg'].get(k)
    if b_ndcg is None or m_ndcg is None:
        return empty
    return wilcoxon_paired(
        (b_ndcg, res_baseline['per_user_ids']),
        (m_ndcg, res_model['per_user_ids']),
    )


def wilcoxon_columns(prefix: str, wilcoxon_result):
    """Expand a Wilcoxon result tuple into CSV-friendly columns.

    Produces the canonical column layout used across every experiment
    CSV so downstream tooling (e.g. thesis plot scripts) can assume a
    stable schema:

        {prefix}_stat, {prefix}_p, {prefix}_n_pairs,
        {prefix}_sign, {prefix}_mean_diff, {prefix}_median_diff

    ``mean_diff`` is the primary magnitude column (it's what ``sign``
    is computed from); ``median_diff`` is kept for back-compat and for
    the rare case where a reader wants a robust-to-outlier magnitude.
    """
    stat, p, n, sign, mean_diff, median_diff = wilcoxon_result
    return {
        f'{prefix}_stat':         stat,
        f'{prefix}_p':            p,
        f'{prefix}_n_pairs':      n,
        f'{prefix}_sign':         sign,
        f'{prefix}_mean_diff':    mean_diff,
        f'{prefix}_median_diff':  median_diff,
    }


def wilcoxon_blank_columns(prefix: str):
    """Columns carrying the blank/sentinel values for rows where the
    paired test doesn't apply (e.g. the baseline compared against itself).
    Keeps every CSV's schema identical."""
    return wilcoxon_columns(prefix, (float('nan'), float('nan'),
                                     0, 0,
                                     float('nan'), float('nan')))


DATASET_CHOICES = ('ml-small', 'ml-1m', 'netflix-prize')


def load_dataset(dataset: str, split_mode: str = 'temporal',
                 split_seed: int = 0, test_ratio: float = 0.2,
                 subsample_users: int = None,
                 min_interactions: int = None):
    """Load ratings + perform an 80/20 split.

    Parameters
    ----------
    dataset : {'ml-small', 'ml-1m', 'netflix-prize'}
        Which corpus to load. ``netflix-prize`` reads
        ``data/netflixprize/`` (4 combined_data files, ~100M ratings,
        480k users, 17.7k items). On first load the parser caches a
        parquet file next to the source for fast reload.
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
    subsample_users : int or None
        Netflix-prize only: optional cap on number of users (development
        speedup). Default None = use all ~480k users.
    min_interactions : int or None
        Override the per-dataset default interaction threshold. Defaults
        are: ml-small=5, ml-1m=5, netflix-prize=20.

    Returns
    -------
    train, test_positive, threshold : train df, positive-rating test df,
        and the rating threshold used to filter the test set.
    """
    if dataset == 'ml-1m':
        mi = 5 if min_interactions is None else min_interactions
        ratings, _ = load_movielens_1m('data/ml-1m', min_interactions=mi)
        threshold = 4.0
    elif dataset == 'netflix-prize':
        mi = 20 if min_interactions is None else min_interactions
        ratings, _ = load_netflix_prize(
            'data/netflixprize',
            min_interactions=mi,
            subsample_users=subsample_users)
        threshold = 4.0
    elif dataset == 'ml-small':
        mi = 5 if min_interactions is None else min_interactions
        ratings, _ = load_movielens('data/ml-latest-small',
                                    min_interactions=mi)
        threshold = 3.5
    else:
        raise ValueError(
            f"Unknown dataset={dataset!r}; "
            f"expected one of {DATASET_CHOICES}.")

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


# ---------------------------------------------------------------------------
# Metric helpers -- used by every experiment to assemble CSV rows uniformly.
# ---------------------------------------------------------------------------

def metric_cols_at_ks(res, ks, primary_k=None):
    """Return the standard metric columns from an ``evaluate_at_ks`` result.

    Produces one ``{metric}@{k}`` column per (metric, k) combination for
    the ``STANDARD_METRICS`` list, plus — if ``primary_k`` is given —
    the ``{metric}@k`` aliases pointed at ``primary_k`` (used by the
    plotting code that only knows the primary cutoff).

    All seven standard metrics are returned on every row: the four
    accuracy ones (NDCG/MAP/HitRate/Recall) and the three
    diversity/coverage ones (Coverage/Gini/Novelty). Persisting the
    diversity metrics alongside accuracy is required so the thesis can
    report — e.g. — a Laplacian win on NDCG offset by a Coverage loss.
    """
    out = {}
    for kk in ks:
        for m in STANDARD_METRICS:
            key = f'{m}@{kk}'
            if key in res:
                out[key] = res[key]
    if primary_k is not None:
        for m in STANDARD_METRICS:
            key = f'{m}@{primary_k}'
            if key in res:
                out[f'{m}@k'] = res[key]
    return out


# ---------------------------------------------------------------------------
# Bucket helpers (popularity quintiles) -- companion to every aggregate
# NDCG row the experiment scripts emit. Importing from
# ``head_tail_analysis`` would be circular, so the bucketing math is
# duplicated here (it's five lines).
# ---------------------------------------------------------------------------

# Default number of popularity buckets used when an experiment script
# emits a companion buckets CSV. Five = thesis-standard quintiles
# (q1 = most popular, q5 = tail).
DEFAULT_N_BUCKETS = 5


def quantile_labels(n_buckets):
    """Bucket labels: head/torso/tail for 3, q1..qN otherwise."""
    if n_buckets == 3:
        return ('head', 'torso', 'tail')
    return tuple(f'q{i+1}' for i in range(n_buckets))


def item_popularity_buckets(train_df, n_buckets=DEFAULT_N_BUCKETS,
                            labels=None):
    """Bucket items by training popularity (rank-based equal-frequency).

    ``labels[0]`` goes to the most-popular items, ``labels[-1]`` to the
    least popular. Returns a dict ``{item_id -> bucket_label}``.
    """
    if labels is None:
        labels = quantile_labels(n_buckets)
    counts = train_df.groupby('item_id')['user_id'].nunique()
    sorted_counts = counts.sort_values(ascending=False)
    chunks = np.array_split(sorted_counts.index.to_numpy(), n_buckets)
    bucket_of = {}
    for label, chunk in zip(labels, chunks):
        for item in chunk:
            bucket_of[item] = label
    return bucket_of


def bucketed_metrics_at_ks(model, train_df, test_df,
                           ks=(10, 20),
                           n_buckets=DEFAULT_N_BUCKETS,
                           bucket_of=None):
    """Per-bucket NDCG/Recall/HitRate at multiple ks for a fitted model.

    Mirrors ``evaluate_at_ks`` but restricts each user's relevant-item
    set to a specific popularity bucket before computing per-user
    scores. A user may contribute to several buckets if their held-out
    items span several popularity levels.

    Parameters
    ----------
    bucket_of : dict | None
        Pre-built ``{item_id -> bucket_label}`` map. If None, the map
        is built from ``train_df`` by ``item_popularity_buckets``.

    Returns
    -------
    dict
        ``{bucket_label: {NDCG@k: float, Recall@k: float, HitRate@k: float,
                          n_users: int}}`` for each bucket (plus the
        ``overall`` key). Always returns the full label set even if a
        bucket is empty (to keep CSV schemas stable across experiments).
    """
    # Local imports to avoid a top-level dependency from ``_shared``
    # on the evaluation package's heavier imports.
    from evaluation.metrics import (
        ndcg_at_k, recall_at_k, hit_rate_at_k,
    )

    ks = tuple(sorted(set(int(k) for k in ks)))
    k_max = ks[-1]

    if bucket_of is None:
        bucket_of = item_popularity_buckets(train_df, n_buckets=n_buckets)
    bucket_labels = list(quantile_labels(n_buckets))

    pred_matrix = model.pred
    if hasattr(pred_matrix, 'toarray'):
        pred_matrix = pred_matrix.toarray()

    user_enc = model.ease.user_enc
    item_enc = model.ease.item_enc
    known_users = set(user_enc.classes_)
    known_items = set(item_enc.classes_)

    test_grouped = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    train_grouped = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    # ``per_bucket[b][k]`` is a list of per-user metric tuples
    per_bucket = {b: {k: {'ndcg': [], 'recall': [], 'hit': []}
                       for k in ks}
                  for b in (['overall'] + bucket_labels)}
    n_users_in_bucket = {b: 0 for b in (['overall'] + bucket_labels)}

    for user_id, relevant in test_grouped.items():
        if user_id not in known_users:
            continue
        relevant = relevant & known_items
        if not relevant:
            continue

        user_idx = user_enc.transform([user_id])[0]
        scores = pred_matrix[user_idx, :].copy()
        for item_id in train_grouped.get(user_id, set()):
            if item_id in known_items:
                scores[item_enc.transform([item_id])[0]] = -np.inf

        top_idx = np.argpartition(scores, -k_max)[-k_max:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_items_max = item_enc.inverse_transform(top_idx).tolist()

        rel_list = list(relevant)
        per_bucket_rel = {b: [] for b in bucket_labels}
        for item_id in rel_list:
            bucket = bucket_of.get(item_id)
            if bucket is not None:
                per_bucket_rel[bucket].append(item_id)

        n_users_in_bucket['overall'] += 1
        for k in ks:
            top_k = top_items_max[:k]
            per_bucket['overall'][k]['ndcg'].append(
                ndcg_at_k(top_k, rel_list, k))
            per_bucket['overall'][k]['recall'].append(
                recall_at_k(top_k, rel_list, k))
            per_bucket['overall'][k]['hit'].append(
                hit_rate_at_k(top_k, rel_list, k))

        for bucket in bucket_labels:
            bucket_rel = per_bucket_rel[bucket]
            if not bucket_rel:
                continue
            n_users_in_bucket[bucket] += 1
            for k in ks:
                top_k = top_items_max[:k]
                per_bucket[bucket][k]['ndcg'].append(
                    ndcg_at_k(top_k, bucket_rel, k))
                per_bucket[bucket][k]['recall'].append(
                    recall_at_k(top_k, bucket_rel, k))
                per_bucket[bucket][k]['hit'].append(
                    hit_rate_at_k(top_k, bucket_rel, k))

    summary = {}
    for bucket in (['overall'] + bucket_labels):
        summary[bucket] = {'n_users': n_users_in_bucket[bucket]}
        for k in ks:
            metrics = per_bucket[bucket][k]
            summary[bucket][f'NDCG@{k}'] = (
                float(np.mean(metrics['ndcg']))
                if metrics['ndcg'] else 0.0)
            summary[bucket][f'Recall@{k}'] = (
                float(np.mean(metrics['recall']))
                if metrics['recall'] else 0.0)
            summary[bucket][f'HitRate@{k}'] = (
                float(np.mean(metrics['hit']))
                if metrics['hit'] else 0.0)
    return summary


def bucket_rows(base_row, bucketed, ks, n_buckets=DEFAULT_N_BUCKETS):
    """Expand a bucketed-metrics summary into long-form CSV rows.

    Each returned row has all the keys of ``base_row`` plus a
    ``bucket`` column and per-k NDCG/Recall/HitRate + ``n_users``
    columns. The ``overall`` bucket is included so a consumer can
    always verify that sum/weighted-mean of bucket NDCGs is consistent
    with the aggregate row reported in the main CSV.
    """
    labels = ['overall'] + list(quantile_labels(n_buckets))
    out = []
    for bucket in labels:
        if bucket not in bucketed:
            continue
        row = dict(base_row)
        row['bucket'] = bucket
        row['n_users'] = bucketed[bucket]['n_users']
        for k in ks:
            for m in ('NDCG', 'Recall', 'HitRate'):
                key = f'{m}@{k}'
                if key in bucketed[bucket]:
                    row[key] = bucketed[bucket][key]
        out.append(row)
    return out


def write_buckets_csv(out_dir, filename_stem, bucket_rows_list):
    """Persist a list of bucket rows to ``<stem>_buckets.csv`` next to
    the main results CSV. No-op if the list is empty."""
    if not bucket_rows_list:
        return None
    df = pd.DataFrame(bucket_rows_list)
    path = Path(out_dir) / f'{filename_stem}_buckets.csv'
    df.to_csv(path, index=False)
    return path
