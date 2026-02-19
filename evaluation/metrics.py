"""
Ranking evaluation metrics for recommender systems.
All metrics are computed per-user and then averaged.

Accuracy:   Precision@k, Recall@k, NDCG@k, HitRate@k, MAP@k, MRR@k
Diversity:  Coverage, Gini index
Novelty:    Average self-information
"""

import numpy as np
from collections import Counter


def precision_at_k(recommended, relevant, k):
    """Fraction of top-k recommendations that are relevant."""
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items that appear in top-k."""
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain at k."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended, relevant, k):
    """1 if any relevant item is in top-k, else 0."""
    rec_k = recommended[:k]
    return 1.0 if len(set(rec_k) & set(relevant)) > 0 else 0.0


def map_at_k(recommended, relevant, k):
    """
    Mean Average Precision at k.
    Average of precision values at each position where a relevant item appears.
    """
    rec_k = recommended[:k]
    score = 0.0
    hits = 0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if len(relevant) > 0 else 0.0


def mrr_at_k(recommended, relevant, k):
    """Mean Reciprocal Rank at k -- 1/rank of first relevant item."""
    rec_k = recommended[:k]
    for i, item in enumerate(rec_k):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def coverage(all_recommendations, n_items):
    """Fraction of all items recommended to at least one user."""
    unique_recommended = set()
    for recs in all_recommendations:
        unique_recommended.update(recs)
    return len(unique_recommended) / n_items


def gini_index(all_recommendations):
    """
    Gini index of recommendation frequency distribution.
    0 = perfectly equal (every item recommended equally often)
    1 = perfectly unequal (only one item ever recommended)
    Lower is better for diversity.
    """
    counts = Counter()
    for recs in all_recommendations:
        counts.update(recs)

    if len(counts) == 0:
        return 1.0

    freqs = np.array(sorted(counts.values()), dtype=float)
    n = len(freqs)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * freqs) / (n * np.sum(freqs))) - (n + 1) / n


def novelty(all_recommendations, item_popularity):
    """
    Average self-information of recommended items.
    Higher = more novel (less popular items recommended).
    item_popularity: dict mapping item_id -> fraction of users who interacted.
    """
    scores = []
    for recs in all_recommendations:
        for item in recs:
            pop = item_popularity.get(item, 1e-10)
            scores.append(-np.log2(pop))
    return np.mean(scores) if scores else 0.0


def evaluate(model, train_df, test_df, k=10):
    """
    Evaluate a fitted model against held-out test interactions.
    Returns dict with all ranking, coverage, diversity, and novelty metrics.
    """
    pred_matrix = model.pred
    if hasattr(pred_matrix, 'toarray'):
        pred_matrix = pred_matrix.toarray()

    user_enc = model.ease.user_enc
    item_enc = model.ease.item_enc

    known_users = set(user_enc.classes_)
    known_items = set(item_enc.classes_)

    test_grouped = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    train_grouped = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    # Item popularity for novelty
    n_users_total = train_df['user_id'].nunique()
    item_pop_counts = train_df.groupby('item_id')['user_id'].nunique()
    item_popularity = (item_pop_counts / n_users_total).to_dict()

    precisions, recalls, ndcgs, hit_rates = [], [], [], []
    maps, mrrs = [], []
    all_recs = []

    n_evaluated = 0
    for user_id, relevant_items in test_grouped.items():
        if user_id not in known_users:
            continue

        relevant_items = relevant_items & known_items
        if len(relevant_items) == 0:
            continue

        user_idx = user_enc.transform([user_id])[0]
        scores = pred_matrix[user_idx, :].copy()

        # Mask training items
        train_items = train_grouped.get(user_id, set())
        for item_id in train_items:
            if item_id in known_items:
                item_idx = item_enc.transform([item_id])[0]
                scores[item_idx] = -np.inf

        # Top-k sorted by score descending
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        top_k_items = item_enc.inverse_transform(top_k_idx).tolist()

        relevant_list = list(relevant_items)
        precisions.append(precision_at_k(top_k_items, relevant_list, k))
        recalls.append(recall_at_k(top_k_items, relevant_list, k))
        ndcgs.append(ndcg_at_k(top_k_items, relevant_list, k))
        hit_rates.append(hit_rate_at_k(top_k_items, relevant_list, k))
        maps.append(map_at_k(top_k_items, relevant_list, k))
        mrrs.append(mrr_at_k(top_k_items, relevant_list, k))
        all_recs.append(top_k_items)
        n_evaluated += 1

    n_total_items = len(item_enc.classes_)

    results = {
        'Precision@k': np.mean(precisions),
        'Recall@k':    np.mean(recalls),
        'NDCG@k':      np.mean(ndcgs),
        'HitRate@k':   np.mean(hit_rates),
        'MAP@k':       np.mean(maps),
        'MRR@k':       np.mean(mrrs),
        'Coverage':    coverage(all_recs, n_total_items),
        'Gini':        gini_index(all_recs),
        'Novelty':     novelty(all_recs, item_popularity),
        'n_users_evaluated': n_evaluated,
        'k': k,
    }

    return results


def print_results(results, model_name="Model"):
    """Pretty-print evaluation results."""
    k = results['k']
    print(f"\n{'─' * 50}")
    print(f"  {model_name} (k={k}, {results['n_users_evaluated']} users)")
    print(f"{'─' * 50}")
    print(f"  Precision@{k}:  {results['Precision@k']:.4f}")
    print(f"  Recall@{k}:     {results['Recall@k']:.4f}")
    print(f"  NDCG@{k}:       {results['NDCG@k']:.4f}")
    print(f"  HitRate@{k}:    {results['HitRate@k']:.4f}")
    print(f"  MAP@{k}:        {results['MAP@k']:.4f}")
    print(f"  MRR@{k}:        {results['MRR@k']:.4f}")
    print(f"  Coverage:       {results['Coverage']:.4f}")
    print(f"  Gini:           {results['Gini']:.4f}")
    print(f"  Novelty:        {results['Novelty']:.2f}")
    print(f"{'─' * 50}")