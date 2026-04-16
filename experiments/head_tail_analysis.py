"""
Head vs tail analysis: where does Laplacian regularization help most?

Items are partitioned into buckets (head/torso/tail) by training-set
popularity, then NDCG@k is computed **restricted to relevant items in
each bucket** for both the baseline (vanilla EASE) and Laplacian-EASE.

The output shows the absolute NDCG gain per bucket -- Laplacian
regularization usually helps the torso and (especially) the tail, since
the graph mixes in multi-hop neighbour information that shallow linear
models can't see.

Example:
    python -m experiments.head_tail_analysis --dataset ml-small --k 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k
from models import HybridEASE_RP3beta

from experiments._shared import ensure_results_dir, load_dataset


def _item_popularity_buckets(train_df, n_buckets=3,
                             labels=('head', 'torso', 'tail')):
    """Bucket items by training popularity (equal-frequency quantiles).

    Returns a dict {item_id -> bucket_label}. Items with identical
    popularity fall into the same bucket.
    """
    counts = train_df.groupby('item_id')['user_id'].nunique()
    # qcut with duplicates='drop' can merge buckets -- fall back to pd.cut
    # on the sorted counts to force the requested number of buckets even
    # when popularity distribution is lopsided.
    sorted_counts = counts.sort_values(ascending=False)
    n = len(sorted_counts)
    bucket_of = {}
    # Split the ranked list into ~equal chunks (head = most popular).
    chunks = np.array_split(sorted_counts.index.to_numpy(), n_buckets)
    for label, chunk in zip(labels, chunks):
        for item in chunk:
            bucket_of[item] = label
    return bucket_of


def _evaluate_by_bucket(model, train_df, test_df, bucket_of, k=10):
    """Evaluate NDCG@k per popularity bucket.

    For each user in the test set, we split the relevant items by bucket
    and compute NDCG@k using only the relevant items from that bucket.
    The top-k recommendations themselves are NOT restricted -- we still
    recommend any item; we just measure how well we ranked tail items
    when the ground truth *is* a tail item.
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

    per_bucket = {b: {'ndcg': [], 'recall': [], 'hit': []}
                  for b in set(bucket_of.values())}
    overall = {'ndcg': [], 'recall': [], 'hit': []}

    for user_id, relevant in test_grouped.items():
        if user_id not in known_users:
            continue
        relevant = relevant & known_items
        if not relevant:
            continue

        user_idx = user_enc.transform([user_id])[0]
        scores = pred_matrix[user_idx, :].copy()

        # Mask training items
        for item_id in train_grouped.get(user_id, set()):
            if item_id in known_items:
                scores[item_enc.transform([item_id])[0]] = -np.inf

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        top_k_items = item_enc.inverse_transform(top_k_idx).tolist()

        rel_list = list(relevant)

        overall['ndcg'].append(ndcg_at_k(top_k_items, rel_list, k))
        overall['recall'].append(recall_at_k(top_k_items, rel_list, k))
        overall['hit'].append(hit_rate_at_k(top_k_items, rel_list, k))

        for bucket in per_bucket:
            rel_in_bucket = [i for i in rel_list
                             if bucket_of.get(i) == bucket]
            if not rel_in_bucket:
                continue
            per_bucket[bucket]['ndcg'].append(
                ndcg_at_k(top_k_items, rel_in_bucket, k))
            per_bucket[bucket]['recall'].append(
                recall_at_k(top_k_items, rel_in_bucket, k))
            per_bucket[bucket]['hit'].append(
                hit_rate_at_k(top_k_items, rel_in_bucket, k))

    summary = {
        'overall': {m: float(np.mean(v)) if v else 0.0
                    for m, v in overall.items()},
    }
    for b, metrics in per_bucket.items():
        summary[b] = {m: float(np.mean(v)) if v else 0.0
                      for m, v in metrics.items()}
        summary[b]['n_users'] = len(metrics['ndcg'])
    summary['overall']['n_users'] = len(overall['ndcg'])
    return summary


def run(dataset='ml-small', k=10,
        ease_lambda=None, gamma=None, rp3_beta=0.6, rp3_topK=200,
        graph_source='rp3beta', normalise='none',
        n_buckets=3,
        out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if gamma is None:
        gamma = 100.0 if dataset == 'ml-1m' else 10.0

    out_dir = ensure_results_dir('head_tail_analysis'
                                 if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)
    labels = ('head', 'torso', 'tail') if n_buckets == 3 else tuple(
        f'q{i+1}' for i in range(n_buckets))
    bucket_of = _item_popularity_buckets(train, n_buckets=n_buckets,
                                         labels=labels)

    # ---- Baseline: vanilla EASE (gamma=0) ----
    print("\n[baseline] EASE")
    t0 = time.time()
    base = HybridEASE_RP3beta()
    base.fit(train, method='score', fusion_alpha=1.0,
             ease_lambda=ease_lambda, rp3_alpha=1.0,
             rp3_beta=rp3_beta, rp3_topK=rp3_topK)
    base.pred = base.ease.X.dot(base.ease.B)
    base_summary = _evaluate_by_bucket(base, train, test_positive,
                                       bucket_of, k=k)
    print(f"  EASE (overall NDCG={base_summary['overall']['ndcg']:.4f}) "
          f"  ({time.time()-t0:.1f}s)")

    # ---- Laplacian ----
    print(f"\n[Laplacian] source={graph_source}, gamma={gamma}, "
          f"normalise={normalise}")
    t0 = time.time()
    lap = HybridEASE_RP3beta()
    lap.fit(train, method='laplacian',
            ease_lambda=ease_lambda, rp3_alpha=1.0,
            rp3_beta=rp3_beta, rp3_topK=rp3_topK,
            graph_reg_gamma=gamma, graph_source=graph_source,
            laplacian_normalise=normalise)
    lap_summary = _evaluate_by_bucket(lap, train, test_positive,
                                      bucket_of, k=k)
    print(f"  Laplacian (overall NDCG={lap_summary['overall']['ndcg']:.4f}) "
          f"  ({time.time()-t0:.1f}s)")

    # ---- Assemble result table ----
    rows = []
    for bucket in ('overall',) + labels:
        for metric in ('ndcg', 'recall', 'hit'):
            base_val = base_summary[bucket][metric]
            lap_val = lap_summary[bucket][metric]
            rows.append({
                'dataset': dataset,
                'graph_source': graph_source,
                'normalise': normalise,
                'bucket': bucket,
                'n_users': base_summary[bucket]['n_users'],
                'metric': metric,
                'EASE': base_val,
                'Laplacian': lap_val,
                'abs_gain': lap_val - base_val,
                'rel_gain': (lap_val - base_val) /
                            base_val if base_val > 0 else 0.0,
            })
    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    src_suffix = f'_{graph_source}' if graph_source != 'rp3beta' else ''
    csv_path = out_dir / f'head_tail_{dataset}{src_suffix}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Plot: absolute gain per bucket ----
    ndcg_rows = df[df['metric'] == 'ndcg']
    buckets = ndcg_rows['bucket'].tolist()
    gains = ndcg_rows['abs_gain'].tolist()
    colors = ['#4c72b0' if g >= 0 else '#c44e52' for g in gains]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(buckets, gains, color=colors)
    for bar, g in zip(bars, gains):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{g:+.4f}",
                ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel(f'Δ NDCG@{k}  (Laplacian − EASE)')
    title_extra = f' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Head vs tail gain — {dataset} ({graph_source}){title_extra}')
    ax.grid(True, axis='y', ls=':', alpha=0.5)
    fig.tight_layout()
    png_path = out_dir / f'head_tail_{dataset}{src_suffix}{suffix}.png'
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot to {png_path}")

    print("\nPer-bucket NDCG summary:")
    pivot = ndcg_rows.set_index('bucket')[['EASE', 'Laplacian',
                                           'abs_gain', 'rel_gain']]
    print(pivot.to_string())

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--lambda_', dest='ease_lambda', type=float, default=None)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'],
                   help='Laplacian normalisation. Default: none.')
    p.add_argument('--n_buckets', type=int, default=3)
    args = p.parse_args()

    run(dataset=args.dataset, k=args.k,
        gamma=args.gamma, ease_lambda=args.ease_lambda,
        rp3_beta=args.rp3_beta, graph_source=args.graph_source,
        normalise=args.normalise,
        n_buckets=args.n_buckets)


if __name__ == '__main__':
    main()
