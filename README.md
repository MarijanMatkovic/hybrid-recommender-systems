# Graph Laplacian-Regularized EASE for Recommender Systems

Hybrid recommender system combining **EASE** (linear autoencoder) and **RP3beta** (graph-based random walks) through four hybridization strategies, including a novel Graph Laplacian regularization of the EASE objective function.

Master's thesis project — University of Zagreb, 2026.

## Key Results (MovieLens 1M, temporal split)

| Model | NDCG@10 | MAP@10 | HR@10 | vs EASE |
|---|---|---|---|---|
| EASE (baseline) | 0.1163 | 0.0563 | 0.5091 | — |
| RP3beta (baseline) | 0.1163 | 0.0569 | 0.4958 | +0.0% |
| Score hybrid | 0.1227 | 0.0597 | 0.5297 | +5.5% |
| **Laplacian EASE** | **0.1212** | **0.0600** | **0.5057** | **+4.2%** |
| Matrix hybrid | 0.1238 | 0.0607 | 0.5305 | +6.4% |
| GraphReg | 0.1136 | 0.0547 | 0.5057 | −2.3% |

**Laplacian EASE** is the best single-model approach that incorporates graph structure into the EASE objective on the temporal split. Matrix hybrid achieves higher accuracy but requires maintaining two separate models.

> **Protocol note.** The table above is from a single deterministic
> temporal split (Steck's EASE setting) and is kept for comparability
> with prior work. The thesis's headline numbers are reported under the
> **primary protocol**: random 80/20 per user × 5 seeds, with
> per-family Wilcoxon-vs-EASE pooled across seeds. Run the primary job
> with `python main.py --dataset ml-1m --protocol primary` (or
> `qsub hpc/run_primary_multiseed.pbs`). See
> [Evaluation protocol](#evaluation-protocol) below.

## Evaluation protocol

We report results under two split protocols and treat them
asymmetrically:

| Protocol   | Split                                    | Seeds | Role                                                           |
|------------|------------------------------------------|-------|----------------------------------------------------------------|
| **primary** (default) | random 80/20 per user             | 5     | Headline numbers, CIs, Wilcoxon-vs-EASE, thesis tables         |
| temporal   | each user's most recent 20% held out     | 1     | Sanity check, comparability with Steck's EASE paper            |

`main.py` accepts `--protocol {primary,temporal}` (default: `primary`).
Each run writes per-family CSVs + companion per-bucket CSVs
(popularity quintiles q1..q5) to
`results/baselines/<protocol>/`. Under `primary` it additionally
writes:

* `<family>_<dataset>_primary_<ts>_summary.csv` — mean/std/count over seeds
* `<family>_<dataset>_primary_<ts>_buckets_summary.csv` — same for buckets
* `leaderboard_<dataset>_primary_<ts>.csv` — best config per family
* `wilcoxon_<dataset>_primary_<ts>.csv` — pooled (across seeds)
  paired Wilcoxon signed-rank of each family's best mean-NDCG config
  vs EASE's best mean-NDCG config, using every user seen in any seed
  (user IDs are prefixed by seed to keep the test genuinely paired).

Every CSV carries the full seven-metric standard at each cut-off:
`NDCG`, `MAP`, `HitRate`, `Recall`, `Coverage`, `Gini`, `Novelty` —
persisting the diversity columns next to the accuracy ones means the
thesis can report (e.g.) a Laplacian NDCG win offset by a Coverage
regression without having to re-fit.

The reason for anchoring on the random-split primary: the temporal
protocol turns out to be distributionally easier for popularity-biased
models (Laplacian-EASE gains on the temporal split partially evaporate
on random splits — see `experiments/edlae_multiseed.py`). Reporting
both is the honest thing to do; anchoring the headline on the one with
built-in CIs and multi-seed significance tests is the statistically
defensible thing to do.

## Models

### Base models
- **EASE** (Steck, 2019): Linear autoencoder with closed-form solution. Learns dense item-item weight matrix B by minimizing reconstruction error with L2 regularization.
- **RP3beta** (Paudel et al., 2016): 3-step random walk on the user-item bipartite graph. Produces sparse item-item similarity matrix W with popularity damping.

### Hybridization strategies

**Level 1 — Score fusion**: Weighted sum of normalized predictions from both models.

```
score(u,i) = α · norm(X·B) + (1−α) · norm(X·W)
```

**Level 2 — Matrix fusion**: L2-normalized item-item matrices combined before prediction.

```
S = α · normalize(B) + (1−α) · normalize(W)
```

**Level 3 — Graph regularization (‖B−W‖²)**: Penalizes EASE's B for deviating from RP3beta's W. **Does not improve accuracy** — the sparse-dense mismatch causes the penalty to act as additional L2 regularization rather than injecting graph structure.

```
min_B ‖X−XB‖² + λ‖B‖² + γ‖B−W‖²
```

**Level 4 — Graph Laplacian regularization (novel contribution)**: Penalizes B only for giving different weight vectors to items that the graph says are similar. Items not connected in W contribute zero penalty.

```
min_B ‖X−XB‖² + λ‖B‖² + γ·tr(BᵀLB),  diag(B) = 0
```

where L = D − W_sym is the graph Laplacian of the symmetrized RP3beta similarity matrix. Has a closed-form solution:

```
P = (XᵀX + λI + γL)⁻¹
B = P·XᵀX  (with diagonal constraint applied)
```

## Project Structure

```
├── main.py                       # Baseline sweep runner (6-family leaderboard)
├── models/
│   ├── ease.py                   # EASE closed-form solution
│   ├── rp3beta.py                # RP3beta random-walk similarity
│   ├── p3alpha.py                # P3alpha (unnormalised random walk)
│   ├── itemknn.py                # ItemKNN cosine similarity
│   ├── edlae.py                  # EDLAE (dropout regularised EASE)
│   ├── slim.py                   # SLIM + Laplacian-SLIM (ISTA solver)
│   └── hybrid.py                 # All 4 hybridisation strategies + Lap-EASE
├── data/
│   ├── loader.py                 # Data loading + temporal/random splits
│   ├── ml-1m/                    # MovieLens 1M dataset (6040 users, 3706 items)
│   ├── ml-latest-small/          # MovieLens Small dataset (610 users, 3650 items)
│   └── netflixprize/             # Netflix Prize (~480k users, ~17.7k items, ~100M ratings)
├── evaluation/
│   └── metrics.py                # NDCG, MAP, HR, MRR, Precision, Recall,
│                                 # Coverage, Gini, Novelty; evaluate_at_ks
├── experiments/
│   ├── _shared.py                # Shared helpers: load_dataset, wilcoxon_paired,
│   │                             # bucketed_metrics_at_ks, metric_cols_at_ks
│   ├── gamma_sensitivity.py      # NDCG vs γ sweep (temporal)
│   ├── graph_source_ablation.py  # Graph source ablation: rp3β/p3α/itemknn/binary
│   ├── head_tail_analysis.py     # Head/tail NDCG breakdown + γ-sweep (temporal)
│   ├── head_tail_primary_multiseed.py  # Head/tail user-activity, 5 primary seeds
│   │                                   # + pooled ~30k Wilcoxon per bucket
│   ├── slim_experiments.py       # SLIM ElasticNet grid + Laplacian-SLIM sweep
│   ├── slim_pooled_wilcoxon.py   # Pooled ~30k Wilcoxon: SLIM-Lap vs EASE,
│   │                             # 5 primary seeds (reads best config per seed)
│   ├── edlae_experiments.py      # EDLAE dropout sweep (temporal)
│   ├── edlae_multiseed.py        # EDLAE null hypothesis, 5 primary seeds
│   ├── gs_ease_multiseed.py      # GS-EASE (Graph-Shrunk EASE) primary evaluation
│   ├── b_matrix_analysis.py      # B-matrix structural metrics vs γ
│   └── wallclock_summary.py      # Runtime table for Methods chapter
├── hpc/                          # PBS Pro job scripts for Supek HPC
│   └── README.md                 # HPC submission guide
└── results/                      # Output CSVs and PNGs (gitignored)
    ├── baselines/{primary,temporal}/   # 6-family leaderboard + Wilcoxon
    ├── gs_ease_multiseed/              # GS-EASE + L² spectral filter results
    ├── slim_primary/                   # Per-seed SLIM + pooled Wilcoxon
    ├── edlae_multiseed/                # EDLAE null results
    ├── head_tail_analysis/             # Head/tail breakdown (temporal + primary)
    ├── gamma_sensitivity/              # γ-sweep curves
    ├── graph_source_ablation/          # Graph source comparison
    ├── b_matrix_analysis/              # B-matrix structural plots
    └── wallclock_summary/              # Runtime table
```

## Setup

```bash
# Python 3.8+
pip install numpy scipy pandas scikit-learn
```

No deep learning frameworks required — all models have closed-form or analytical solutions.

## Usage

### Quick run

```python
from data import load_movielens_1m, temporal_train_test_split
from models import HybridEASE_RP3beta
from evaluation.metrics import evaluate

ratings, movies = load_movielens_1m('data/ml-1m')
train, test = temporal_train_test_split(ratings, test_ratio=0.2)

# Laplacian EASE (best single-model)
model = HybridEASE_RP3beta()
model.fit(train, method='laplacian',
          ease_lambda=100, rp3_beta=0.3, rp3_topK=300,
          graph_reg_gamma=50)

test_positive = test[test['rating'] >= 4]
results = evaluate(model, train, test_positive, k=10)
```

### Full sweep

```bash
# Primary protocol (random 80/20 × 5 seeds, thesis headline):
python main.py --dataset ml-1m --protocol primary

# Secondary protocol (temporal split, single deterministic run):
python main.py --dataset ml-1m --protocol temporal
```

Each invocation writes per-family CSVs + companion per-bucket CSVs to
`results/baselines/<protocol>/`, plus a leaderboard and (under
`primary`) a pooled-across-seeds Wilcoxon table.

## Evaluation Metrics

**Accuracy**: Precision@k, Recall@k, NDCG@k, HitRate@k, MAP@k, MRR@k  
**Diversity**: Coverage (catalog coverage), Gini index (lower = more equal distribution)  
**Novelty**: Average self-information of recommended items (higher = less popular items)

## Why Laplacian Works but GraphReg Doesn't

The regularization term `tr(BᵀLB) = Σ_{i,j} W_sym[i,j] · ‖bᵢ − bⱼ‖²` only penalizes pairs of items that are connected in the graph, proportional to their similarity weight. Disconnected items contribute zero penalty and remain free.

In contrast, `‖B−W‖²` penalizes every element of B. Since W is ~97% zeros (after top-K pruning) and B is dense, this mostly pushes B toward zero — equivalent to extra L2 regularization, not graph injection.

## Datasets

Evaluated on:

- **[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)** (6,040 users, 3,706 items, ~1M ratings) — primary thesis dataset.
- **[MovieLens Small](https://grouplens.org/datasets/movielens/latest/)** (610 users, ~3,650 items) — fast smoke runs.
- **[Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)** (~480k users, ~17.7k items, ~100M ratings) — scale-up validation. First-time parse takes ~2 min and writes `data/netflixprize/ratings_cache.parquet` for fast subsequent loads. Default `min_interactions=20` (denser than MovieLens). Output filenames carry `netflix-prize` to distinguish from ML-1M results.

All datasets use rating ≥ 4.0 (≥ 3.5 for ml-small) as positive implicit feedback. Both 80/20 temporal and 80/20 random per-user splits are supported.

Pass `--dataset netflix-prize` to `main.py` or any experiment script. Memory: full Netflix EASE peak ~20 GB (allocate ≥ 32 GB).

411 hyperparameter configurations tested across all models on ML-1M; the Netflix grid is leaner (~19 configs/seed) due to ~125x slower per-fit cost.

## References

- Steck, H. (2019). Embarrassingly Shallow Autoencoders for Sparse Data. *WWW 2019*.
- Paudel, B. et al. (2016). Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications. *TiiS*.
- Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *TiiS 5(4)*.

## License

Academic use. MovieLens data subject to [GroupLens license terms](https://grouplens.org/datasets/movielens/).
