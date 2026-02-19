# Graph Laplacian-Regularized EASE for Recommender Systems

Hybrid recommender system combining **EASE** (linear autoencoder) and **RP3beta** (graph-based random walks) through four hybridization strategies, including a novel Graph Laplacian regularization of the EASE objective function.

Master's thesis project — University of Zagreb, 2026.

## Key Results (MovieLens 1M)

| Model | NDCG@10 | MAP@10 | HR@10 | vs EASE |
|---|---|---|---|---|
| EASE (baseline) | 0.1163 | 0.0563 | 0.5091 | — |
| RP3beta (baseline) | 0.1163 | 0.0569 | 0.4958 | +0.0% |
| Score hybrid | 0.1227 | 0.0597 | 0.5297 | +5.5% |
| **Laplacian EASE** | **0.1212** | **0.0600** | **0.5057** | **+4.2%** |
| Matrix hybrid | 0.1238 | 0.0607 | 0.5305 | +6.4% |
| GraphReg | 0.1136 | 0.0547 | 0.5057 | −2.3% |

**Laplacian EASE** is the best single-model approach that incorporates graph structure into the EASE objective. Matrix hybrid achieves higher accuracy but requires maintaining two separate models.

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
├── main.py                  # Experiment runner and hyperparameter sweeps
├── models/
│   ├── __init__.py
│   ├── ease.py              # EASE with closed-form solution
│   ├── rp3beta.py           # RP3beta random walk similarity
│   └── hybrid.py            # All 4 hybridization strategies
├── data/
│   ├── loader.py            # MovieLens data loading and temporal split
│   └── ml-1m/               # MovieLens 1M dataset
└── evaluation/
    └── metrics.py           # NDCG, MAP, HR, MRR, Precision, Recall,
                             # Coverage, Gini, Novelty
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
python main.py
```

Runs all models across hyperparameter grid. Results saved to CSV.

## Evaluation Metrics

**Accuracy**: Precision@k, Recall@k, NDCG@k, HitRate@k, MAP@k, MRR@k  
**Diversity**: Coverage (catalog coverage), Gini index (lower = more equal distribution)  
**Novelty**: Average self-information of recommended items (higher = less popular items)

## Why Laplacian Works but GraphReg Doesn't

The regularization term `tr(BᵀLB) = Σ_{i,j} W_sym[i,j] · ‖bᵢ − bⱼ‖²` only penalizes pairs of items that are connected in the graph, proportional to their similarity weight. Disconnected items contribute zero penalty and remain free.

In contrast, `‖B−W‖²` penalizes every element of B. Since W is ~97% zeros (after top-K pruning) and B is dense, this mostly pushes B toward zero — equivalent to extra L2 regularization, not graph injection.

## Datasets

Evaluated on [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (6,040 users, 3,706 items, ~1M ratings). Temporal split 80/20, ratings ≥ 4.0 as positive implicit feedback.

411 hyperparameter configurations tested across all models.

## References

- Steck, H. (2019). Embarrassingly Shallow Autoencoders for Sparse Data. *WWW 2019*.
- Paudel, B. et al. (2016). Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications. *TiiS*.
- Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets: History and Context. *TiiS 5(4)*.

## License

Academic use. MovieLens data subject to [GroupLens license terms](https://grouplens.org/datasets/movielens/).
