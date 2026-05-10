"""
Learned-B diagnostics: sparsity, graph-alignment, condition number.

Addresses two questions that the NDCG-vs-gamma plot alone doesn't answer:

1. **Sparsity / structure of B.** EASE's pitch is interpretability via the
   learned item-item matrix B. Does the Laplacian penalty change B's
   distribution -- does it make B denser, sparser, more "graph-like"?
   We report the fraction of |B_ij| above a handful of thresholds plus a
   histogram overlay at a few gammas.

2. **Graph alignment.** Intuition: the penalty ``tr(B^T L B)`` pushes B
   towards the nullspace of L, i.e. slowly-varying signals over the graph
   W. If the mechanism is working as advertised, the support of B should
   align more with the support of W as gamma grows:
     - Pearson correlation of entry-wise |B| and W.
     - Jaccard of the per-row top-K support of B with the per-row top-K
       of W (both off-diagonal).
     - Mean |B_ij| on W-edges vs non-edges.

3. **Condition number.** EASE solves ``(X^T X + λI)^{-1}``; the Laplacian
   variant solves ``(X^T X + λI + γ L)^{-1}``. At large γ the regularised
   Gram matrix can become ill-conditioned, which is one candidate
   explanation for the NDCG collapse at the high end of the gamma curve
   (rather than hand-waving "γ too big"). We report ``np.linalg.cond``
   plus min/max eigenvalues of the solved system.

Writes:
    results/b_matrix_analysis/b_matrix_<dataset>[<suffix>].csv
    results/b_matrix_analysis/b_matrix_<dataset>[<suffix>]_overview.png
    results/b_matrix_analysis/b_matrix_<dataset>[<suffix>]_hist.png

Example:
    python -m experiments.b_matrix_analysis --dataset ml-small --k 10
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
from scipy.sparse import csr_matrix

from evaluation.metrics import evaluate_at_ks
from models import HybridEASE_RP3beta

from experiments._shared import (
    ensure_results_dir,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Diagnostics on the learned B matrix
# ---------------------------------------------------------------------------

def sparsity_stats(B: np.ndarray,
                   thresholds=(1e-4, 1e-3, 1e-2, 1e-1)) -> dict:
    """Distribution summary of |B_ij| (off-diagonal).

    The diagonal is masked out because EASE's ``diag(B) = 0`` constraint
    makes it structurally zero -- including it would confound the true
    sparsity of the off-diagonal weights.
    """
    off = ~np.eye(B.shape[0], dtype=bool)
    absB = np.abs(B[off])
    n = absB.size
    out = {
        'B_off_diag_n_entries': int(n),
        'B_l1': float(absB.sum()),
        'B_l2': float(np.sqrt((absB ** 2).sum())),
        'B_mean_abs': float(absB.mean()),
        'B_median_abs': float(np.median(absB)),
        'B_max_abs': float(absB.max()) if n else 0.0,
    }
    for t in thresholds:
        out[f'B_frac_above_{t:g}'] = float((absB > t).sum() / max(n, 1))
    return out


def _topk_support_per_row(M: np.ndarray, k: int,
                          use_abs: bool = True) -> np.ndarray:
    """Return a boolean mask of shape (n, n): True where column j is in
    row i's top-k. Diagonal is forced to False."""
    n = M.shape[0]
    k = min(k, max(n - 1, 1))
    scores = np.abs(M) if use_abs else M
    np.fill_diagonal(scores, -np.inf)  # exclude self
    # Top-k indices per row.
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True
    np.fill_diagonal(mask, False)
    return mask


def graph_alignment_stats(B: np.ndarray,
                          W: csr_matrix | np.ndarray,
                          topK: int = 20) -> dict:
    """How much does B's support agree with W's?

    Three complementary views:

    * Entry-wise Pearson correlation between |B| and W (off-diagonal,
      dense). Robust signal when both matrices are dense-enough.
    * Per-row Jaccard of the top-K supports. Captures ranking overlap,
      which is what the recommender actually uses at inference time.
    * Mean |B_ij| on W-edges vs non-edges: the ratio tells us whether B
      concentrates mass on the graph or not.
    """
    if hasattr(W, 'toarray'):
        W_dense = np.asarray(W.toarray(), dtype=float)
    else:
        W_dense = np.asarray(W, dtype=float)
    n = B.shape[0]
    off = ~np.eye(n, dtype=bool)

    absB = np.abs(B)

    # Off-diagonal Pearson correlation.
    b_flat = absB[off]
    w_flat = W_dense[off]
    if b_flat.std() < 1e-12 or w_flat.std() < 1e-12:
        pearson = float('nan')
    else:
        pearson = float(np.corrcoef(b_flat, w_flat)[0, 1])

    # Per-row top-K Jaccard.
    B_top = _topk_support_per_row(absB, topK, use_abs=False)  # already abs
    W_top = _topk_support_per_row(W_dense, topK, use_abs=False)
    inter = (B_top & W_top).sum(axis=1)
    union = (B_top | W_top).sum(axis=1)
    # Rows where neither has any top-K (e.g. all-zero row) get Jaccard=0.
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    # Mean magnitude of B on W-edges vs non-edges.
    w_edge = (W_dense > 0) & off
    w_nonedge = (~w_edge) & off
    b_on_edge = absB[w_edge].mean() if w_edge.any() else 0.0
    b_off_edge = absB[w_nonedge].mean() if w_nonedge.any() else 0.0
    ratio = (b_on_edge / b_off_edge) if b_off_edge > 0 else float('inf')

    return {
        'align_pearson_absB_W':        float(pearson),
        'align_jaccard_topK_mean':     float(jaccard.mean()),
        'align_jaccard_topK_median':   float(np.median(jaccard)),
        'align_mean_absB_on_W_edge':   float(b_on_edge),
        'align_mean_absB_off_W_edge':  float(b_off_edge),
        'align_edge_to_nonedge_ratio': float(ratio),
        'align_topK':                  int(topK),
    }


def condition_number_stats(G_raw: np.ndarray, lam: float,
                           gamma: float,
                           L_scaled: np.ndarray | None) -> dict:
    """Spectral properties of the EASE-style solve matrix.

    For gamma=0 (plain EASE) and for gamma>0 (Laplacian-EASE), reports
    ``numpy.linalg.cond`` of ``G_raw + λI + γL_scaled`` plus the smallest
    and largest eigenvalues separately (diagnostic: a blow-up in cond
    can come from the top or the bottom).
    """
    n = G_raw.shape[0]
    A = G_raw.copy()
    idx = np.diag_indices(n)
    A[idx] += lam
    if L_scaled is not None and gamma != 0.0:
        A = A + gamma * L_scaled
    # A is symmetric PSD by construction -- use eigvalsh for stability.
    evals = np.linalg.eigvalsh(A)
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    cond = float(lam_max / lam_min) if lam_min > 0 else float('inf')
    return {
        'cond_number':     cond,
        'eig_min':         lam_min,
        'eig_max':         lam_max,
        'eig_min_log10':   float(np.log10(lam_min)) if lam_min > 0 else -np.inf,
        'eig_max_log10':   float(np.log10(lam_max)) if lam_max > 0 else -np.inf,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(dataset='ml-small', k=10,
        ease_lambda=None, gammas=None,
        graph_source='rp3beta', rp3_beta=0.6, rp3_topK=200,
        normalise='none',
        sparsity_thresholds=(1e-4, 1e-3, 1e-2, 1e-1),
        alignment_topK=20,
        hist_gammas=None,
        ks=(10, 20),
        out_dir=None):
    if ease_lambda is None:
        ease_lambda = 500 if dataset == 'ml-1m' else 200
    if gammas is None:
        gammas = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    else:
        # Always anchor the sweep at γ=0 (plain EASE).
        gammas = sorted(set([0.0] + list(gammas)))
    ks = tuple(sorted(set(list(ks) + [k])))

    out_dir = ensure_results_dir('b_matrix_analysis'
                                 if out_dir is None else out_dir)

    train, test_positive, _ = load_dataset(dataset)

    # Keep a few (B, gamma) around for the histogram overlay at the end.
    if hist_gammas is None:
        # 4 evenly-spread gammas on the log scale, plus gamma=0.
        if len(gammas) <= 4:
            hist_gammas = list(gammas)
        else:
            nonzero = [g for g in gammas if g > 0]
            picks = [nonzero[i] for i in
                     np.linspace(0, len(nonzero) - 1, 3, dtype=int)]
            hist_gammas = [0.0] + picks
    hist_snapshots = {}

    rows = []
    for gamma in gammas:
        print(f"\n[gamma={gamma:<8.3f}]  normalise={normalise}")
        t0 = time.time()
        model = HybridEASE_RP3beta()
        if gamma == 0.0:
            model.fit(train, method='score', fusion_alpha=1.0,
                      ease_lambda=ease_lambda, rp3_alpha=1.0,
                      rp3_beta=rp3_beta, rp3_topK=rp3_topK)
            # For alignment we still need W at this gamma. Build it the
            # same way _fit_laplacian_ease would, so the gamma=0 anchor
            # is apples-to-apples with the rest of the sweep.
            from models.graph_sources import build_graph, build_laplacian
            W = build_graph(model.ease.X, source=graph_source,
                            topK=rp3_topK, rp3_beta=rp3_beta,
                            implicit=True)
            L_raw, _ = build_laplacian(W, normalise=normalise)
            G_raw = model.ease.X.T.dot(model.ease.X).toarray()
            g_diag_mean = float(np.mean(np.diag(G_raw)))
            l_diag_mean = float(np.mean(np.diag(L_raw)))
            L_scaled = (L_raw * (g_diag_mean / l_diag_mean)
                        if l_diag_mean > 0 else L_raw)
            # hybrid.fit(method='score', fusion_alpha=1.0) already sets
            # self.pred to make_pred(X, B); no eager override needed.
        else:
            model.fit(train, method='laplacian',
                      ease_lambda=ease_lambda, rp3_alpha=1.0,
                      rp3_beta=rp3_beta, rp3_topK=rp3_topK,
                      graph_reg_gamma=gamma,
                      graph_source=graph_source,
                      laplacian_normalise=normalise)
            W = model.rp3.W
            L_scaled = model.L_scaled
            G_raw = model.ease.X.T.dot(model.ease.X).toarray()

        res = evaluate_at_ks(model, train, test_positive, ks=ks)
        dt = time.time() - t0

        sp = sparsity_stats(model.ease.B,
                            thresholds=sparsity_thresholds)
        al = graph_alignment_stats(model.ease.B, W, topK=alignment_topK)
        cn = condition_number_stats(G_raw, ease_lambda, gamma, L_scaled)

        print(f"  NDCG@{k}={res[f'NDCG@{k}']:.4f}  "
              f"cond={cn['cond_number']:.2e}  "
              f"frac|B|>1e-3={sp.get('B_frac_above_0.001', float('nan')):.4f}  "
              f"align_jaccard={al['align_jaccard_topK_mean']:.3f}  "
              f"({dt:.1f}s)")

        row = {
            'dataset': dataset,
            'ease_lambda': ease_lambda,
            'graph_source': graph_source,
            'rp3_beta': rp3_beta,
            'rp3_topK': rp3_topK,
            'normalise': normalise,
            'gamma': gamma,
            'train_time_s': dt,
            'n_items': int(model.ease.B.shape[0]),
        }
        for kk in ks:
            for m in ('NDCG', 'MAP', 'HitRate', 'Recall'):
                row[f'{m}@{kk}'] = res[f'{m}@{kk}']
        row['NDCG@k'] = res[f'NDCG@{k}']
        row.update(sp)
        row.update(al)
        row.update(cn)
        rows.append(row)

        if gamma in hist_gammas:
            off = ~np.eye(model.ease.B.shape[0], dtype=bool)
            hist_snapshots[gamma] = np.abs(model.ease.B[off]).copy()

    df = pd.DataFrame(rows)
    suffix = '_sym' if normalise == 'sym' else ''
    csv_path = out_dir / f'b_matrix_{dataset}{suffix}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # ---- Overview plot: cond, sparsity, alignment all on one figure ----
    _plot_overview(df, k=k, out_path=out_dir /
                   f'b_matrix_{dataset}{suffix}_overview.png',
                   dataset=dataset, normalise=normalise)

    # ---- Histogram overlay of |B_ij| at a few gammas ----
    if hist_snapshots:
        _plot_hist(hist_snapshots,
                   out_path=out_dir /
                   f'b_matrix_{dataset}{suffix}_hist.png',
                   dataset=dataset, normalise=normalise)

    return df


def _plot_overview(df: pd.DataFrame, k: int, out_path: Path,
                   dataset: str, normalise: str) -> None:
    """Four-panel summary: NDCG, condition number, sparsity, alignment.
    Gamma=0 shows up as the leftmost point on each log-scale panel (we
    replace it with ``min(gammas[>0]) / 3`` so it plots without warnings,
    and keep an explicit label)."""
    df = df.sort_values('gamma').copy()
    nonzero = df[df['gamma'] > 0]['gamma']
    x_anchor = float(nonzero.min()) / 3.0 if len(nonzero) else 1.0
    df['gamma_plot'] = df['gamma'].where(df['gamma'] > 0, x_anchor)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(df['gamma_plot'], df[f'NDCG@{k}'], marker='o')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$ (log; γ=0 plotted at left)')
    ax.set_ylabel(f'NDCG@{k}')
    ax.set_title('Retrieval quality')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    ax = axes[0, 1]
    ax.plot(df['gamma_plot'], df['cond_number'], marker='o',
            color='tab:red', label='cond(A)')
    ax.plot(df['gamma_plot'], df['eig_max'], marker='^',
            color='tab:orange', ls='--', label=r'$\lambda_{\max}$')
    ax.plot(df['gamma_plot'], df['eig_min'], marker='v',
            color='tab:green', ls='--', label=r'$\lambda_{\min}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('value (log)')
    ax.set_title(r'Condition number of $X^\top X + \lambda I + \gamma L$')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    ax = axes[1, 0]
    # Plot a couple of sparsity thresholds if available.
    for col, lbl in [('B_frac_above_0.0001', r'$|B|>10^{-4}$'),
                     ('B_frac_above_0.001',  r'$|B|>10^{-3}$'),
                     ('B_frac_above_0.01',   r'$|B|>10^{-2}$')]:
        if col in df.columns:
            ax.plot(df['gamma_plot'], df[col], marker='o', label=lbl)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('fraction of off-diagonal |B_ij|')
    ax.set_title('B density by threshold')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    ax = axes[1, 1]
    ax.plot(df['gamma_plot'], df['align_jaccard_topK_mean'],
            marker='o', label='Jaccard(top-K rows)')
    ax.plot(df['gamma_plot'], df['align_pearson_absB_W'],
            marker='s', label=r'Pearson($|B|$, $W$)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel('alignment')
    ax.set_title('Graph alignment of B with W')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    title_extra = ' [sym L]' if normalise == 'sym' else ''
    fig.suptitle(f'B-matrix diagnostics — {dataset}{title_extra}',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved overview plot to {out_path}")


def _plot_hist(snapshots: dict, out_path: Path,
               dataset: str, normalise: str) -> None:
    """Log-scale histogram overlay of |B_ij| at a handful of gammas."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps = 1e-12
    for gamma in sorted(snapshots):
        vals = snapshots[gamma]
        # log bins; clamp zeros so they don't crash log10.
        vals = np.clip(vals, eps, None)
        ax.hist(np.log10(vals), bins=60, alpha=0.45,
                label=fr'$\gamma={gamma:g}$', density=True)
    ax.set_xlabel(r'$\log_{10}|B_{ij}|$ (off-diagonal)')
    ax.set_ylabel('density')
    title_extra = ' [sym L]' if normalise == 'sym' else ''
    ax.set_title(f'Distribution of |B_ij| vs γ — {dataset}{title_extra}')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved histogram plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ease_lambda', type=float, default=None)
    p.add_argument('--rp3_beta', type=float, default=0.6)
    p.add_argument('--rp3_topK', type=int, default=200)
    p.add_argument('--graph_source', default='rp3beta')
    p.add_argument('--gammas', type=str, default=None,
                   help='Comma-separated gamma values (optional).')
    p.add_argument('--normalise', default='none',
                   choices=['none', 'sym'])
    p.add_argument('--alignment_topK', type=int, default=20,
                   help='Per-row top-K used for the Jaccard alignment '
                        'score between B and W. Default: 20.')
    p.add_argument('--hist_gammas', type=str, default=None,
                   help='Gammas to snapshot for the |B_ij| histogram '
                        '(comma-separated). Default: auto-pick 3 plus γ=0.')
    p.add_argument('--ks', type=str, default='10,20')
    args = p.parse_args()

    gammas = None
    if args.gammas:
        gammas = [float(x) for x in args.gammas.split(',')]
    hist_gammas = None
    if args.hist_gammas:
        hist_gammas = [float(x) for x in args.hist_gammas.split(',')]
    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())

    run(dataset=args.dataset, k=args.k,
        ease_lambda=args.ease_lambda, gammas=gammas,
        graph_source=args.graph_source,
        rp3_beta=args.rp3_beta, rp3_topK=args.rp3_topK,
        normalise=args.normalise,
        alignment_topK=args.alignment_topK,
        hist_gammas=hist_gammas,
        ks=ks)


if __name__ == '__main__':
    main()
