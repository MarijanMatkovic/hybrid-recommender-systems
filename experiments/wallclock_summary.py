"""
Aggregate training wall-clock times across every experiment CSV.

Scans ``results/**/*.csv``, pulls ``train_time_s`` out of each row, infers
the model family from either the ``model`` column (SLIM / EDLAE / EASE /
SLIM-Laplacian / EDLAE-Laplacian) or the experiment folder name (e.g.
``gamma_sensitivity`` and ``graph_source_ablation`` are both
"Laplacian-EASE" sweeps), and writes:

    results/wallclock_summary/wallclock_summary.csv
    results/wallclock_summary/wallclock_summary.md
    results/wallclock_summary/wallclock_summary.png

The reported statistic is the median ``train_time_s`` per (dataset, model)
bucket, with min/max/n so readers can see the spread.

The Markdown table is the main artefact -- paste it into the thesis to
make the "Laplacian regularisation is essentially free on top of these
closed-form models" claim explicit (EASE takes ~100 s, Laplacian-EDLAE
~3 s etc.).

Example:
    python -m experiments.wallclock_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments._shared import ensure_results_dir


# Some experiment CSVs don't carry an explicit ``model`` column; every row
# in those files corresponds to a single model family, which we infer from
# the experiment folder name. Files that *do* carry a ``model`` column
# (slim_, edlae_, edlae_multiseed_) are grouped by that column instead.
FOLDER_TO_MODEL = {
    'gamma_sensitivity':      'Laplacian-EASE',
    'graph_source_ablation':  'Laplacian-EASE',
    'head_tail_analysis':     None,  # per-bucket rows, no single model
}


def _model_family_from_row(row: pd.Series, folder: str) -> str | None:
    """Determine the model family for a single row.

    Precedence: explicit ``model`` column > folder default > None.
    """
    m = row.get('model')
    if isinstance(m, str) and m:
        return m
    return FOLDER_TO_MODEL.get(folder)


def _dataset_from_row(row: pd.Series, stem: str) -> str:
    """Prefer the ``dataset`` column, fall back to parsing the filename."""
    ds = row.get('dataset')
    if isinstance(ds, str) and ds:
        return ds
    # filename convention: ``<exp>_<dataset>[suffix].csv``
    for token in ('ml-1m', 'ml-small', 'synth'):
        if token in stem:
            return token
    return 'unknown'


def collect(results_root: Path) -> pd.DataFrame:
    """Return a long-form DataFrame with one row per training event."""
    records = []
    for csv_path in sorted(results_root.glob('*/*.csv')):
        folder = csv_path.parent.name
        stem = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover
            print(f"[skip] {csv_path}: {exc}")
            continue
        if 'train_time_s' not in df.columns:
            # Summary CSVs emitted by other scripts (e.g. the
            # edlae_multiseed ``*_summary.csv``) hold aggregated rows
            # with no per-fit wall-clock -- skip them here.
            continue
        for _, row in df.iterrows():
            t = row.get('train_time_s')
            if t is None or (isinstance(t, float) and np.isnan(t)):
                continue
            model = _model_family_from_row(row, folder)
            if model is None:
                # head_tail_analysis rows are per-bucket, not per-fit --
                # we don't have a single train_time for the model there.
                continue
            records.append({
                'dataset': _dataset_from_row(row, stem),
                'model': model,
                'experiment': folder,
                'source_csv': csv_path.name,
                'train_time_s': float(t),
            })
    return pd.DataFrame.from_records(records)


def summarise(long_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (dataset, model) with median/min/max/n."""
    if long_df.empty:
        return pd.DataFrame(columns=['dataset', 'model', 'n_fits',
                                     'median_s', 'min_s', 'max_s'])
    g = (long_df.groupby(['dataset', 'model'])['train_time_s']
         .agg(['median', 'min', 'max', 'count'])
         .reset_index()
         .rename(columns={'median': 'median_s',
                          'min': 'min_s',
                          'max': 'max_s',
                          'count': 'n_fits'}))
    # Sort: dataset asc, then median time desc (slowest first) so the
    # thesis table leads with the expensive baselines.
    g = g.sort_values(['dataset', 'median_s'], ascending=[True, False])
    return g[['dataset', 'model', 'n_fits', 'median_s', 'min_s', 'max_s']]


def format_markdown(summary: pd.DataFrame) -> str:
    """Render the summary as a GitHub-flavoured Markdown table."""
    if summary.empty:
        return "_No training events found under results/._\n"
    lines = [
        "| Dataset | Model | N fits | Median (s) | Min (s) | Max (s) |",
        "| --- | --- | ---:| ---:| ---:| ---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | "
            f"{int(row['n_fits'])} | "
            f"{row['median_s']:.2f} | "
            f"{row['min_s']:.2f} | "
            f"{row['max_s']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def plot_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: median training time per (dataset, model)."""
    if summary.empty:
        return
    datasets = sorted(summary['dataset'].unique())
    models = sorted(summary['model'].unique())
    x = np.arange(len(models))
    width = 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(models)), 4.5))
    for i, ds in enumerate(datasets):
        sub = summary[summary['dataset'] == ds].set_index('model')
        heights = [sub.loc[m, 'median_s'] if m in sub.index else 0.0
                   for m in models]
        ax.bar(x + (i - (len(datasets) - 1) / 2) * width, heights,
               width, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Median training time (s)')
    ax.set_yscale('log')
    ax.set_title('Training wall-clock per model family')
    ax.grid(True, which='both', axis='y', ls=':', alpha=0.5)
    ax.legend(frameon=False, title='dataset')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(results_root: str | Path = 'results',
        out_dir: str | Path | None = None) -> pd.DataFrame:
    results_root = Path(results_root)
    out_dir = ensure_results_dir('wallclock_summary'
                                 if out_dir is None else out_dir,
                                 root=results_root)

    long_df = collect(results_root)
    summary = summarise(long_df)

    csv_path = out_dir / 'wallclock_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    md_path = out_dir / 'wallclock_summary.md'
    md_path.write_text(format_markdown(summary), encoding='utf-8')
    print(f"Saved Markdown table to {md_path}")

    png_path = out_dir / 'wallclock_summary.png'
    plot_bars(summary, png_path)
    if png_path.exists():
        print(f"Saved bar chart to {png_path}")

    print("\n" + format_markdown(summary))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_root', default='results',
                   help='Directory to scan for experiment CSVs '
                        '(default: ./results).')
    args = p.parse_args()
    run(results_root=args.results_root)


if __name__ == '__main__':
    main()
