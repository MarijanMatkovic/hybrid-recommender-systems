"""
Run every Week 1-2 experiment end-to-end.

Ordering is cheap-to-expensive so failures surface early:

    1. graph_source_ablation  (needs all four graph sources)
    2. gamma_sensitivity      (Laplacian sweep across gamma)
    3. head_tail_analysis     (per-popularity-bucket NDCG)
    4. edlae_experiments      (closed-form, fast)
    5. slim_experiments       (slow -- per-item CD)

Example:
    python -m experiments.run_all --dataset ml-small --k 10
"""

from __future__ import annotations

import argparse
import time

from experiments import (
    gamma_sensitivity,
    graph_source_ablation,
    head_tail_analysis,
    slim_experiments,
    edlae_experiments,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml-small',
                   choices=['ml-small', 'ml-1m', 'netflix-prize'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--skip', type=str, default='',
                   help='Comma-separated list of experiments to skip')
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    order = [
        ('graph_source_ablation',
         lambda: graph_source_ablation.run(dataset=args.dataset, k=args.k)),
        ('gamma_sensitivity',
         lambda: gamma_sensitivity.run(dataset=args.dataset, k=args.k)),
        ('head_tail_analysis',
         lambda: head_tail_analysis.run(dataset=args.dataset, k=args.k)),
        ('edlae',
         lambda: edlae_experiments.run(dataset=args.dataset, k=args.k)),
        ('slim',
         lambda: slim_experiments.run(dataset=args.dataset, k=args.k)),
    ]

    t_global = time.time()
    for name, fn in order:
        if name in skip:
            print(f"\n>>> SKIP {name}")
            continue
        print(f"\n{'='*70}\n>>> {name}\n{'='*70}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {e!r}")
            raise
        print(f">>> {name} done in {(time.time()-t0)/60:.1f} min")

    print(f"\nAll experiments finished in "
          f"{(time.time()-t_global)/60:.1f} min")


if __name__ == '__main__':
    main()
