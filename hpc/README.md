# Running the experiments on Supek (SRCE HPC)

Scripts in this folder submit the full experiment suite to the University
of Zagreb Supek supercomputer. Supek uses the **PBS Pro** scheduler
(`qsub`, `qstat`, `qdel`), not SLURM.

Reference: <https://wiki.srce.hr/spaces/NR/pages/121966084/>

## Files

| File                                   | Purpose                                                                      |
|----------------------------------------|------------------------------------------------------------------------------|
| `env_setup.sh`                         | One-time Python-env setup (run on the login node)                            |
| `_prologue.sh`                         | Shared setup sourced by every PBS job                                        |
| **Baselines**                          |                                                                              |
| `run_primary_multiseed.pbs`            | **PRIMARY**: full 6-family sweep, random 80/20 × 5 seeds (headline numbers) |
| `run_main_baselines.pbs`               | Secondary: same 6-family sweep, single temporal split (sanity check)        |
| **Temporal deep-dives**                |                                                                              |
| `run_edlae.pbs`                        | EDLAE + Laplacian-EDLAE sweep on ml-1m (temporal split)                     |
| `run_slim.pbs`                         | SLIM + Laplacian-SLIM sweep on ml-1m, temporal split (long!)                |
| `run_gamma_sensitivity.pbs`            | NDCG vs γ, log-scale (temporal)                                              |
| `run_graph_ablation.pbs`               | RP3beta vs ItemKNN vs P3alpha vs binary adjacency (temporal)                 |
| `run_head_tail.pbs`                    | Head / torso / tail NDCG breakdown, single γ (temporal)                     |
| `run_head_tail_gamma.pbs`              | Head/tail NDCG gain across a γ grid (temporal sweep plot)                   |
| `run_b_matrix_analysis.pbs`            | B-matrix structural metrics vs γ: ‖B‖₁, ‖B‖₂, sparsity, W-alignment        |
| **Primary-protocol experiments**       |                                                                              |
| `run_edlae_multiseed.pbs`              | Multi-seed EDLAE/Laplacian-EDLAE for mean ± std (ml-small)                  |
| `run_edlae_multiseed_ml1m.pbs`         | Same as above but for ml-1m (slower)                                         |
| `run_edlae_null_primary.pbs`           | EDLAE null hypothesis: dropout vs explicit Laplacian (primary, 5 seeds)     |
| `run_slim_primary.pbs`                 | SLIM + Laplacian-SLIM, primary protocol (submit once per seed with SPLIT_SEED=N) |
| `run_gs_ease_primary.pbs`              | GS-EASE (Graph-Shrunk EASE) vs Lap-EASE, primary × 5 seeds                  |
| `run_graph_ablation_primary.pbs`       | Graph source ablation (rp3β/p3α/itemknn/binary) on primary × 5 seeds        |
| `run_spectral_filter_primary.pbs`      | Spectral filter L² vs standard Laplacian L, primary × 5 seeds               |
| `run_head_tail_primary.pbs`            | Head/tail user-activity breakdown, single primary seed                       |
| `run_head_tail_primary_multiseed.pbs`  | Head/tail user-activity for Lap-EASE(γ=3) across all 5 primary seeds        |
| `run_slim_pooled_wilcoxon.pbs`         | Pooled ~30k Wilcoxon: SLIM-Lap(γ=1) vs EASE, 5 primary seeds (long!)       |
| **Netflix Prize jobs**                 | (counterparts of ml-1m jobs; output filenames carry `netflix-prize` tag)    |
| `run_primary_multiseed_netflix.pbs`    | PRIMARY 6-family sweep on Netflix, random 80/20 x 5 seeds                   |
| `run_main_baselines_netflix.pbs`       | Temporal sanity check on Netflix (single deterministic split)               |
| `run_gamma_sensitivity_netflix.pbs`    | NDCG vs gamma on Netflix (temporal)                                         |
| `run_graph_ablation_netflix.pbs`       | Graph source ablation on Netflix (temporal)                                 |
| `run_head_tail_netflix.pbs`            | Head/tail item-popularity NDCG, single gamma (Netflix temporal)             |
| `run_head_tail_primary_multiseed_netflix.pbs` | Head/tail user-activity for Lap-EASE(γ=3), 5 primary seeds (Netflix)  |
| `run_edlae_multiseed_netflix.pbs`      | Multi-seed EDLAE / Lap-EDLAE on Netflix (5 primary seeds)                   |
| `run_gs_ease_primary_netflix.pbs`      | GS-EASE primary 5-seed sweep on Netflix                                     |
| **Closed-form roadmap (items 1-9)**    | (compass_artifact_*.md highest-impact strategies)                            |
| `run_spectral_kernels_primary.pbs`     | Smola-Kondor menu (heat, p-step, reg-Lap, inv-cos) x graph source (item 5)   |
| `run_closed_form_extensions_primary.pbs` | Multi-Lap + Gram-shrink + Mahalanobis-shrink combined sweep (items 3, 4, 6) |
| `run_poly_filter_primary.pbs`          | Turbo-CF / Chebyshev pre-processing of X then EASE (item 7)                  |
| `run_cease_primary.pbs`                | CEASE + Add-EASE side-info baselines (item 2, mandatory)                     |
| `run_score_fusion_primary.pbs`         | RRF + CombMNZ + inverse-variance fusion ablation (items 1, 9)                |
| **Closed-form roadmap (items 10-15)**  |                                                                              |
| `run_competitor_baselines_primary.pbs` | L^3AE + DAN + SVD-AE (with/without filter) competitor baselines (10, 12, 13) |
| `run_heat_gram_primary.pbs`            | Heat / PPR pre-smoothed Gram sweep (item 14)                                 |
| `run_signed_laplacian_primary.pbs`     | Signed-Laplacian EASE on liked/disliked co-occurrence (item 15)              |
| `run_retrofit_ease_primary.pbs`        | Faruqui retrofitting of EASE B columns over genre graph (item 11)            |
| **Utilities**                          |                                                                              |
| `submit_all.sh`                        | `qsub`s every job in one go                                                  |
| `submit_symmetric.sh`                  | Sym-Laplacian sweep + diagnostics                                            |

## 1. Log in

```bash
ssh <username>@supek.srce.hr
```

You land on a **login node**. Do NOT run experiments here -- they must
go through the scheduler. The login node is only for editing files,
cloning repos, submitting jobs, and reading results.

## 2. Clone the repository

```bash
cd ~                              # or wherever you keep code
git clone <your-repo-url> diplomski
cd diplomski
```

Everything below assumes you're in the repo root.

## 3. One-time environment setup

```bash
bash hpc/env_setup.sh
```

This loads `cray-python/3.11.7` (Cray's build with numpy/scipy linked
against LibSci), creates a virtualenv at `~/diplomski_env` with
`--system-site-packages` so it inherits the tuned numpy/scipy, then
pip-installs scikit-learn, pandas, matplotlib, pytest on top. Finally
it runs `pytest` as a sanity check.

If you want a different Python, list available modules with
`module spider python` and override:

```bash
PYTHON_MODULE=utils/python/3.12.2 bash hpc/env_setup.sh
```

On Supek the available options are `cray-python/{3.9.13.1, 3.10.10,
3.11.7}` and `utils/python/{2.7.18, 3.12.2}`. Prefer the `cray-python`
family -- the `utils/python` builds are plain CPython with unoptimised
numpy.

## 4. Submit the experiments

### All at once

```bash
bash hpc/submit_all.sh
```

### Or individually

```bash
# Primary (headline) numbers — MUST run this:
qsub hpc/run_primary_multiseed.pbs

# Secondary (temporal) sanity check, cheap:
qsub hpc/run_main_baselines.pbs

# Per-experiment deep-dives:
qsub hpc/run_edlae.pbs
qsub hpc/run_graph_ablation.pbs
qsub hpc/run_gamma_sensitivity.pbs
qsub hpc/run_head_tail.pbs
qsub hpc/run_slim.pbs
```

`run_primary_multiseed.pbs` is the single most important job: it
produces the six-family leaderboard + pooled Wilcoxon tests across five
random splits, which is what the thesis headline table is computed
from. `run_main_baselines.pbs` is the temporal-split sanity check and
is cheap enough to always run alongside it.

### With custom hyperparameters

`run_head_tail.pbs` accepts environment variables via `qsub -v`:

```bash
qsub -v EASE_LAMBDA=200,GAMMA=75,RP3_BETA=0.3 hpc/run_head_tail.pbs
```

`run_head_tail_gamma.pbs` (γ-sweep mode) accepts the same vars plus a
`GAMMAS` list and a `BUCKET_BY` toggle:

```bash
qsub -v EASE_LAMBDA=500,GAMMAS="0,3,10,30,50,75",BUCKET_BY=user_activity \
    hpc/run_head_tail_gamma.pbs
```

`run_edlae_multiseed.pbs` reports mean ± std over random splits:

```bash
qsub -v N_SEEDS=5,DROPOUT=0.5,GAMMA=50 hpc/run_edlae_multiseed.pbs
# or with explicit seeds:
qsub -v SPLIT_SEEDS="0,1,2,7,13",DROPOUT=0.5,GAMMA=50 \
    hpc/run_edlae_multiseed.pbs
```

Every experiment script also accepts `--ks "10,20"` (configurable cut-off
list). The default produces both `NDCG@10` and `NDCG@20` columns in the
output CSVs — no need to override unless you want more cutoffs.

## 5. Monitor

```bash
qstat -u $USER           # list your jobs
qstat -fx <JOBID>        # detailed info for one job
tail -f logs/slim.log    # follow a running job's stdout
qdel <JOBID>             # cancel a job
```

Job states: `Q` = queued, `R` = running, `F` = finished, `H` = held.

## 6. Retrieve results

Results land in `results/` and log output in `logs/`. Pull them down
with rsync:

```bash
rsync -avz supek.srce.hr:~/diplomski/results/ ./results/
rsync -avz supek.srce.hr:~/diplomski/logs/    ./logs/
```

## Resource profile

| Job                                   | CPUs | RAM  | Walltime | Why                                                                                    |
|---------------------------------------|------|------|----------|----------------------------------------------------------------------------------------|
| `run_primary_multiseed.pbs`           | 8    | 32GB | 40h      | 6 families × ~30 configs × 5 seeds (~900 fits on ml-1m)                               |
| `run_main_baselines.pbs`              | 8    | 32GB | 6h       | Same grid, single temporal split (1 seed); bumped from 2h after walltime-kill          |
| `run_edlae.pbs`                       | 8    | 32GB | 2h       | Closed-form, cheap                                                                     |
| `run_graph_ablation.pbs`              | 8    | 32GB | 6h       | Four graph sources × gamma grid (temporal)                                             |
| `run_gamma_sensitivity.pbs`           | 8    | 32GB | 6h       | Wide log-scale gamma grid, 12 values (temporal)                                        |
| `run_head_tail.pbs`                   | 8    | 32GB | 2h       | Single model, per-bucket NDCG (temporal)                                               |
| `run_head_tail_gamma.pbs`             | 8    | 32GB | 4h       | ~6 γ values × 1 Laplacian fit ≈ 6 × single head/tail (temporal)                       |
| `run_b_matrix_analysis.pbs`           | 8    | 32GB | 2h       | B-matrix structural metrics vs γ (temporal, single run)                                |
| `run_edlae_multiseed.pbs`             | 8    | 32GB | 4h       | 3 models × N_SEEDS closed-form fits (ml-small)                                         |
| `run_edlae_multiseed_ml1m.pbs`        | 8    | 32GB | 8h       | Same as above on ml-1m (heavier matrix inversion)                                      |
| `run_edlae_null_primary.pbs`          | 8    | 32GB | 6h       | EDLAE null: 5 seeds × (EDLAE + Lap-EASE) closed-form fits                             |
| `run_slim_primary.pbs`                | 16   | 64GB | 10h      | 1 primary seed: 4 SLIM configs + 5 SLIM-Lap gammas (~8h per seed)                     |
| `run_gs_ease_primary.pbs`             | 8    | 32GB | 40h      | GS-EASE: 5 seeds × (gamma, W_source) grid + Lap-EASE reference                        |
| `run_graph_ablation_primary.pbs`      | 8    | 32GB | 40h      | Graph source ablation: 5 seeds × 4 sources × gamma grid                               |
| `run_spectral_filter_primary.pbs`     | 8    | 32GB | 40h      | L² vs L spectral filter: 5 seeds × gamma grid                                         |
| `run_head_tail_primary.pbs`           | 8    | 32GB | 2h       | Head/tail user-activity, single primary seed                                           |
| `run_head_tail_primary_multiseed.pbs` | 8    | 32GB | 2h       | Head/tail user-activity for Lap-EASE(γ=3), all 5 primary seeds (~23 min)              |
| `run_slim_pooled_wilcoxon.pbs`        | 16   | 64GB | 4h       | Pooled Wilcoxon SLIM-Lap vs EASE: 5 seeds × (EASE + SLIM + SLIM-Lap) (~95 min)        |
| `run_slim.pbs`                        | 16   | 64GB | 20h      | Coordinate descent per item (~3700 items on ml-1m, temporal)                          |
| `run_primary_multiseed_netflix.pbs`   | 16   | 64GB | 72h      | Same grid leaner than ml-1m; 17.7k items => EASE inversion ~125x slower per fit       |
| `run_main_baselines_netflix.pbs`      | 16   | 64GB | 12h      | Single temporal split, all 6 families on Netflix                                       |
| `run_gamma_sensitivity_netflix.pbs`   | 16   | 64GB | 12h      | Wide log-scale gamma grid on Netflix (temporal)                                        |
| `run_graph_ablation_netflix.pbs`      | 16   | 64GB | 12h      | Graph source ablation on Netflix (temporal)                                            |
| `run_head_tail_netflix.pbs`           | 16   | 64GB | 8h       | Single-gamma head/tail breakdown on Netflix                                            |
| `run_head_tail_primary_multiseed_netflix.pbs` | 16 | 64GB | 12h | Lap-EASE(γ=3), 5 primary seeds, head/tail user-activity                                |
| `run_edlae_multiseed_netflix.pbs`     | 16   | 64GB | 12h      | EDLAE closed-form, 5 primary seeds; Lap-EDLAE comparison                                |
| `run_gs_ease_primary_netflix.pbs`     | 16   | 64GB | 72h      | GS-EASE primary 5-seed (gamma, W_source) grid                                          |

If SLIM keeps hitting walltime, shorten the gamma grid or drop
`--n_iter` in `experiments/slim_experiments.py`.

## Troubleshooting

**`qsub: Job rejected by all possible destinations`**
The queue name (`-q cpu`) doesn't exist on your cluster. Run
`qstat -Q` to list available queues and update the `#PBS -q` line.

**`module: command not found`**
The login shell isn't sourcing the Environment Modules system. Try
`source /etc/profile.d/modules.sh` before `bash hpc/env_setup.sh`.

**`ModuleNotFoundError: No module named 'experiments'`**
You submitted from a directory other than the repo root. `qsub` from
the repo root; the scripts use `$PBS_O_WORKDIR` to `cd` back to where
you submitted.

**Job crashes instantly, log says "No such file: ml-1m"**
Download the MovieLens data first:
```bash
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
cd ..
```
