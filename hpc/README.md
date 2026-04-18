# Running the experiments on Supek (SRCE HPC)

Scripts in this folder submit the full experiment suite to the University
of Zagreb Supek supercomputer. Supek uses the **PBS Pro** scheduler
(`qsub`, `qstat`, `qdel`), not SLURM.

Reference: <https://wiki.srce.hr/spaces/NR/pages/121966084/>

## Files

| File                             | Purpose                                            |
|----------------------------------|----------------------------------------------------|
| `env_setup.sh`                   | One-time Python-env setup (run on the login node)  |
| `_prologue.sh`                   | Shared setup sourced by every PBS job              |
| `run_primary_multiseed.pbs`      | **PRIMARY**: full 6-family sweep, random 80/20 × 5 seeds (headline numbers) |
| `run_main_baselines.pbs`         | Secondary: same 6-family sweep, single temporal split (sanity check) |
| `run_edlae.pbs`                  | EDLAE + Laplacian-EDLAE sweep on ml-1m             |
| `run_slim.pbs`                   | SLIM + Laplacian-SLIM sweep on ml-1m (long!)       |
| `run_gamma_sensitivity.pbs`      | NDCG vs gamma, log-scale                           |
| `run_graph_ablation.pbs`         | RP3beta vs ItemKNN vs P3alpha vs binary adjacency  |
| `run_head_tail.pbs`              | Head / torso / tail NDCG breakdown (single γ)      |
| `run_head_tail_gamma.pbs`        | Head/tail NDCG gain across a γ grid (sweep plot)   |
| `run_edlae_multiseed.pbs`        | Multi-seed EDLAE/Laplacian-EDLAE for mean ± std    |
| `submit_all.sh`                  | `qsub`s every job in one go                        |
| `submit_symmetric.sh`            | Tonight's sym-Laplacian sweep + new diagnostics    |

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

| Job                          | CPUs | RAM  | Walltime | Why                                                          |
|------------------------------|------|------|----------|--------------------------------------------------------------|
| `run_primary_multiseed.pbs`  | 8    | 32GB | 8h       | 6 families × ~30 configs × 5 seeds (~900 fits on ml-1m)      |
| `run_main_baselines.pbs`     | 8    | 32GB | 2h       | Same grid, single temporal split (1 seed)                    |
| `run_edlae.pbs`              | 8    | 32GB | 2h       | Closed-form, cheap                                           |
| `run_graph_ablation.pbs`     | 8    | 32GB | 6h       | Four graph sources x gamma grid                              |
| `run_gamma_sensitivity.pbs`  | 8    | 32GB | 6h       | Wide log-scale gamma grid (12 values)                        |
| `run_head_tail.pbs`          | 8    | 32GB | 2h       | Single model, per-bucket NDCG                                |
| `run_head_tail_gamma.pbs`    | 8    | 32GB | 4h       | ~6 γ values × 1 Laplacian fit ≈ 6 × single head/tail         |
| `run_edlae_multiseed.pbs`    | 8    | 32GB | 4h       | 3 models × N_SEEDS closed-form fits (~1-2 min each on ml-1m) |
| `run_slim.pbs`               | 16   | 64GB | 20h      | Coordinate descent per item (~3700 items on ml-1m)           |

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
