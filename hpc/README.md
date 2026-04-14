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
| `run_edlae.pbs`                  | EDLAE + Laplacian-EDLAE sweep on ml-1m             |
| `run_slim.pbs`                   | SLIM + Laplacian-SLIM sweep on ml-1m (long!)       |
| `run_gamma_sensitivity.pbs`      | NDCG vs gamma, log-scale                           |
| `run_graph_ablation.pbs`         | RP3beta vs ItemKNN vs P3alpha vs binary adjacency  |
| `run_head_tail.pbs`              | Head / torso / tail NDCG breakdown                 |
| `submit_all.sh`                  | `qsub`s every job in one go                        |

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
qsub hpc/run_edlae.pbs
qsub hpc/run_graph_ablation.pbs
qsub hpc/run_gamma_sensitivity.pbs
qsub hpc/run_head_tail.pbs
qsub hpc/run_slim.pbs
```

### With custom hyperparameters

`run_head_tail.pbs` accepts environment variables via `qsub -v`:

```bash
qsub -v EASE_LAMBDA=200,GAMMA=75,RP3_BETA=0.3 hpc/run_head_tail.pbs
```

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

| Job                       | CPUs | RAM  | Walltime | Why                                                          |
|---------------------------|------|------|----------|--------------------------------------------------------------|
| `run_edlae.pbs`           | 8    | 32GB | 2h       | Closed-form, cheap                                           |
| `run_graph_ablation.pbs`  | 8    | 32GB | 6h       | Four graph sources x gamma grid                              |
| `run_gamma_sensitivity.pbs` | 8  | 32GB | 6h       | Wide log-scale gamma grid (12 values)                        |
| `run_head_tail.pbs`       | 8    | 32GB | 2h       | Single model, per-bucket NDCG                                |
| `run_slim.pbs`            | 16   | 64GB | 20h      | Coordinate descent per item (~3700 items on ml-1m)           |

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
