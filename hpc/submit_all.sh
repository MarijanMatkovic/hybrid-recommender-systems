#!/bin/bash
# =====================================================================
# Submit all experiment jobs to the Supek scheduler in one go.
#
# All five jobs are independent of each other -- they read the same
# dataset but write disjoint result files -- so we just fire them off
# and let PBS decide ordering. The scheduler will run what fits; the
# rest queue until resources free up.
#
# Typical turnaround (ml-1m):
#   run_edlae.pbs            ~5-30 min  (closed-form)
#   run_graph_ablation.pbs   ~1-2 h
#   run_gamma_sensitivity.pbs ~1-3 h
#   run_head_tail.pbs        ~15-30 min
#   run_slim.pbs             ~10-20 h   <-- long pole
#
# Usage (from the repo root on the Supek login node):
#     bash hpc/submit_all.sh
# =====================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs results

echo "============================================================"
echo "  Submitting Laplacian-EASE experiment suite to Supek"
echo "  Repo root: $PWD"
echo "  Date     : $(date -Is)"
echo "============================================================"

submit () {
    local script="$1"
    local label="$2"
    local jid
    jid=$(qsub "$script")
    printf "  %-28s %s\n" "$label" "$jid"
    echo "$jid"
}

echo ""
echo ">>> Submitting PRIMARY headline job (random 80/20 x 5 seeds)"
JID_PRIMARY=$(submit hpc/run_primary_multiseed.pbs "primary multi-seed baselines")
JID_MAIN=$(submit    hpc/run_main_baselines.pbs    "temporal-split baselines (sanity)")

echo ""
echo ">>> Submitting cheap jobs next"
JID_EDLAE=$(submit hpc/run_edlae.pbs            "EDLAE sweep")
JID_HEAD=$(submit  hpc/run_head_tail.pbs        "head-vs-tail analysis")

echo ""
echo ">>> Submitting medium-length jobs"
JID_GRAPH=$(submit hpc/run_graph_ablation.pbs   "graph-source ablation")
JID_GAMMA=$(submit hpc/run_gamma_sensitivity.pbs "gamma sensitivity")

echo ""
echo ">>> Submitting the long one"
JID_SLIM=$(submit  hpc/run_slim.pbs             "SLIM + SLIM-Laplacian")

echo ""
echo "============================================================"
echo "  All jobs submitted. Useful commands:"
echo ""
echo "    qstat -u \$USER            # your jobs"
echo "    qstat -fx <JOBID>          # full details for one job"
echo "    tail -f logs/slim.log      # follow a running job's output"
echo "    qdel <JOBID>               # cancel a job"
echo ""
echo "  Results will appear under results/ as each job finishes."
echo "============================================================"
