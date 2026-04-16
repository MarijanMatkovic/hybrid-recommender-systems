#!/bin/bash
# =====================================================================
# Tonight's sweep: symmetric normalised Laplacian + optimal
# hyperparameters discovered from earlier ml-1m runs.
#
# Sym normalised Laplacian (L = I - D^{-1/2} W D^{-1/2}) keeps all
# eigenvalues in [0, 2] so popular (high-degree) items no longer
# over-dominate the regulariser. This should help the torso/tail at
# the cost of a tiny hit in the head, if the head-bias hypothesis
# holds.
#
# Optimal ml-1m hyperparameters from prior sweeps (RP3beta graph):
#   lambda_ = 500
#   rp3_beta = 0.3
#   gamma = 50        (NDCG peak was 0.1208)
#
# We pair each sym job with a control 'none' run at the same settings
# (when the previous run didn't already use those exact hyperparams),
# so the sym-vs-none delta is apples-to-apples.
#
# Usage (from the repo root on the Supek login node):
#     bash hpc/submit_symmetric.sh
# =====================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs results

echo "============================================================"
echo "  Submitting symmetric-Laplacian experiment suite"
echo "  Repo root: $PWD"
echo "  Date     : $(date -Is)"
echo "============================================================"

submit () {
    # submit <script> <label> <name> <log> [<-v VAR=val>...]
    local script="$1"; shift
    local label="$1"; shift
    local name="$1"; shift
    local log="$1"; shift
    local jid
    # Remaining args are passed through as qsub options (typically -v).
    jid=$(qsub -N "$name" -o "$log" "$@" "$script")
    printf "  %-38s %s\n" "$label" "$jid"
}

# ---------------------------------------------------------------------
# Priority 1 — head/tail with BOTH Laplacian variants at the
# RP3beta sweet spot. This is the headline comparison: does the
# symmetric normalisation move the gain from head to tail?
# ---------------------------------------------------------------------
echo ""
echo ">>> Head-vs-tail (RP3beta @ λ=500, β=0.3, γ=50)"

submit hpc/run_head_tail.pbs \
    "head-tail RP3β [none]" \
    lap_ht_rp3_none \
    logs/head_tail_rp3_none.log \
    -v EASE_LAMBDA=500,GAMMA=50,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=none,N_BUCKETS=5

submit hpc/run_head_tail.pbs \
    "head-tail RP3β [sym]" \
    lap_ht_rp3_sym \
    logs/head_tail_rp3_sym.log \
    -v EASE_LAMBDA=500,GAMMA=50,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=sym,N_BUCKETS=5

# ---------------------------------------------------------------------
# Priority 2 — head/tail with the ItemKNN graph (which on ml-small
# was the strongest source at γ≈30, λ=100). Two variants again.
# ---------------------------------------------------------------------
echo ""
echo ">>> Head-vs-tail (ItemKNN @ λ=100, γ=30)"

submit hpc/run_head_tail.pbs \
    "head-tail ItemKNN [none]" \
    lap_ht_iknn_none \
    logs/head_tail_itemknn_none.log \
    -v EASE_LAMBDA=100,GAMMA=30,RP3_BETA=0.3,GRAPH_SOURCE=itemknn,NORMALISE=none,N_BUCKETS=5

submit hpc/run_head_tail.pbs \
    "head-tail ItemKNN [sym]" \
    lap_ht_iknn_sym \
    logs/head_tail_itemknn_sym.log \
    -v EASE_LAMBDA=100,GAMMA=30,RP3_BETA=0.3,GRAPH_SOURCE=itemknn,NORMALISE=sym,N_BUCKETS=5

# ---------------------------------------------------------------------
# Priority 3 — gamma-sensitivity curve for the sym variant. If the
# optimum shifts (e.g. to a much larger γ) that is itself a finding.
# ---------------------------------------------------------------------
echo ""
echo ">>> Gamma sensitivity (sym Laplacian, full log grid)"

submit hpc/run_gamma_sensitivity.pbs \
    "gamma sensitivity [sym]" \
    lap_gamma_sens_sym \
    logs/gamma_sensitivity_sym.log \
    -v NORMALISE=sym,GRAPH_SOURCE=rp3beta

# ---------------------------------------------------------------------
# Priority 4 — graph-source ablation with sym. Do the source
# rankings change when the Laplacian no longer over-penalises hubs?
# ---------------------------------------------------------------------
echo ""
echo ">>> Graph-source ablation (sym)"

submit hpc/run_graph_ablation.pbs \
    "graph-source ablation [sym]" \
    lap_graph_ablation_sym \
    logs/graph_ablation_sym.log \
    -v NORMALISE=sym

# ---------------------------------------------------------------------
# Priority 5 — EDLAE-Laplacian with sym (closed form, cheap).
# ---------------------------------------------------------------------
echo ""
echo ">>> EDLAE + Laplacian (sym)"

submit hpc/run_edlae.pbs \
    "EDLAE sweep [sym]" \
    lap_edlae_sym \
    logs/edlae_sym.log \
    -v NORMALISE=sym

# ---------------------------------------------------------------------
# Priority 6 — SLIM + Laplacian (long pole; ~10-20 h). Only submit
# if you want the full thesis-complete picture. Comment out otherwise.
# ---------------------------------------------------------------------
echo ""
echo ">>> SLIM + Laplacian (sym) — long pole"

submit hpc/run_slim.pbs \
    "SLIM sweep [sym]" \
    lap_slim_sym \
    logs/slim_sym.log \
    -v NORMALISE=sym

echo ""
echo "============================================================"
echo "  All sym-variant jobs submitted. Monitor with:"
echo ""
echo "    qstat -u \$USER"
echo "    tail -f logs/head_tail_rp3_sym.log"
echo ""
echo "  Results land under results/<exp>/*_sym.csv and .png"
echo "============================================================"
