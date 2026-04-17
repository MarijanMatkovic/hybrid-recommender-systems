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

# ---------------------------------------------------------------------
# Priority 7 — Head/tail GAMMA SWEEP. This is the missing diagnostic:
# per-bucket NDCG gain as a function of gamma, on the same axes. Four
# variants so we can tell whether the head/tail tradeoff is monotonic
# in gamma, whether symmetric normalisation flips the curve, and
# whether user_activity buckets tell a different story than
# item_popularity (e.g. diagnosing the q5=0 NDCG mystery).
# ---------------------------------------------------------------------
echo ""
echo ">>> Head/tail gamma sweep (item-popularity buckets)"

submit hpc/run_head_tail_gamma.pbs \
    "head-tail γ-sweep RP3β [none]" \
    lap_ht_gamma_rp3_none \
    logs/head_tail_gamma_rp3_none.log \
    -v EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=none,N_BUCKETS=5,BUCKET_BY=item_popularity,GAMMAS="0,3,10,30,50,75"

submit hpc/run_head_tail_gamma.pbs \
    "head-tail γ-sweep RP3β [sym]" \
    lap_ht_gamma_rp3_sym \
    logs/head_tail_gamma_rp3_sym.log \
    -v EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=sym,N_BUCKETS=5,BUCKET_BY=item_popularity,GAMMAS="0,3,10,30,50,75"

echo ""
echo ">>> Head/tail gamma sweep (user-activity buckets — bucket audit)"

submit hpc/run_head_tail_gamma.pbs \
    "head-tail γ-sweep user-act [none]" \
    lap_ht_gamma_user_none \
    logs/head_tail_gamma_user_none.log \
    -v EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=none,N_BUCKETS=5,BUCKET_BY=user_activity,GAMMAS="0,3,10,30,50,75"

submit hpc/run_head_tail_gamma.pbs \
    "head-tail γ-sweep user-act [sym]" \
    lap_ht_gamma_user_sym \
    logs/head_tail_gamma_user_sym.log \
    -v EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=sym,N_BUCKETS=5,BUCKET_BY=user_activity,GAMMAS="0,3,10,30,50,75"

# ---------------------------------------------------------------------
# Priority 8 — Multi-seed EDLAE for mean ± std CIs. Since EDLAE's
# closed-form is deterministic given the training matrix, the variance
# comes entirely from the train/test split. 5 seeds is enough for
# tight 95% CIs; running both NORMALISE values so the paper can report
# EASE / EDLAE / EDLAE-Lap[none] / EDLAE-Lap[sym] side by side.
# ---------------------------------------------------------------------
echo ""
echo ">>> Multi-seed EDLAE (mean ± std over random splits)"

submit hpc/run_edlae_multiseed.pbs \
    "EDLAE multi-seed [none]" \
    lap_edlae_multiseed_none \
    logs/edlae_multiseed_none.log \
    -v N_SEEDS=5,DROPOUT=0.5,GAMMA=50,LAMBDA_=500,GRAPH_SOURCE=rp3beta,RP3_BETA=0.3,NORMALISE=none

submit hpc/run_edlae_multiseed.pbs \
    "EDLAE multi-seed [sym]" \
    lap_edlae_multiseed_sym \
    logs/edlae_multiseed_sym.log \
    -v N_SEEDS=5,DROPOUT=0.5,GAMMA=50,LAMBDA_=500,GRAPH_SOURCE=rp3beta,RP3_BETA=0.3,NORMALISE=sym

# ---------------------------------------------------------------------
# Priority 9 — B-matrix diagnostics (sparsity, graph alignment,
# condition number) across a gamma sweep. Answers the two reviewer
# questions that the NDCG-vs-gamma plot doesn't:
#   - does the Laplacian make B more 'graph-like'?
#   - is the NDCG collapse at large gamma numerical?
# Run both Laplacian variants since the story may differ.
# ---------------------------------------------------------------------
echo ""
echo ">>> B-matrix diagnostics (gamma sweep)"

submit hpc/run_b_matrix_analysis.pbs \
    "B-matrix diagnostics [none]" \
    lap_b_matrix_none \
    logs/b_matrix_none.log \
    -v DATASET=ml-1m,EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=none,GAMMAS="0,1,3,10,30,50,100,300,1000"

submit hpc/run_b_matrix_analysis.pbs \
    "B-matrix diagnostics [sym]" \
    lap_b_matrix_sym \
    logs/b_matrix_sym.log \
    -v DATASET=ml-1m,EASE_LAMBDA=500,RP3_BETA=0.3,GRAPH_SOURCE=rp3beta,NORMALISE=sym,GAMMAS="0,1,3,10,30,50,100,300,1000"

# ---------------------------------------------------------------------
# Priority 10 — Multi-seed EDLAE on ML-1M (the VALIDATION run). The
# earlier synth multi-seed job had std > delta-between-means because
# dataset noise dominated — useless. This one uses the real data with
# the sym + gamma=10 + dropout=0.75 config that was optimal in prior
# sweeps, so the 3-seed std we report actually reflects split noise
# rather than tiny-dataset noise. 3 seeds is enough: closed-form fits
# are fully deterministic given the train matrix.
# ---------------------------------------------------------------------
echo ""
echo ">>> Multi-seed EDLAE on ML-1M (validation of EDLAE-Lap gain)"

submit hpc/run_edlae_multiseed_ml1m.pbs \
    "EDLAE multi-seed ml-1m [sym]" \
    lap_edlae_multi_ml1m_sym \
    logs/edlae_multiseed_ml1m_sym.log \
    -v N_SEEDS=3,DROPOUT=0.75,GAMMA=10,LAMBDA_=500,GRAPH_SOURCE=rp3beta,RP3_BETA=0.3,NORMALISE=sym

echo ""
echo "============================================================"
echo "  All sym-variant jobs submitted. Monitor with:"
echo ""
echo "    qstat -u \$USER"
echo "    tail -f logs/head_tail_rp3_sym.log"
echo ""
echo "  Results land under results/<exp>/*_sym.csv and .png"
echo "  New diagnostics:"
echo "    - results/head_tail_analysis/head_tail_gamma_sweep_*.{csv,png}"
echo "    - results/edlae_multiseed/edlae_multiseed_*_summary.csv"
echo "============================================================"
