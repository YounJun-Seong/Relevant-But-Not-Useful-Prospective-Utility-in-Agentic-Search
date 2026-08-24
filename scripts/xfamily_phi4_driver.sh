#!/bin/bash
# Post-hoc cross-family replication: microsoft/phi-4, HotpotQA representative
# subset (no failure enrichment), same frozen protocol as the 14B replication
# (4 turns, 3 docs, 8 candidate families, tau=0.1). Controls are skipped
# because confirmatory endpoints are defined on raw outcomes.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL="microsoft/phi-4"
ARGS="--out results/xfamily_phi4 --model $MODEL --pre-tag phi4 --split validation --pre-offset 800 --n-pre 400 --n-rep 350 --n-fail 0 --reliability-frac 0.01 --skip-diag"
GPU=${GPU:-1}; MEM=${MEM:-0.55}
echo "[xf] base 400"; CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem $MEM > logs/xf_base.log 2>&1; grep -a "BASE_DONE" logs/xf_base.log | tail -1
echo "[xf] select"; $PY scripts/run_gate1.py $ARGS --phase select
echo "[xf] plan"; CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/xf_plan.log 2>&1; grep -a "PLAN_DONE" logs/xf_plan.log | tail -1
echo "[xf] arms (base + I2_add)"; CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase arms --arm-types base,I2_add --gpu-mem $MEM > logs/xf_arms.log 2>&1; grep -a "ARMS_DONE" logs/xf_arms.log | tail -1
echo "[xf] utils"; $PY -u scripts/run_gate1.py $ARGS --phase utils > logs/xf_utils.log 2>&1; grep -a "UTILS_DONE" logs/xf_utils.log | tail -1
echo "[xf] pairs"; $PY scripts/build_pairs.py results/xfamily_phi4 0.1 > logs/xf_pairs.log 2>&1
echo "[xf] CE exact + BGE on D_t"; CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/ce_scores_candidates.py > logs/xf_ce.log 2>&1; CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/strong_reranker.py --tags xfamily_phi4 > logs/xf_bge.log 2>&1
echo "XFAMILY_CHAIN_DONE"
