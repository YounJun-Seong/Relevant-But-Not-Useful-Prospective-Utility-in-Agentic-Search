#!/bin/bash
# Gate 3 (R11): pairwise label generation at scale on the TRAIN split.
# 1,200 questions (600 rep + 600 fail-enriched from a 3,000-question pre-pool),
# reliability by state 20%, diagnostics' nDCG skipped (labels only).
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${PIU_GPU:-0}
ARGS="--out results/gate3 --split train --n-pre 3000 --n-rep 600 --n-fail 600 --reliability-by-state --reliability-frac 0.2 --skip-diag"

mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)
  python3 -c "print(max(0.20, min(0.85, round(($FREE_MB - 2048) / $TOTAL_MB, 2))))"
}

MEM=$(mem_fit)
echo "[g3] phase base 3000 train qs (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem $MEM > logs/g3_base.log 2>&1
grep -a "BASE_DONE" logs/g3_base.log | tail -1
echo "[g3] phase select"
$PY scripts/run_gate1.py $ARGS --phase select
echo "[g3] phase plan"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/g3_plan.log 2>&1
grep -a "PLAN_DONE" logs/g3_plan.log | tail -1
MEM=$(mem_fit)
echo "[g3] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase arms --gpu-mem $MEM > logs/g3_arms.log 2>&1
grep -a "ARMS_DONE" logs/g3_arms.log | tail -1
echo "[g3] phase utils (skip-diag)"
$PY -u scripts/run_gate1.py $ARGS --phase utils > logs/g3_utils.log 2>&1
grep -a "UTILS_DONE" logs/g3_utils.log | tail -1
$PY scripts/build_pairs.py results/gate3 0.1
echo "GATE3_COMPLETE"
