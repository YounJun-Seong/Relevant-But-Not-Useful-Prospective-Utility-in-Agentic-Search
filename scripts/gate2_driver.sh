#!/bin/bash
# Gate 2 official run (R10 pre-registered): 300 questions (150 rep + 150 fail),
# reliability by state (3x replay for all candidates of 20% of states).
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${PIU_GPU:-0}
ARGS="--out results/gate2 --n-pre 800 --n-rep 150 --n-fail 150 --reliability-by-state --reliability-frac 0.2"

mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)
  python3 -c "print(max(0.20, min(0.85, round(($FREE_MB - 2048) / $TOTAL_MB, 2))))"
}

MEM=$(mem_fit)
echo "[g2] phase base 800 qs (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem $MEM > logs/g2_base.log 2>&1
grep -a "BASE_DONE" logs/g2_base.log | tail -1
echo "[g2] phase select"
$PY scripts/run_gate1.py $ARGS --phase select
echo "[g2] phase plan"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/g2_plan.log 2>&1
grep -a "PLAN_DONE" logs/g2_plan.log | tail -1
MEM=$(mem_fit)
echo "[g2] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase arms --gpu-mem $MEM > logs/g2_arms.log 2>&1
grep -a "ARMS_DONE" logs/g2_arms.log | tail -1
echo "[g2] phase utils"
$PY -u scripts/run_gate1.py $ARGS --phase utils > logs/g2_utils.log 2>&1
grep -a "UTILS_DONE" logs/g2_utils.log | tail -1
$PY scripts/build_pairs.py results/gate2 0.1
$PY scripts/analyze_gate2.py results/gate2
echo "GATE2_COMPLETE"
