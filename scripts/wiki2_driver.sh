#!/bin/bash
# R17: 2Wiki cross-dataset replication with the frozen protocol.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ARGS="--out results/wiki2 --dataset 2wiki --split validation --n-pre 900 --n-rep 175 --n-fail 175 --reliability-by-state --reliability-frac 0.2 --skip-diag"
mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  python3 -c "print(max(0.20, min(0.70, round(($FREE_MB - 4096) / 81559, 2))))"
}
MEM=$(mem_fit)
echo "[w2] phase base 900 2wiki qs (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem $MEM > logs/w2_base.log 2>&1
grep -a "BASE_DONE" logs/w2_base.log | tail -1
echo "[w2] phase select"
$PY scripts/run_gate1.py $ARGS --phase select
echo "[w2] phase plan"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/w2_plan.log 2>&1
grep -a "PLAN_DONE" logs/w2_plan.log | tail -1
MEM=$(mem_fit)
echo "[w2] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_gate1.py $ARGS --phase arms --gpu-mem $MEM > logs/w2_arms.log 2>&1
grep -a "ARMS_DONE" logs/w2_arms.log | tail -1
echo "[w2] phase utils"
$PY -u scripts/run_gate1.py $ARGS --phase utils > logs/w2_utils.log 2>&1
grep -a "UTILS_DONE" logs/w2_utils.log | tail -1
$PY scripts/build_pairs.py results/wiki2 0.1
echo "[w2] EVAL (confirmatory, once)"
$PY -u scripts/wiki2_eval.py > logs/wiki2_eval.log 2>&1
grep -a "WIKI2_EVAL_DONE" logs/wiki2_eval.log
echo "WIKI2_CHAIN_DONE"
