#!/bin/bash
# R16 lockbox: freeze artifacts FIRST, then generate untouched data with the
# frozen pipeline, then open exactly once.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${PIU_GPU:-0}
ARGS="--out results/lockbox --split validation --pre-offset 800 --n-pre 900 --n-rep 175 --n-fail 175 --reliability-by-state --reliability-frac 0.2 --skip-diag"

mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  python3 -c "print(max(0.20, min(0.85, round(($FREE_MB - 2048) / 81559, 2))))"
}

echo "[lb] freezing models"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/freeze_models.py > logs/freeze.log 2>&1
grep -a "FREEZE_DONE" logs/freeze.log

MEM=$(mem_fit)
echo "[lb] phase base 900 dev qs offset 800 (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem $MEM > logs/lb_base.log 2>&1
grep -a "BASE_DONE" logs/lb_base.log | tail -1
echo "[lb] phase select"
$PY scripts/run_gate1.py $ARGS --phase select
echo "[lb] phase plan"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/lb_plan.log 2>&1
grep -a "PLAN_DONE" logs/lb_plan.log | tail -1
MEM=$(mem_fit)
echo "[lb] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase arms --gpu-mem $MEM > logs/lb_arms.log 2>&1
grep -a "ARMS_DONE" logs/lb_arms.log | tail -1
echo "[lb] phase utils"
$PY -u scripts/run_gate1.py $ARGS --phase utils > logs/lb_utils.log 2>&1
grep -a "UTILS_DONE" logs/lb_utils.log | tail -1
$PY scripts/build_pairs.py results/lockbox 0.1
echo "[lb] gate features"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/gate_features.py --gate-dir results/lockbox > logs/lb_gatefeat.log 2>&1
grep -a "GATE_FEATURES_DONE" logs/lb_gatefeat.log
echo "[lb] semantic features"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/semantic_features.py results/lockbox > logs/lb_semantic.log 2>&1
grep -a "SEMANTIC_DONE" logs/lb_semantic.log
echo "[lb] OPENING LOCKBOX (once)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/lockbox_eval.py > logs/lockbox_eval.log 2>&1
grep -a "LOCKBOX_OPENED_ONCE" logs/lockbox_eval.log
echo "LOCKBOX_CHAIN_DONE"
