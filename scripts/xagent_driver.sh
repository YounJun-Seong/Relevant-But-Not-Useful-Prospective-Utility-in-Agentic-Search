#!/bin/bash
# R18: cross-agent replication (Mistral-7B-Instruct-v0.3), frozen protocol.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
ARGS="--out results/xagent --model $MODEL --pre-tag mistral --split validation --pre-offset 800 --n-pre 900 --n-rep 175 --n-fail 175 --reliability-by-state --reliability-frac 0.2 --skip-diag"
GPU=1

echo "[xa] smoke: Mistral protocol adherence"
CUDA_VISIBLE_DEVICES=$GPU MODEL=$MODEL GPU_MEM=0.5 $PY scripts/smoke_agent.py > logs/xa_smoke.log 2>&1 || true
if grep -q "SMOKE_OK" logs/xa_smoke.log; then echo "[xa] SMOKE PASSED"; else echo "[xa] SMOKE FAILED"; tail -12 logs/xa_smoke.log; exit 1; fi

echo "[xa] phase base 900 (agent B)"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase base --gpu-mem 0.5 > logs/xa_base.log 2>&1
grep -a "BASE_DONE" logs/xa_base.log | tail -1
echo "[xa] phase select"
$PY scripts/run_gate1.py $ARGS --phase select
echo "[xa] phase plan"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase plan > logs/xa_plan.log 2>&1
grep -a "PLAN_DONE" logs/xa_plan.log | tail -1
echo "[xa] phase arms"
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_gate1.py $ARGS --phase arms --gpu-mem 0.5 > logs/xa_arms.log 2>&1
grep -a "ARMS_DONE" logs/xa_arms.log | tail -1
echo "[xa] phase utils"
$PY -u scripts/run_gate1.py $ARGS --phase utils > logs/xa_utils.log 2>&1
grep -a "UTILS_DONE" logs/xa_utils.log | tail -1
$PY scripts/build_pairs.py results/xagent 0.1
echo "[xa] EVAL (confirmatory, once)"
$PY -u scripts/xagent_eval.py > logs/xagent_eval.log 2>&1
grep -a "XAGENT_EVAL_DONE" logs/xagent_eval.log
echo "XAGENT_CHAIN_DONE"
