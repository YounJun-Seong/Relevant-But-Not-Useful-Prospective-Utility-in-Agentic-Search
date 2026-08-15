#!/bin/bash
# Diagnostic stress test (R8-6): docs/search=2, max_turns=6, base-failure
# questions only. Runs AFTER the main Gate 1 completes. Results are reported
# separately (results/gate1_stress) and must never be merged into the main
# condition.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${PIU_GPU:-0}
MAIN_LOG=$1

echo "[stress] waiting for main Gate 1 to complete..."
until grep -aq "GATE1_COMPLETE" "$MAIN_LOG" 2>/dev/null; do sleep 60; done

mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)
  python3 -c "print(max(0.20, min(0.60, round(($FREE_MB - 2048) / $TOTAL_MB, 2))))"
}

MEM=$(mem_fit)
echo "[stress] phase base (docs=2, turns=6, gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --stress --phase base --gpu-mem $MEM 2>&1 | grep -vE "^(INFO|WARNING)" | tail -3
echo "[stress] phase select"
$PY scripts/run_gate1.py --stress --phase select
echo "[stress] phase plan"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --stress --phase plan 2>&1 | grep -vE "^(INFO|WARNING)" | tail -3
MEM=$(mem_fit)
echo "[stress] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --stress --phase arms --gpu-mem $MEM 2>&1 | grep -vE "^(INFO|WARNING)" | tail -3
echo "[stress] phase utils"
$PY scripts/run_gate1.py --stress --phase utils
$PY scripts/analyze_gate1.py results/gate1_stress
echo "[stress] STRESS_COMPLETE"
