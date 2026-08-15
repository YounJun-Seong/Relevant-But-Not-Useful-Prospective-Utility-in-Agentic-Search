#!/bin/bash
# Gate 1 driver: waits for the E5 embedding, then runs phases as separate
# processes (vLLM and E5/CE never share GPU memory simultaneously).
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${PIU_GPU:-0}

echo "[g1] waiting for corpus embedding..."
until [ -f data/e5_embeddings.fp16.npy ] && ! pgrep -f embed_corpus.py > /dev/null; do sleep 60; done
$PY -c "
import numpy as np
e = np.load('data/e5_embeddings.fp16.npy', mmap_mode='r')
import json
n = len(json.load(open('data/corpus_titles.json')))
assert e.shape[0] == n, f'embedding rows {e.shape[0]} != corpus {n}'
tail = np.asarray(e[-1], dtype=np.float32)
assert abs(np.linalg.norm(tail) - 1.0) < 0.05, 'last row not normalized - incomplete?'
print('[g1] embeddings OK', e.shape)"

mem_fit () {
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)
  python3 -c "print(max(0.20, min(0.60, round(($FREE_MB - 2048) / $TOTAL_MB, 2))))"
}

MEM=$(mem_fit)
echo "[g1] phase base (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --phase base --gpu-mem $MEM 2>&1 | grep -vE "^(INFO|WARNING)" | tail -3

echo "[g1] phase select"
$PY scripts/run_gate1.py --phase select

echo "[g1] phase plan (E5+CE)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --phase plan 2>&1 | grep -vE "^(INFO|WARNING)" | tail -4

MEM=$(mem_fit)
echo "[g1] phase arms (gpu-mem $MEM)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_gate1.py --phase arms --gpu-mem $MEM 2>&1 | grep -vE "^(INFO|WARNING)" | tail -3

echo "[g1] phase utils"
$PY scripts/run_gate1.py --phase utils

$PY scripts/analyze_gate1.py results/gate1
echo "[g1] GATE1_COMPLETE"
