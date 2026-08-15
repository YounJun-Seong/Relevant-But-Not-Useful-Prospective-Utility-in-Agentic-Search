#!/bin/bash
# Gate 0 end-to-end driver: waits for prerequisites, then runs everything.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[driver] waiting for corpus build..."
until [ -f data/corpus_titles.json ]; do sleep 20; done
echo "[driver] corpus ready: $(wc -c < data/corpus.jsonl) bytes"

echo "[driver] waiting for vllm >= 0.10..."
until $PY -c "import vllm; assert vllm.__version__ >= '0.10'" 2>/dev/null; do sleep 20; done
$PY -c "import vllm, torch, datasets; print('[driver] vllm', vllm.__version__, 'torch', torch.__version__, 'datasets', datasets.__version__)"

if [ ! -f data/bm25_index/docs_meta.json ]; then
  echo "[driver] building BM25 index..."
  $PY scripts/build_index.py data/corpus.jsonl data/bm25_index
fi
echo "[driver] index ready"

# auto-fit GPU memory utilization to what is actually free (2 GiB safety margin)
GPU=${PIU_GPU:-0}
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU)
AUTO_MEM=$(python3 -c "print(max(0.20, min(0.35, round(($FREE_MB - 2048) / $TOTAL_MB, 2))))")
PIU_GPU_MEM=${PIU_GPU_MEM:-$AUTO_MEM}
echo "[driver] GPU $GPU free ${FREE_MB}MiB -> gpu_memory_utilization $PIU_GPU_MEM"

echo "[driver] smoke test..."
CUDA_VISIBLE_DEVICES=${PIU_GPU:-0} GPU_MEM=${PIU_GPU_MEM:-0.35} $PY scripts/smoke_agent.py > logs/smoke.log 2>&1 || true
grep -E "answer:|replay consistency:|state points:" logs/smoke.log || true
if grep -q "SMOKE_OK" logs/smoke.log; then
  echo "[driver] SMOKE PASSED"
else
  echo "[driver] SMOKE FAILED (see logs/smoke.log)"
  tail -20 logs/smoke.log
  exit 1
fi

echo "[driver] running Gate 0 (30 questions)..."
CUDA_VISIBLE_DEVICES=${PIU_GPU:-0} $PY scripts/run_gate0.py --n-questions 30 --out results/gate0 --gpu-mem ${PIU_GPU_MEM:-0.35} 2>&1 | grep -vE "^(INFO|WARNING)" | tail -60

$PY scripts/analyze_gate0.py results/gate0
echo "[driver] GATE0_COMPLETE"
