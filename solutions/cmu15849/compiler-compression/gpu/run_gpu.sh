#!/usr/bin/env bash
# Run the GPU experiments; each step holds the shared GPU lock for one script (< 10 min).
# Usage: bash gpu/run_gpu.sh [step ...]   (default: all steps)
# Env: CORES (taskset list, default 47,48), HF_HOME (models/ and data/wikitext2/ inside).
set -u
cd "$(dirname "$0")"
CORES=${CORES:-47,48}
PY=${PY:-/root/miniconda3/bin/python}
LOCK=${LOCK:-/root/autodl-tmp/gpu.lock}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf} HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# cold compile caches so compile_s is the real cost
export TORCHINDUCTOR_CACHE_DIR=/tmp/c1_inductor_$$ TRITON_CACHE_DIR=/tmp/c1_triton_$$
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

run() { echo "=== $* $(date +%T)"; flock "$LOCK" taskset -c "$CORES" "$PY" "$@" || echo "FAILED: $*"; }

steps=${*:-"chains mlp triton dynamic matmul qspeed quant_gpt2 quant_gpt2m prune_ppl prune_speed lora"}
for s in $steps; do
  case $s in
    chains)      run fusion_bench.py --part chains ;;
    mlp)         run fusion_bench.py --part mlp ;;
    triton)      run fusion_bench.py --part triton ;;
    dynamic)     run fusion_bench.py --part dynamic ;;
    matmul)      run matmul_search.py ;;
    qspeed)      run quant_speed.py ;;
    quant_gpt2)  run quant_llm.py --model gpt2 ;;
    quant_gpt2m) run quant_llm.py --model gpt2-medium ;;
    prune_ppl)   run prune_bench.py --part ppl --model gpt2 ;;
    prune_speed) run prune_bench.py --part speed ;;
    lora)        run lora_ft.py ;;
    test)        flock "$LOCK" taskset -c "$CORES" "$PY" -m pytest -q ../tests ;;
    *) echo "unknown step $s" ;;
  esac
done
echo "=== done $(date +%T)"
