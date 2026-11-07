#!/usr/bin/env bash
# Run every GPU experiment; each part holds the shared GPU lock separately (each < ~10 min).
# Usage (on the GPU box, from the lab dir): bash gpu/run_gpu.sh [--quick]
set -u
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=2 MAX_JOBS=2
export P1_BUILD_DIR=${P1_BUILD_DIR:-/root/autodl-tmp/p1_build} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/autodl-tmp/p1_triton}
CORES=${CORES:-38,40}
LOCK=${LOCK:-/root/autodl-tmp/gpu.lock}
Q=${1:-}
cd "$(dirname "$0")/.."
# MLSYS_LOCK_HELD means the campaign runner already holds $LOCK. Flocking it
# again from this process blocks forever, and the GPU then sits idle.
run() {
  echo "=== $* ($(date +%T))"
  if [ -n "${MLSYS_LOCK_HELD:-}" ]; then
    taskset -c "$CORES" python "$@" || echo "!!! failed: $*"
  else
    flock "$LOCK" taskset -c "$CORES" python "$@" || echo "!!! failed: $*"
  fi
}

taskset -c "$CORES" python -c "import sys; sys.path.insert(0,'gpu'); import dwconv_cuda as d; d.load_ext()"
run -m pytest -q tests/test_gpu.py
run gpu/bench_roofline.py
run gpu/bench_attention.py --part fwd $Q
run gpu/bench_attention.py --part bwd $Q
run gpu/bench_attention.py --part bh $Q
run gpu/bench_attention.py --part blocks $Q
run gpu/bench_dwconv.py
run gpu/train_tiny.py --part recall $Q
run gpu/train_tiny.py --part mqar_sweep $Q
run gpu/train_tiny.py --part mqar $Q
run gpu/train_tiny.py --part throughput $Q
run gpu/ncu_attention.py
echo "=== all done ($(date +%T))"
