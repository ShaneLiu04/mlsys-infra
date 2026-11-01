#!/usr/bin/env bash
# Build and run the GPU counterpart of asst1; results -> artifacts/raw/gpu/*.jsonl.
# Usage: bash gpu/run_gpu.sh   (from the lab root; needs nvcc and an NVIDIA GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-native}
OUT=artifacts/raw/gpu
mkdir -p "$OUT" gpu/bin
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -fmad=false -o gpu/bin/a1_gpu gpu/a1_gpu.cu
B=gpu/bin/a1_gpu
$B info > "$OUT/info.jsonl"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv >> "$OUT/nvidia_smi.csv"
: > "$OUT/mandel.jsonl"
for v in 1 2; do for s in 1 4; do $B mandel --view $v --scale $s >> "$OUT/mandel.jsonl"; done; done
$B sqrt > "$OUT/sqrt.jsonl"
$B saxpy > "$OUT/saxpy.jsonl"
$B kmeans > "$OUT/kmeans.jsonl"
echo "gpu done"; wc -l "$OUT"/*.jsonl
