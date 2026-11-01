#!/usr/bin/env bash
# Build the three CUDA programs into gpu/build/ (RTX 5090: sm_120, CUDA 12.8).
# render uses -fmad=false so device arithmetic matches the host reference bit for bit.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-sm_120}
mkdir -p build
$NVCC -O3 -arch=$ARCH -std=c++17 -o build/saxpy saxpy.cu &
$NVCC -O3 -arch=$ARCH -std=c++17 -o build/scan scan.cu &
$NVCC -O3 -arch=$ARCH -std=c++17 -fmad=false -lineinfo -o build/render render.cu &
wait
ls -l build
