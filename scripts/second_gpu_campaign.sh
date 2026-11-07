#!/usr/bin/env bash
# Re-run the GPU labs on this machine's card. The RTX 5090 raw files stay
# where they are: each lab's artifacts/raw/gpu is snapshotted, the lab writes
# a fresh tree, that tree is copied to $OUT/<lab>, then the snapshot is put
# back. One lab at a time. A failed step is logged and the campaign continues.
# New runs should use `python -m mlsys_harness.campaign run campaigns/next_gpu.json`
# instead of growing this script. That runner holds the GPU lock once and checks
# the result file, so a process that exits 0 with a bad measurement is a failure.
set -u
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/mlsys-infra
LOCK=/root/autodl-tmp/gpu.lock
touch "$LOCK"

eval "$($PY - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
slug = "".join(ch if ch.isalnum() else "_" for ch in p.name.lower()).strip("_")
print(f"export GPU_NAME={p.name!r}")
print(f"export GPU_SLUG={slug}")
print(f"export GPU_CC={p.major}.{p.minor}")
print(f"export GPU_SMS={p.multi_processor_count}")
print(f"export TORCH_CUDA_ARCH_LIST={p.major}.{p.minor}")
print("export CUDA_ARCH=native")
print("export ARCH=native")
PY
)"
OUT=/root/autodl-tmp/results_${GPU_SLUG}
mkdir -p "$OUT" /root/autodl-tmp/logs
echo "device $GPU_NAME cc=$GPU_CC sms=$GPU_SMS arch=$TORCH_CUDA_ARCH_LIST out=$OUT"

CORES=$($PY - <<'PY'
import time
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3:4].isdigit():
            parts = line.split()
            nums = [int(x) for x in parts[1:]]
            out[int(parts[0][3:])] = (nums[3] + nums[4], sum(nums))
    return out
a = snap(); time.sleep(0.4); b = snap()
rank = []
for cpu, (i1, t1) in a.items():
    i2, t2 = b.get(cpu, (i1, t1))
    dt = t2 - t1
    rank.append(((i2 - i1) / dt if dt else 0.0, cpu))
rank.sort(reverse=True)
def siblings(cpu):
    try:
        text = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read().strip()
    except OSError:
        return {cpu}
    out = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out
chosen, used = [], set()
for frac, cpu in rank:
    sib = siblings(cpu)
    if frac < 0.5 or used & sib:
        continue
    chosen.append(cpu)
    used.update(sib)
    if len(chosen) == 4:
        break
print(",".join(str(c) for c in (chosen or [0, 1, 2, 3])))
PY
)
echo "cores $CORES"

( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader \
      | awk -v d="$(date +%H:%M:%S)" '{print d","$0}'
    sleep 30
  done ) > "$OUT/smi.csv" &
SMI=$!

REL="" RAW="" SNAP="" DEST=""
finish_lab() {
  [ -n "$REL" ] || return 0
  mkdir -p "$DEST"
  cp -a "$RAW/." "$DEST/" 2>/dev/null || true
  if [ -n "$(ls -A "$SNAP" 2>/dev/null)" ]; then
    find "$RAW" -mindepth 1 -delete
    cp -a "$SNAP/." "$RAW/"
  fi
  echo "----- saved $REL -> $DEST"
  REL=""
}
trap 'finish_lab; kill $SMI 2>/dev/null || true' EXIT

begin() {
  finish_lab
  REL="$1"
  local lab="$ROOT/$REL"
  RAW="$lab/artifacts/raw/gpu"
  local name=${REL//\//_}
  SNAP="$OUT/snap/$name"
  DEST="$OUT/$name"
  mkdir -p "$RAW" "$SNAP" "$DEST"
  if [ -z "$(ls -A "$SNAP" 2>/dev/null)" ]; then
    cp -a "$RAW/." "$SNAP/" 2>/dev/null || true
  fi
  if [ -n "$(ls -A "$SNAP" 2>/dev/null)" ]; then
    find "$RAW" -mindepth 1 -delete
  fi
  echo "===== $REL  $(date +%H:%M:%S)"
}
# Do not flock here. A3 and the compiler runner flock the same file internally,
# and a second flock from this shell blocks forever (the GPU then sits at 0%).
step() {
  echo "--- $*  $(date +%H:%M:%S)"
  ( cd "$ROOT/$REL" && taskset -c "$CORES" "$@" ) \
    && echo "OK $*" || echo "FAILED $*"
}

FROM=${1:-all}
echo "from $FROM"

if [ "$FROM" = all ]; then
begin solutions/cs229s/project1-transformer-kernels
step "$PY" -c "import sys; sys.path.insert(0,'gpu'); import dwconv_cuda as d; d.load_ext(); print('dwconv ok')"
step "$PY" gpu/bench_roofline.py
step "$PY" gpu/bench_attention.py --part fwd
step "$PY" gpu/bench_attention.py --part bwd
step "$PY" gpu/bench_attention.py --part bh
step "$PY" gpu/bench_attention.py --part blocks
step "$PY" gpu/bench_dwconv.py
step "$PY" gpu/train_tiny.py --part recall
step "$PY" gpu/train_tiny.py --part throughput

begin solutions/cs149/asst4-dnn-accelerator
step "$PY" gpu/run_gpu.py build
step "$PY" gpu/run_gpu.py all
fi

begin solutions/cs149/asst3-scan-render
step bash gpu/build.sh
step bash gpu/run_gpu.sh saxpy
step bash gpu/run_gpu.sh scan
step bash gpu/run_gpu.sh render
step timeout 90 bash gpu/run_gpu.sh ncu

begin solutions/cs149/asst4-dnn-accelerator
step "$PY" gpu/fp8_triton.py

begin solutions/cs149/asst1-cpu-parallel
step bash gpu/run_gpu.sh

begin solutions/cmu15849/compiler-compression
step bash gpu/run_gpu.sh matmul qspeed chains mlp triton dynamic quant_gpt2 prune_speed

begin solutions/cs229s/project2-kv-speculative
step "$PY" gpu/run_gpu.py hw
step "$PY" gpu/run_gpu.py latency
step "$PY" gpu/run_gpu.py batch
step "$PY" gpu/run_gpu.py memory
step "$PY" gpu/run_gpu.py spec

if [ -d "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B" ]; then
  begin solutions/cmu15849/serving-memory
  step "$PY" gpu/serve_bench.py verify
  step "$PY" gpu/serve_bench.py roofline
  step "$PY" gpu/serve_bench.py paged
  step "$PY" gpu/serve_bench.py waste
  step "$PY" gpu/serve_bench.py engine --policy continuous --rates 10,20,40
  step "$PY" gpu/serve_bench.py engine --policy chunked --rates 10,20,40
else
  echo "skip serving: Qwen2.5-0.5B not in HF_HOME"
fi

finish_lab
echo "===== campaign done $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
trap - EXIT
kill $SMI 2>/dev/null || true
