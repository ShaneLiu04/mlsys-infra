#!/usr/bin/env bash
# Runner: executes all GPU experiments and writes raw results to artifacts/raw/gpu/.
# Each experiment holds the shared GPU lock separately (if /root/autodl-tmp/gpu.lock exists).
#   bash gpu/run_gpu.sh [saxpy|scan|render|ncu|all]
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
B="$LAB/gpu/build"
RAW="$LAB/artifacts/raw/gpu"
mkdir -p "$RAW/images"
LOCK=/root/autodl-tmp/gpu.lock
# MLSYS_LOCK_HELD means the campaign runner already holds $LOCK. A second flock blocks.
run() {
  if [ -n "${MLSYS_LOCK_HELD:-}" ] || [ ! -e "$LOCK" ]; then "$@"; else flock "$LOCK" "$@"; fi
}
what=${1:-all}
if [[ $what == saxpy || $what == all ]]; then run "$B/saxpy" "$RAW/saxpy.jsonl"; fi
if [[ $what == scan || $what == all ]]; then run "$B/scan" "$RAW/scan.jsonl" "$RAW/find_repeats.jsonl" 28; fi
if [[ $what == render || $what == all ]]; then
  run "$B/render" "$RAW/render.jsonl" "$RAW/images" all
  run "$B/render" "$RAW/render_sweep.jsonl" /tmp sweep
  run "$B/render" "$RAW/render_attrs.jsonl" /tmp attrs
  run "$B/scan" "$RAW/scan_attrs.jsonl" /tmp/unused.jsonl attrs
fi
if [[ $what == ncu || $what == all ]]; then
  NCU=/usr/local/cuda/bin/ncu
  M=sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers
  run $NCU --csv --metrics $M -k regex:'blelloch_block|raking_block|upsweep_k|downsweep_k|uniform_add|DeviceScanKernel' \
      "$B/scan" /tmp/a3_s.jsonl /tmp/a3_f.jsonl 24 16777216 > "$RAW/ncu_scan.csv" 2> "$RAW/ncu_scan.err" || true
  run $NCU --csv --metrics $M -k regex:'k_render|k_naive|k_allcircles|k_bin' \
      "$B/render" /tmp/a3_r.jsonl /tmp rand100k ncu > "$RAW/ncu_render.csv" 2> "$RAW/ncu_render.err" || true
fi
