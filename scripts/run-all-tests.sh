#!/usr/bin/env bash
# Run every test suite (calculator + all labs) on Linux; same suites as run-all-tests.ps1.
# Usage: PYTHON=python3 ./scripts/run-all-tests.sh   (suites run in parallel, one pytest each)
set -u
PY=${PYTHON:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SUITES=(
  "tools mlsys_calc|tools/tests"
  "cs149 asst1 cpu-parallel|solutions/cs149/asst1-cpu-parallel/tests"
  "cs149 asst2 task-scheduler|solutions/cs149/asst2-task-scheduler/tests"
  "cs149 asst3 scan-render|solutions/cs149/asst3-scan-render/tests"
  "cs149 asst4 dnn-accelerator|solutions/cs149/asst4-dnn-accelerator/tests"
  "cs149 asst5 big-graphs|solutions/cs149/asst5-big-graphs/tests"
  "cs229s project1 kernels|solutions/cs229s/project1-transformer-kernels/tests"
  "cs229s project2 kv-spec|solutions/cs229s/project2-kv-speculative/tests"
  "cmu15849 distributed-training|solutions/cmu15849/distributed-training/tests"
  "cmu15849 serving-memory|solutions/cmu15849/serving-memory/tests"
  "cmu15849 compiler-compression|solutions/cmu15849/compiler-compression/tests"
)
# Suites run concurrently up to the number of usable cores, except the task-scheduler suite:
# its spinning workers need real cores and its deadlock timeout fires when starved, so it runs alone.
TMP=$(mktemp -d)
JOBS=$(nproc)
run_suite() {
  local i=$1 dir=$2
  ( cd "$(dirname "$dir")" && OMP_NUM_THREADS=1 "$PY" -m pytest -q -p no:cacheprovider "$(basename "$dir")" > "$TMP/$i.out" 2>&1; echo $? > "$TMP/$i.rc" )
}
i=0
for s in "${SUITES[@]}"; do
  dir=$ROOT/${s#*|}
  if [[ "$s" != *asst2* ]]; then
    run_suite $i "$dir" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 0.2; done
  fi
  i=$((i+1))
done
wait
i=0
for s in "${SUITES[@]}"; do
  [[ "$s" == *asst2* ]] && run_suite $i "$ROOT/${s#*|}"
  i=$((i+1))
done
fail=0
i=0
for s in "${SUITES[@]}"; do
  name=${s%%|*}
  rc=$(cat "$TMP/$i.rc"); summary=$(grep -E "passed|failed|error" "$TMP/$i.out" | tail -1)
  if [ "$rc" = 0 ]; then res=PASS; else res=FAIL; fail=1; tail -30 "$TMP/$i.out"; fi
  printf "%-32s %-5s %s\n" "$name" "$res" "$summary"
  i=$((i+1))
done
rm -rf "$TMP"
exit $fail
