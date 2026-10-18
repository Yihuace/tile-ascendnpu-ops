#!/usr/bin/env bash
# Batch msprof-op measurement: baseline vs candidate, interleaved (A/B/A/B).
# Outputs one "RESULT <wl> <dtype> <bias> <baseline_us> <candidate_us>" line each.
set -u
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
BASE=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/_conv1d_unit_stride_kernel.py
CAND="$KDIR/_conv1d_unit_stride_kernel.py"
source "$KDIR/env.sh"

# workload dtype bias
measure() {
  local mod="$1" wl="$2" dt="$3" bias="$4"
  local out
  out=$(python3 "$KDIR/measure.py" "$mod" "$wl" "$dt" "$bias" 2>/dev/null | grep MEASURE | head -1)
  echo "$out" | sed -n 's/.*duration_us=\([0-9.]*\).*/\1/p'
}

run_pair() {
  local wl="$1" dt="$2" bias="$3"
  local b c
  b=$(measure "$BASE" "$wl" "$dt" "$bias")
  c=$(measure "$CAND" "$wl" "$dt" "$bias")
  echo "RESULT $wl $dt bias=$bias base=${b:-NA} cand=${c:-NA}"
}

for pair in \
  "whisper-conv1 float16 0" \
  "whisper-conv1 bfloat16 0" \
  "whisper-conv1-bias float16 1" \
  "whisper-conv1-bias bfloat16 1" \
  "encodec-deep float16 0" \
  "encodec-deep bfloat16 0" \
  "encodec-deep-bias bfloat16 1" \
  "encodec-deep-bias float16 1" \
; do
  read -r wl dt bias <<< "$pair"
  run_pair "$wl" "$dt" "$bias"
done
