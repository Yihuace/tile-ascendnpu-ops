#!/usr/bin/env bash
# R1-A measurement: 7 unit_stride cases, one process each (msprof reliability).
set -u
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
BENCH=benchmarks/ops/bench_conv1d.py
run() {
  printf '### %s\n' "$2"
  bash "$KDIR/run_bench.sh" "$BENCH" "$1" "$2" > "$KDIR/logs/$2.log" 2>&1
  echo "  rc=$? $(tail -1 "$KDIR/logs/$2.log")"
}
run "whisper-conv1-float16 and not bias"        r1a-whisper-fp16
run "whisper-conv1-bfloat16 and not bias"       r1a-whisper-bf16
run "whisper-conv1-bias-float16"                r1a-whisper-bias-fp16
run "whisper-conv1-bias-bfloat16"               r1a-whisper-bias-bf16
run "encodec-deep-float16 and not bias"         r1a-encodec-deep-fp16
run "encodec-deep-bfloat16 and not bias"        r1a-encodec-deep-bf16
run "encodec-deep-bias-bfloat16"                r1a-encodec-deep-bias-bf16
