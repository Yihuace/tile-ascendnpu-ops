#!/usr/bin/env bash
# Re-run cases that fell back to "events" in the batched run.
# msprof cannot reliably attach several times inside one pytest process, so
# each case gets its own process (same protocol as the single-case baseline).
set -u
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
BENCH=benchmarks/ops/bench_conv1d.py
run() {
  printf '### %s\n' "$1"
  bash "$KDIR/run_bench.sh" "$BENCH" "$1" "$2" > "$KDIR/logs/$2.log" 2>&1
  echo "  rc=$? $(tail -1 "$KDIR/logs/$2.log")"
}
run "smoke-k3-s1-bfloat16"                     smoke-k3-s1-bf16
run "wav2vec2-layer1-float16 and not bias"     wav2vec2-fp16
run "wav2vec2-layer1-bfloat16 and not bias"    wav2vec2-bf16
run "encodec-init-bfloat16 and not bias"       encodec-init-bf16
