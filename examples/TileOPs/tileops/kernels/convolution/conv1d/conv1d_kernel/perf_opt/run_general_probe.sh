#!/usr/bin/env bash
# Baseline + candidate measurement sweep for the general conv1d kernel.
# Usage: run_general_probe.sh <module.py> <tag> [workloads...]
set -u
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
MOD="${1:?module}"; TAG="${2:?tag}"; shift 2
LIST=("$@")
if [ ${#LIST[@]} -eq 0 ]; then
  LIST=(encodec-init encodec-init-bias wav2vec2-layer1 wav2vec2-layer1-bias)
fi
for wl in "${LIST[@]}"; do
  for dt in float16 bfloat16; do
    "$KDIR/gen_measure.sh" "$MOD" "$wl" "$dt" "$TAG/$wl-$dt"
  done
done
