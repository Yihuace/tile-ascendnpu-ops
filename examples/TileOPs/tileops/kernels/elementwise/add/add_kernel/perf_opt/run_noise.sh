#!/usr/bin/env bash
# AddFwdOp: repeat linear-bias fp16/bf16 to establish the noise floor (§13.4 style).
set -u
REPO=/mnt/workspace/npuir-migration/examples/TileOPs
KDIR=$REPO/tileops/kernels/elementwise/add/add_kernel/perf_opt
KERNEL=$REPO/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py
: > "$KDIR/logs/noise.tsv"
for rep in 1 2 3; do
  for dt in float16 bfloat16; do
    tag="noise${rep}-linear-bias-$dt"
    out="$KDIR/profiles/$tag"; rm -rf "$out"; mkdir -p "$out"; chmod 700 "$out"
    msprof op --kernel-name=main --launch-count=20 --warm-up=5 --aic-metrics=Default,Roofline \
        --output="$out" python "$KDIR/bench_kernel.py" --module "$KERNEL" \
        --workload linear-bias --dtype "$dt" --launches 25 \
        > "$KDIR/logs/$tag.log" 2>&1
    dur=$(grep -oP 'Task Duration\(us\):\s*\K[0-9.]+' "$KDIR/logs/$tag.log" | tail -1)
    printf '%s\t%s\t%s\n' "$tag" "$dt" "${dur:-NA}" | tee -a "$KDIR/logs/noise.tsv"
  done
done
