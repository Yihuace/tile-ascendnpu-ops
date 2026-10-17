#!/usr/bin/env bash
# AddFwdOp Round 1 sweep: block_size x {baseline, ub_reuse} on linear-bias.
# Each measurement is an independent msprof process (workflow §3).
set -u
REPO=/mnt/workspace/npuir-migration/examples/TileOPs
KDIR=$REPO/tileops/kernels/elementwise/add/add_kernel/perf_opt
BASE=$REPO/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py
A2=$KDIR/candidates/add_fwd_kernel_ub_reuse.py
OUT="$KDIR/logs/r1_sweep.tsv"
: > "$OUT"

one() {
  local mod=$1 bs=$2 dt=$3 rep=$4 tag="$5"
  local out="$KDIR/profiles/r1_${tag}_bs${bs}_r${rep}"
  rm -rf "$out"; mkdir -p "$out"; chmod 700 "$out"
  msprof op --kernel-name=main --launch-count=20 --warm-up=5 --aic-metrics=Default,Roofline \
      --output="$out" python "$KDIR/bench_kernel.py" --module "$mod" \
      --workload linear-bias --dtype "$dt" --block-size "$bs" --launches 25 \
      > "$KDIR/logs/r1_${tag}_bs${bs}_r${rep}.log" 2>&1
  local dur; dur=$(grep -oP 'Task Duration\(us\):\s*\K[0-9.]+' "$KDIR/logs/r1_${tag}_bs${bs}_r${rep}.log" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$bs" "$dt" "$rep" "${dur:-NA}" | tee -a "$OUT"
}

for rep in 1 2 3; do
  for dt in float16 bfloat16; do
    one "$BASE" 12288 "$dt" "$rep" base
    one "$A2"    8192 "$dt" "$rep" a2
    one "$A2"   12288 "$dt" "$rep" a2
    one "$A2"   16384 "$dt" "$rep" a2
  done
done
