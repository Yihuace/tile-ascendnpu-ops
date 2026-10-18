#!/usr/bin/env bash
# GeluTanhAndMulFwdOp baseline: run tune workloads serially, one msprof process each.
set -u
REPO=/mnt/workspace/npuir-migration/examples/TileOPs
KDIR=$REPO/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/perf_opt
KERNEL=${1:-$REPO/tileops/kernels/elementwise/gelu_tanh_and_mul/gelu_tanh_and_mul_kernel/gelu_tanh_and_mul_fwd_kernel.py}
TAG=${2:-base}
KNAME=${3:-main}
mkdir -p "$KDIR/logs" "$KDIR/profiles"
: > "$KDIR/logs/${TAG}.tsv"

run_case() {
  local label=$1 dt=$2 tag="$1-$2"
  local out="$KDIR/profiles/${TAG}_$tag"
  rm -rf "$out"; mkdir -p "$out"; chmod 700 "$out"
  echo "### $tag" >&2
  msprof op --kernel-name="$KNAME" --launch-count=20 --warm-up=5 --aic-metrics=Default,Roofline \
      --output="$out" python "$KDIR/bench_kernel.py" --module "$KERNEL" \
      --workload "$label" --dtype "$dt" --launches 25 \
      > "$KDIR/logs/${TAG}_$tag.log" 2>&1
  local rc=$?
  local csv; csv=$(find "$out" -name OpBasicInfo.csv | head -1)
  local dur="NA"
  [ -n "$csv" ] && dur=$(awk -F, 'NR==2{print $3}' "$csv")
  printf '%s\t%s\t%s\t%s\n' "$tag" "$dt" "$dur" "$rc" | tee -a "$KDIR/logs/${TAG}.tsv"
}

run_case prefill float16
run_case prefill bfloat16
run_case decode bfloat16
run_case swiglu-b32 float16
run_case swiglu-b32 bfloat16
