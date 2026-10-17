#!/usr/bin/env bash
# AddFwdOp baseline: 11 tune cases, one msprof process each (workflow §3).
set -u
REPO=/mnt/workspace/npuir-migration/examples/TileOPs
KDIR=$REPO/tileops/kernels/elementwise/add/add_kernel/perf_opt
KERNEL=$REPO/tileops/kernels/elementwise/add/add_kernel/add_fwd_kernel.py
mkdir -p "$KDIR/logs" "$KDIR/profiles"
: > "$KDIR/logs/baseline.tsv"

run_case() {
  local label=$1 dt=$2 tag="$1-$2"
  local out="$KDIR/profiles/base_$tag"
  rm -rf "$out"; mkdir -p "$out"; chmod 700 "$out"
  echo "### $tag"
  msprof op --kernel-name=main --launch-count=20 --warm-up=5 --aic-metrics=Default,Roofline \
      --output="$out" python "$KDIR/bench_kernel.py" --module "$KERNEL" \
      --workload "$label" --dtype "$dt" --launches 25 \
      > "$KDIR/logs/base_$tag.log" 2>&1
  local rc=$?
  local csv; csv=$(find "$out" -name OpBasicInfo.csv | head -1)
  local dur="NA"
  [ -n "$csv" ] && dur=$(awk -F, 'NR==2{print $3}' "$csv")
  printf '%s\t%s\t%s\t%s\n' "$tag" "$dt" "$dur" "$rc" | tee -a "$KDIR/logs/baseline.tsv"
}

for dt in float32 bfloat16; do run_case hidden-state-prefill "$dt"; done
for dt in float32 float16 bfloat16; do run_case cnn-feat-broadcast "$dt"; done
for dt in float32 float16 bfloat16; do run_case llama-7b-ffn "$dt"; done
for dt in float32 float16 bfloat16; do run_case linear-bias "$dt"; done
