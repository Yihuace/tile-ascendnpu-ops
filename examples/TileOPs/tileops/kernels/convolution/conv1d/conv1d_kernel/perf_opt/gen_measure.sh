#!/usr/bin/env bash
# Measure one (module, workload) pair with msprof op and store the raw profile.
# Usage: gen_measure.sh <module.py> <workload> <dtype> <outdir-tag>
# Prints:  MEASURE <workload> <dtype> <duration_us> <samples>  (or FAILED)
set -u
MOD="${1:?module}"; WL="${2:?workload}"; DT="${3:?dtype}"; TAG="${4:?tag}"
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
cd "$KDIR" || exit 1
# shellcheck disable=SC1091
source "$KDIR/env.sh"
PROF="$KDIR/profiles/$TAG"
mkdir -p "$PROF"
BIAS=0
case "$WL" in *-bias) BIAS=1 ;; esac
LOG="$KDIR/logs/$TAG"
mkdir -p "$LOG"
python3 "$KDIR/measure.py" "$MOD" "$WL" "$DT" "$BIAS" --out "$PROF" \
  >"$LOG/measure.out" 2>"$LOG/measure.err"
rc=$?
if [ $rc -ne 0 ]; then
  echo "MEASURE $WL $DT FAILED rc=$rc"
else
  grep -h '^MEASURE' "$LOG/measure.out" || echo "MEASURE $WL $DT NO_OUTPUT"
fi
