#!/usr/bin/env bash
# Run the candidate module's own precision ladder (L0 -> L1 -> L2 -> boundary).
# Usage: run_lall.sh <module.py> <logdir-tag>
set -u
MOD="${1:?module}"; TAG="${2:?tag}"
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
case "$MOD" in /*) ;; *) MOD="$KDIR/$MOD" ;; esac
LOG="$KDIR/logs/$TAG"
mkdir -p "$LOG"
cd "$KDIR" || exit 1
# shellcheck disable=SC1091
source "$KDIR/env.sh"
python3 "$MOD" --level all >"$LOG/level_all.log" 2>&1
rc=$?
grep -E "^\[L0\]|^\[L1\]|^\[L2\]|^\[Boundary\]|FAIL" "$LOG/level_all.log" | tail -20
echo "RUN_LALL $TAG rc=$rc"
