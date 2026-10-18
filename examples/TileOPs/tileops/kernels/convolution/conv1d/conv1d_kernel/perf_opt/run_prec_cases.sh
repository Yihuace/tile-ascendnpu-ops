#!/usr/bin/env bash
# Per-case precision sweep with retries: one NPU process per case so a flaky
# bishengir SIGSEGV (CG-2026-0003 family) cannot abort the whole ladder.
# Usage: run_prec_cases.sh <module.py> <tag> [cases...]
set -u
MOD="${1:?module}"; TAG="${2:?tag}"; shift 2
KDIR=/mnt/workspace/npuir-migration/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt
case "$MOD" in /*) ;; *) MOD="$KDIR/$MOD" ;; esac
LOG="$KDIR/logs/$TAG"
mkdir -p "$LOG"
cd "$KDIR" || exit 1
# shellcheck disable=SC1091
source "$KDIR/env.sh"
CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
  CASES=(encodec-init wav2vec2 encodec-init-bias wav2vec2-bias smoke smoke-bias nondiv batch3-bias dilation-bias stride3)
fi
fail=0
for c in "${CASES[@]}"; do
  for attempt in 1 2 3; do
    python3 "$KDIR/prec_case.py" "$MOD" "$c" >"$LOG/prec-$c.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "CASE $c PASS (attempt $attempt)"
      break
    fi
    if grep -q "SIGSEGV\|NPU IR opt failed" "$LOG/prec-$c.log"; then
      echo "CASE $c RETRY (compiler SIGSEGV, attempt $attempt)"
      continue
    fi
    echo "CASE $c FAIL (non-compiler error, attempt $attempt)"
    fail=1
    break
  done
  if [ $rc -ne 0 ]; then
    echo "CASE $c UNRESOLVED"
    fail=1
  fi
done
echo "PREC_CASES_DONE tag=$TAG fail=$fail"
exit $fail
