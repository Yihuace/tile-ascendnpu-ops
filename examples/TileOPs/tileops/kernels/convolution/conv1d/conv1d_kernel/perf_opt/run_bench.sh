#!/usr/bin/env bash
# Stage-4 baseline runner for Conv1dFwdOp (one benchmark case at a time).
#
# Always invoked through .agents/tools/npu_lock.sh so a single case owns the
# device.  Must run under `bash -ic` semantics (CANN env from .bashrc).
#
# usage: run_bench.sh <bench_file> <pytest -k expr> <tag>
set -u

BENCH="${1:?bench file}"
KEXPR="${2:?pytest -k expression}"
TAG="${3:?tag}"

REPO=/mnt/workspace/npuir-migration/examples/TileOPs
KDIR="$REPO/tileops/kernels/convolution/conv1d/conv1d_kernel/perf_opt"
mkdir -p "$KDIR/profiles/$TAG" "$KDIR/logs"

cd "$REPO" || exit 1
export TILEOPS_PROF_MODE=msprof
export TILEOPS_MSPROF_KEEP_OUTPUT=1
export TILEOPS_MSPROF_ARTIFACT_ROOT="$KDIR/profiles/$TAG"
export TILEOPS_BENCHMARK_REPORT_PATH="$KDIR/logs/bench_$TAG.md"

exec python3 -m pytest "$BENCH" -m 'not smoke' -k "$KEXPR" -q -p no:cacheprovider
