#!/usr/bin/env bash
# npu_lock.sh -- run one command while holding the workspace-wide NPU mutex.
#
# Why: concurrent pytest / msprof / report runs on the same Ascend device
# corrupt timing measurements and can OOM the device (see
# docs/cann9_adaptation.md §12.6).  Every NPU-touching command must go through
# this wrapper so runs are strictly serialized and pre-flight checks catch
# stray processes left behind by an earlier session.
#
# Usage:
#   .agents/tools/npu_lock.sh [--wait SEC] [--timeout SEC] -- <command...>
#
#   --wait     seconds to wait for the lock (default 0 = fail fast)
#   --timeout  seconds allowed for the command itself (default 0 = no limit)
#
# Exit codes:
#   75  lock held by another task (or pre-flight found a foreign process)
#   76  NPU not idle before start
#   124 command timed out
#   *   command's own exit code
set -uo pipefail

LOCK_FILE="${NPU_LOCK_FILE:-/tmp/opencode/npu_workspace.lock}"
WAIT_S=0
CMD_TIMEOUT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --wait) WAIT_S="$2"; shift 2 ;;
    --timeout) CMD_TIMEOUT="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "npu_lock: no command given" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOCK_FILE")"
touch "$LOCK_FILE"

# fd 9 holds the mutex for the lifetime of this script.
exec 9>"$LOCK_FILE"
if [ "$WAIT_S" -gt 0 ]; then
  if ! flock -w "$WAIT_S" 9; then
    echo "npu_lock: NPU mutex busy (waited ${WAIT_S}s), refusing to start" >&2
    exit 75
  fi
else
  if ! flock -n 9; then
    echo "npu_lock: NPU mutex busy; another task owns the device" >&2
    exit 75
  fi
fi

# --- pre-flight: no other NPU workload may be running -----------------------
busy=""
if command -v npu-smi >/dev/null 2>&1; then
  if ! npu-smi info 2>/dev/null | grep -q "No running processes found"; then
    busy="$(npu-smi info 2>/dev/null | sed -n '/Process id/,/^+/p' | head -20)"
  fi
fi

# Stray host processes that would compete for the device.  Match the real
# workloads, not this script (the bracket trick avoids self-matching).
stray="$(pgrep -af 'msprof|[p]ytest|[t]ileops\.reporting\.cli|run_fast_sweep|run_perf_all' 2>/dev/null \
         | grep -v "npu_lock" || true)"

if [ -n "$busy" ] || [ -n "$stray" ]; then
  echo "npu_lock: NPU is NOT idle, refusing to start." >&2
  [ -n "$busy" ] && { echo "--- npu-smi ---" >&2; echo "$busy" >&2; }
  [ -n "$stray" ] && { echo "--- host processes ---" >&2; echo "$stray" >&2; }
  exit 76
fi

echo "[npu_lock] start $(date -Is): $*" >&2

rc=0
if [ "$CMD_TIMEOUT" -gt 0 ]; then
  timeout --signal=TERM --kill-after=30 "$CMD_TIMEOUT" "$@" 8>&- 9>&- || rc=$?
else
  "$@" 8>&- 9>&- || rc=$?
fi

# --- post-flight: report anything the command orphaned ----------------------
left="$(pgrep -af 'msprof|[p]ytest|[t]ileops\.reporting\.cli' 2>/dev/null \
        | grep -v "npu_lock" || true)"
if [ -n "$left" ]; then
  echo "[npu_lock] WARNING: leftover processes after command:" >&2
  echo "$left" >&2
fi

echo "[npu_lock] end   $(date -Is): rc=$rc" >&2
exit "$rc"
