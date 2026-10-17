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
#   .agents/tools/npu_lock.sh [--wait SEC] [--wait-idle SEC] [--timeout SEC] -- <command...>
#
#   --wait       seconds to wait for the *lock* (default 0 = fail fast)
#   --wait-idle  seconds to wait for the *device* to become idle before starting
#                (default 0 = fail fast).  Needed because other sessions in this
#                workspace do not necessarily use this lock.
#   --timeout    seconds allowed for the command itself (default 0 = no limit)
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
WAIT_IDLE_S=0

while [ $# -gt 0 ]; do
  case "$1" in
    --wait) WAIT_S="$2"; shift 2 ;;
    --timeout) CMD_TIMEOUT="$2"; shift 2 ;;
    --wait-idle) WAIT_IDLE_S="$2"; shift 2 ;;
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
_busy_detail=""
_stray_detail=""
_is_idle() {
  _busy_detail=""
  _stray_detail=""
  if command -v npu-smi >/dev/null 2>&1; then
    if ! npu-smi info 2>/dev/null | grep -q "No running processes found"; then
      _busy_detail="$(npu-smi info 2>/dev/null | sed -n '/Process id/,/^+/p' | head -20)"
    fi
  fi
  # Stray host processes that would compete for the device.  Match the real
  # workloads, not this script (the bracket trick avoids self-matching).
  _stray_detail="$(pgrep -af 'msprof|[p]ytest|[t]ileops\.reporting\.cli|run_fast_sweep|run_perf_all|design_probe|_probe_env' 2>/dev/null \
                   | grep -v "npu_lock" || true)"
  [ -z "$_busy_detail" ] && [ -z "$_stray_detail" ]
}

if ! _is_idle; then
  if [ "$WAIT_IDLE_S" -gt 0 ]; then
    echo "[npu_lock] NPU busy; waiting up to ${WAIT_IDLE_S}s for it to go idle ..." >&2
    _deadline=$((SECONDS + WAIT_IDLE_S))
    while ! _is_idle; do
      if [ "$SECONDS" -ge "$_deadline" ]; then
        echo "npu_lock: NPU still busy after ${WAIT_IDLE_S}s, refusing to start." >&2
        [ -n "$_busy_detail" ] && { echo "--- npu-smi ---" >&2; echo "$_busy_detail" >&2; }
        [ -n "$_stray_detail" ] && { echo "--- host processes ---" >&2; echo "$_stray_detail" >&2; }
        exit 76
      fi
      sleep 10
    done
    echo "[npu_lock] NPU is idle now; starting." >&2
  else
    echo "npu_lock: NPU is NOT idle, refusing to start." >&2
    [ -n "$_busy_detail" ] && { echo "--- npu-smi ---" >&2; echo "$_busy_detail" >&2; }
    [ -n "$_stray_detail" ] && { echo "--- host processes ---" >&2; echo "$_stray_detail" >&2; }
    exit 76
  fi
fi

echo "[npu_lock] start $(date -Is): $*" >&2

rc=0
if [ "$CMD_TIMEOUT" -gt 0 ]; then
  timeout --signal=TERM --kill-after=30 "$CMD_TIMEOUT" "$@" 8>&- 9>&- || rc=$?
else
  "$@" 8>&- 9>&- || rc=$?
fi

# --- post-flight: reap anything the command orphaned ------------------------
# Killing a parent (timeout) does not kill forked children such as
# bishengir-compile / pytest workers; they keep the NPU busy and would poison
# the next measurement (docs/cann9_adaptation.md §12.6).  The mutex guarantees
# these can only be ours, so terminate them.  Selection is by PID with an
# explicit self/ancestor filter -- never `pkill -f` on the bare pattern, which
# would match the invoking shell's own command line (the §12.6 trap).
PATTERNS='msprof|[p]ytest|[t]ileops\.reporting\.cli|bishengir-compile'

_collect_victims() {
  local p
  for p in $(pgrep -f "$PATTERNS" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    [ "$p" = "$PPID" ] && continue
    [ -r "/proc/$p/cmdline" ] || continue
    # Any process whose command line mentions npu_lock is the wrapper or one
    # of its ancestors -- never kill it.
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "npu_lock" && continue
    echo "$p"
  done
}

victims="$(_collect_victims)"
if [ -n "$victims" ]; then
  echo "[npu_lock] WARNING: reaping leftover processes after command:" >&2
  for p in $victims; do
    echo "  pid=$p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160)" >&2
  done
  kill -TERM $victims 2>/dev/null || true
  sleep 2
  victims="$(_collect_victims)"
  [ -n "$victims" ] && kill -KILL $victims 2>/dev/null || true
  sleep 1
  left2="$(_collect_victims)"
  if [ -n "$left2" ]; then
    echo "[npu_lock] ERROR: could not reap: $left2" >&2
  else
    echo "[npu_lock] leftovers reaped" >&2
  fi
fi

echo "[npu_lock] end   $(date -Is): rc=$rc" >&2
exit "$rc"
