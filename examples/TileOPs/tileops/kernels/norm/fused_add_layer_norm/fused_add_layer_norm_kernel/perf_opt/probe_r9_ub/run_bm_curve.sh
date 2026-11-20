#!/usr/bin/env bash
# R9 diagnostic: block_m scaling curve of the *unmodified* R8 kernel at
# N=2048 (fp16), where block_m up to 5 already fits in UB.  One msprof run per
# (bm), median of 20 Task Duration samples, A/B/A/B-free (single knob).
set -uo pipefail
cd "$(dirname "$0")"
export TILELANG_ASCEND_MODE=Developer
MOD=${MOD:-../FusedAddLayerNormFwdOp.py}
N=${N:-2048}
M=${M:-2048}
DT=${DT:-float16}
OUT=profiles
mkdir -p "$OUT"
for bm in "$@"; do
  dir="$OUT/probe-bmcurve-N${N}-${DT}-bm${bm}"
  rm -rf "$dir"; mkdir -p "$dir"; chmod 700 "$dir"
  msprof op --kernel-name=main --launch-count=20 --warm-up=5 \
    --aic-metrics=Default,Roofline --output="$dir" \
    python3 bench_shape.py --module "$MOD" --m "$M" --n "$N" --dtype "$DT" \
    --block-m "$bm" --launches 25 > "logs_bmcurve_N${N}_bm${bm}.log" 2>&1
  best=$(python3 - "$dir" <<'PY'
import csv, glob, statistics, sys
rows=[]
for p in glob.glob(sys.argv[1]+"/OPPROF_*/*/*/OpBasicInfo_*.csv"):
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("Task Duration(us)"):
                rows.append(float(r["Task Duration(us)"]))
print(f"{statistics.median(rows):.3f} n={len(rows)}" if rows else "FAIL")
PY
)
  echo "bm=$bm TaskDuration=$best"
done
