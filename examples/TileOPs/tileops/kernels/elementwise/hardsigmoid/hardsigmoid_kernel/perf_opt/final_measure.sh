#!/usr/bin/env bash
# HardsigmoidFwdOp Stage-4 finalization (one lock acquisition).
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "==== STEP F1: final phase measurements on HardsigmoidFwdOp.py ===="
for wk_dt in "fc-wide float16" "fc-wide bfloat16" "mbv3-se-gate float16" "mbv3-se-gate bfloat16" "mbv3-se-gate-deep float16" "mbv3-se-gate-deep bfloat16"; do
  set -- $wk_dt
  python3 measure.py --module HardsigmoidFwdOp.py --workload "$1" --dtype "$2" \
      --phase final --candidate pipe2 --round 2 --tag "final-$1-$2-r2"
done

echo "==== STEP F2: full precision regression (--level all) ===="
python3 HardsigmoidFwdOp.py --level all 2>&1 | tail -15
echo "LEVEL_ALL_EXIT=${PIPESTATUS[0]}"

echo "==== FINALIZE DONE ===="
