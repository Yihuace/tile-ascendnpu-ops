#!/usr/bin/env bash
# HardsigmoidFwdOp Stage-4 batch measurement (one lock acquisition).
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "==== STEP 1: L0 gate on pipe2 candidate ===="
python3 HardsigmoidFwdOp.py --level L0 2>&1 | tail -6
echo "L0_EXIT=${PIPESTATUS[0]}"

echo "==== STEP 2: tiny-shape compile check (iters_a=0 path) ===="
python3 -c "
import HardsigmoidFwdOp as K
import torch, torch_npu
for N, dt in ((7680,'float16'), (30720,'bfloat16')):
    k = K.hardsigmoid_fwd_kernel(N, dt)
    x = torch.randn(N, dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}[dt]).npu()
    y = k(x, N).cpu()
    g = K.golden_hardsigmoid(x)
    print(f'iters_a=0 check N={N} {dt}: ok={bool(torch.allclose(y, g, atol=1e-3, rtol=1e-3))}')
" 2>&1 | grep -v "Warning\|WARNING\|CUTLASS\|Composable\|Workspace\|compile success\|Permission"

echo "==== STEP 3: fc-wide baseline + pipelined (fp16/bf16) ===="
python3 measure.py --module hardsigmoid_fwd_kernel.py --workload fc-wide --dtype float16 --phase baseline --candidate baseline --round 0 --tag baseline-fcwide-fp16-r0
python3 measure.py --module hardsigmoid_fwd_kernel.py --workload fc-wide --dtype bfloat16 --phase baseline --candidate baseline --round 0 --tag baseline-fcwide-bf16-r0
python3 measure.py --module HardsigmoidFwdOp.py --workload fc-wide --dtype float16 --phase candidate --candidate pipe2 --round 1 --tag pipe2-fcwide-fp16-r1
python3 measure.py --module HardsigmoidFwdOp.py --workload fc-wide --dtype bfloat16 --phase candidate --candidate pipe2 --round 1 --tag pipe2-fcwide-bf16-r1

echo "==== STEP 4: tiny shapes (baseline + pipe2) ===="
for wk in mbv3-se-gate mbv3-se-gate-deep; do
  for dt in float16 bfloat16; do
    python3 measure.py --module hardsigmoid_fwd_kernel.py --workload "$wk" --dtype "$dt" --phase baseline --candidate baseline --round 0 --tag "base-${wk}-${dt}-r0"
    python3 measure.py --module HardsigmoidFwdOp.py --workload "$wk" --dtype "$dt" --phase candidate --candidate pipe2 --round 1 --tag "pipe2-${wk}-${dt}-r1"
  done
done

echo "==== BATCH DONE ===="
