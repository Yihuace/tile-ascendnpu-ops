#!/usr/bin/env bash
# Run ONE unit_stride case in isolation (retry-friendly; avoids the flaky
# bishengir SIGSEGV CG-2026-0003 killing sibling cases).
# Usage: one_case.sh <module.py> <case>  where case in {whisper-nobias, whisper-bias, encodec-nobias, encodec-bias}
set -u
MOD="${1:?module path}"
CASE="${2:?case}"
REPO=/mnt/workspace/npuir-migration
cd "$REPO/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel"
source perf_opt/env.sh
case "$MOD" in /*) ;; *) MOD="$REPO/$MOD" ;; esac
CASE="$CASE" python3 - "$MOD" <<'PY'
import sys, importlib.util, os
path = sys.argv[1]
case = os.environ["CASE"]
spec = importlib.util.spec_from_file_location("cand", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import torch
cases = {
    "whisper-nobias":  (1, 80, 3000, 1280, 3, False),
    "whisper-bias":    (1, 80, 3000, 1280, 3, True),
    "encodec-nobias":  (1, 128, 600, 256, 10, False),
    "encodec-bias":    (1, 128, 600, 256, 10, True),
}
n, c_in, l_in, c_out, k, bias = cases[case]
ok = m._run_case(n, c_in, l_in, c_out, k, 0, 0, 1, bias, torch.float16, case, seed=0)[0]
print("ONE_CASE", case, "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
