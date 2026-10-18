#!/usr/bin/env bash
# Quick single-case precision sanity for a kernel module (whisper + encodec-deep).
# Usage: quick_l0.sh <module.py>
set -u
MOD="${1:?module path}"
REPO=/mnt/workspace/npuir-migration
cd "$REPO/examples/TileOPs/tileops/kernels/convolution/conv1d/conv1d_kernel"
source perf_opt/env.sh
# Resolve relative module path against repo root
case "$MOD" in
  /*) ;;
  *) MOD="$REPO/$MOD" ;;
esac
python3 - "$MOD" <<'PY'
import sys, importlib.util
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("cand", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ok = True
ok &= m._run_case(1, 80, 3000, 1280, 3, 0, 0, 1, False, __import__('torch').float16, "whisper-nobias", seed=0)[0]
ok &= m._run_case(1, 80, 3000, 1280, 3, 0, 0, 1, True, __import__('torch').float16, "whisper-bias", seed=1)[0]
ok &= m._run_case(1, 128, 600, 256, 10, 0, 0, 1, False, __import__('torch').float16, "encodec-nobias", seed=2)[0]
ok &= m._run_case(1, 128, 600, 256, 10, 0, 0, 1, True, __import__('torch').float16, "encodec-bias", seed=3)[0]
print("QUICK_L0", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
