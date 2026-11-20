#!/usr/bin/env python3
"""R8 UB-budget compile probe: build one candidate kernel variant and report
whether the backend accepts the requested block_m (compile only, no launch)."""
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    name = "r8probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    path = os.path.join(HERE, "..", sys.argv[1])
    M, N = int(sys.argv[2]), int(sys.argv[3])
    dtype = sys.argv[4]
    bm = int(sys.argv[5])
    mod = load(path)
    print(f"[probe] {os.path.basename(path)} M={M} N={N} dtype={dtype} bm={bm} "
          f"auto_mb={mod._AUTO_MULTI_BUFFER} "
          f"footprint={mod._ub_footprint_bytes(bm, N, dtype)} B")
    try:
        mod._fused_add_layer_norm_kernel(M, N, 1e-5, dtype)(bm)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        key = [ln for ln in text.splitlines() if "overflow" in ln or "ValueError" in ln]
        print(f"[probe] RESULT=FAIL {type(exc).__name__}: " + (" | ".join(key[:2]) or text[:300]))
        return 1
    print("[probe] RESULT=OK (compiled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
