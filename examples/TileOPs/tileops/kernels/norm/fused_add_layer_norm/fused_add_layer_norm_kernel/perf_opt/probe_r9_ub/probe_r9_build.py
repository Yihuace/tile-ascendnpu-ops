#!/usr/bin/env python3
"""R9 UB probe: build one kernel variant with the host-side guard bypassed so
that the *backend* is the authority on whether a (block_m, N) working set fits.

    python probe_r9_build.py <module.py> <M> <N> <dtype> <bm> <mb:0|1>

Reports the model footprint, then compile OK/FAIL plus the backend's
`requires ... bits` line when it rejects.
"""
import importlib.util
import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    name = "r9probe_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    path = os.path.join(HERE, "..", sys.argv[1])
    M, N = int(sys.argv[2]), int(sys.argv[3])
    dtype = sys.argv[4]
    bm = int(sys.argv[5])
    mb = sys.argv[6] == "1"
    mod = load(path)
    mod._AUTO_MULTI_BUFFER = mb
    mod._JIT_KWARGS["pass_configs"]["npuir.enable_auto_multi_buffer"] = mb
    mod._UB_SAFETY_BYTES = -10 ** 9  # bypass host guard; backend decides
    print(f"[probe] {os.path.basename(path)} M={M} N={N} dtype={dtype} bm={bm} "
          f"auto_mb={mb} model_footprint={mod._ub_footprint_bytes(bm, N, dtype)} B")
    try:
        mod._fused_add_layer_norm_kernel(M, N, 1e-5, dtype)(bm)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        keys = [ln for ln in text.splitlines()
                if "overflow" in ln or "requires" in ln or "ValueError" in ln]
        print(f"[probe] RESULT=FAIL {type(exc).__name__}: "
              + (" | ".join(keys[:2]) or text[-400:]))
        return 1
    print("[probe] RESULT=OK (compiled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
