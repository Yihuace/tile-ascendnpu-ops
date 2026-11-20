"""Round 21 compile-feasibility probe (one pass per process, wallclock-bounded).

``bishengir-compile`` has three known failure modes on this toolchain
(SIGSEGV / infinite loop / success), so every invocation is launched by the
caller under an external ``timeout`` and this script only builds ONE pass, so
the failure signature is attributable per pass.

Usage:
  timeout 300s python _r21_compile_probe.py <impl.py> <sdw|pds|mul|all> [B S H Hkv D dtype]
"""

import importlib.util
import sys
import time

import torch_npu  # noqa: F401  (device runtime must be initialised first)


def _load(path):
    spec = importlib.util.spec_from_file_location("r21_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r21_probe_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    impl = sys.argv[1]
    which = sys.argv[2]
    shape = sys.argv[3:9] if len(sys.argv) >= 9 else ["4", "512", "32", "8", "128", "float16"]
    B, S, H, Hkv, D = (int(x) for x in shape[:5])
    dtype = shape[5]

    m = _load(impl)
    bm, bn, nkv, nq, ntile, nblk = m._split_geom(B, H, Hkv, S, D)
    print(f"GEOM B{B} S{S} H{H} Hkv{Hkv} D{D} {dtype}: block_m={bm} block_n={bn} "
          f"n_kv={nkv} n_q={nq} ntile={ntile} nblk={nblk}", flush=True)
    if len(sys.argv) > 9 and bn != int(sys.argv[9]):
        print(f"UNEXPECTED_BN got={bn} want={sys.argv[9]}", flush=True)

    t0 = time.time()
    if which in ("sdw", "all"):
        m._gqa_bwd_sdw_builder(B, H, Hkv, S, D, True, dtype, bm, bn, 1)
        print(f"COMPILE_OK sdw {time.time() - t0:.1f}s", flush=True)
    if which in ("pds", "all"):
        t1 = time.time()
        m._gqa_bwd_pds_builder(B, H, Hkv, S, D, True, dtype, bm, bn, 1)
        print(f"COMPILE_OK pds {time.time() - t1:.1f}s", flush=True)
    if which in ("mul", "all"):
        t1 = time.time()
        m._gqa_bwd_mul_builder(B, H, Hkv, S, D, True, dtype, bm, bn, 1)
        print(f"COMPILE_OK mul {time.time() - t1:.1f}s", flush=True)
    print(f"PROBE_DONE {which} total={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
