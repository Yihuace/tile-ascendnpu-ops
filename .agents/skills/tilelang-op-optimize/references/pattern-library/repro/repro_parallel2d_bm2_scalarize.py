# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-parallel2d-bm-ge2-scalarize: 2D T.Parallel(bm,N) fused if_then_else
scalarizes at bm>=2 regardless of operand form (ext_brc materialized + fragment
staging still 0.95 scalar); the explicit vector-op chain is the fast path.
Assertion: fused form >= 3x slower on (2048,4096) fp16 bm=2; both bit-exact.First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

import os
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30
M, N, BM, CORES = 2048, 4096, 2, 48


def build(fused):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        @T.prim_func
        def main(x: T.Tensor((M, N), "float16"), out: T.Tensor((M,), "int64")):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((BM, N), "float16")
                x_w = T.alloc_fragment((BM, N), "float16")
                row_ext = T.alloc_fragment((BM, 1), "float16")
                ext_brc = T.alloc_fragment((BM, N), "float16")
                cmp_eq = T.alloc_fragment((BM, N), "bool")
                idx_j = T.alloc_fragment((BM, N), "float32")
                sent_v = T.alloc_fragment((BM, N), "float32")
                cand = T.alloc_fragment((BM, N), "float32")
                first = T.alloc_fragment((BM, 1), "float32")
                out_ub = T.alloc_shared((BM,), "int64")
                if not fused:
                    T.arange(idx_j, [0, 1], 0)
                    T.vbrc(T.float32(BIG), sent_v)
                for s in T.serial(-(-M // BM // CORES)):
                    bid = s * CORES + cid
                    if bid < (M + BM - 1) // BM:
                        off = bid * BM
                        T.copy(x[off : off + BM, 0:N], x_ub[0:BM, 0:N])
                        T.copy(x_ub, x_w)
                        T.reduce_max(x_w, row_ext, dim=1)
                        T.vbrc(row_ext, ext_brc)
                        if fused:
                            # ext_brc materialized + fragment staging: STILL
                            # scalarizes at bm>=2 (operand-form control).
                            for i, j in T.Parallel(BM, N):
                                cand[i, j] = T.if_then_else(
                                    x_w[i, j] == ext_brc[i, j],
                                    T.cast(j, "float32"),
                                    T.float32(BIG),
                                )
                        else:
                            T.vcmp(x_w, ext_brc, cmp_eq, "eq")
                            T.vselect(cmp_eq, idx_j, sent_v, cand)
                        T.reduce_min(cand, first, dim=1)
                        for i in T.Parallel(BM):
                            out_ub[i] = T.cast(first[i, 0], "int64")
                        T.copy(out_ub[0:BM], out[off : off + BM])

        return main

    return _func(BM)


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ref = torch.argmax(x.cpu(), dim=-1)
    kf, kc = build(True), build(False)
    assert torch.equal(kf(x).cpu(), ref) and torch.equal(kc(x).cpu(), ref)
    ts = []
    for k in (kf, kc):
        for _ in range(5):
            k(x)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            k(x)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / 20)
    print(
        f"fused-ite(bm=2): {ts[0] * 1e6:.1f}us  chain: {ts[1] * 1e6:.1f}us  "
        f"ratio={ts[0] / ts[1]:.2f}"
    )
    assert ts[0] >= 3 * ts[1], "bm>=2 fused-ite scalarization not reproduced"
    print("ASSERT PASS: TRAP-parallel2d-bm-ge2-scalarize reproduced")


if __name__ == "__main__":
    main()
