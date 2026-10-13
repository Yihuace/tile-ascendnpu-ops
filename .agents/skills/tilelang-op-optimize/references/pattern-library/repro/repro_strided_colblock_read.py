# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-gm-to-ub-bw-dilution + PL-1.19-scatter-col-write-amplification:
copy-only control A/B -- (M,tn) 2D strided
column-block GM reads are NOT slower than per-row (1,tn) reads (ratio ~0.98).

This file is the falsification record for a misread metric: the r3d partial's
gm_to_ub_bw 0.8-1.11 GB/s/core was total-time dilution from scalarized
compute (mte2_active_bw was 24.9 GB/s all along), NOT slow reads. The real
r3d->r3e delta was the (M,tn) fused-ite scalarization fix
(TRAP-parallel2d-bm-ge2-scalarize). Assertion: |ratio - 1| < 0.2 (no read-form
difference), with correctness on the copied payload.First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

import os
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, N, TN, NC, CORES = 4, 102400, 1280, 80, 48


def build(per_row):
    @tilelang.jit(out_idx=[1], target="npuir")
    def _func():
        @T.prim_func
        def main(x: T.Tensor((M, N), "float16"), out: T.Tensor((NC, M, TN), "float16")):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                x_ub = T.alloc_shared((M, TN), "float16")
                x_r = T.alloc_shared((1, TN), "float16")
                for _rep in T.serial(8):
                    for s in T.serial(-(-NC // CORES)):
                        cc = cid + s * CORES
                        if cc < NC:
                            if per_row:
                                for r in range(M):
                                    T.copy(x[r : r + 1, cc * TN : (cc + 1) * TN], x_r)
                                    T.copy(x_r, out[cc, r : r + 1, 0:TN])
                            else:
                                T.copy(x[0:M, cc * TN : (cc + 1) * TN], x_ub[0:M, 0:TN])
                                T.copy(x_ub, out[cc, 0:M, 0:TN])

        return main

    return _func()


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    ref = torch.stack([x[:, c * TN : (c + 1) * TN] for c in range(2)]).cpu()
    k2d, k1r = build(False), build(True)
    y2d, y1r = k2d(x), k1r(x)
    for y in (y2d, y1r):
        assert torch.equal(y[:2].cpu(), ref), "copy-only kernel wrong result"
    ts = []
    for k in (k2d, k1r):
        for _ in range(3):
            k(x)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            k(x)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / 10)
    r = ts[0] / ts[1]
    print(
        f"(M,tn) 2D strided read+store: {ts[0] * 1e6:.1f}us  "
        f"per-row: {ts[1] * 1e6:.1f}us  ratio={r:.2f} "
        f"[copy-only control; expected ~1.0 -- no read-form difference]"
    )
    assert 0.8 <= r <= 1.2, "read-form control drifted; revisit the entry"
    print(
        "ASSERT PASS: CONST-gm-to-ub-bw-dilution control (reads NOT the "
        "bottleneck; the r3d floor was TRAP-parallel2d-bm-ge2-scalarize)"
    )


if __name__ == "__main__":
    main()
