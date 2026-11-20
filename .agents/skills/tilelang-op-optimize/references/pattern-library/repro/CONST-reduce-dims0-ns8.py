# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-reduce-dims0-ns8 -- ``T.reduce_sum(dim=0)`` on a skinny ``(ns, 8)``
UB board is exact and costs nothing; the "dim=0 on a narrow second axis is a
per-column serial walk" fear does not materialise at this size.

Entry:  constants.md / CONST-reduce-dims0-ns8
Kind:   constant.  Assertions (runnable part):
          * value is exact vs torch (rel err <= 1e-6) for ns = 4 and ns = 8;
          * two consecutive runs are bit-identical (deterministic, no atomics);
        Measurement part (printed only; absolute values are environment-bound):
          msprof op --kernel-name=main --launch-count=10 --warm-up=3 \\\\
              --aic-metrics=BasicInfo,Roofline python <driver> --stage B
        On the origin task, replacing the merge by a single-row copy changed
        stage-B Duration by -0.07 us (b4) / +0.09 us (b1), i.e. inside run
        noise -- the merge is free, and a ``T.serial(ns)`` scalar fallback was
        not faster (-0.06 us), so ``dim=0`` is kept.

Why the fear existed: docs/Tilelang.language/规约操作/T.reduce_sum.md §2.4 and
the in-repo precedent (_batch_norm_bwd_split_kernel) both reduce the *last*
axis; ``dst`` only has to be 1 on the reduced axis (§2.2.2), so
``(ns,8)->(1,8)`` is legal but was untested for this shape class.

First verified: 2026-10-31, tilelang 0.1.2+3ed7c91420e925ff3ec388fb981dd6ebf45ff4bf
+ CANN 9.0.0 + Ascend910B3, TILELANG_ASCEND_MODE=Developer.
origin_task: examples/engram_decode_mcore/engram_decode_mcore (Stage 3)
Re-verify: python repro/CONST-reduce-dims0-ns8.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

F32 = "float32"
ROWS = 4
COLS = 8


def _build(ns):
    @tilelang.jit(target="npuir")
    def factory():
        @T.prim_func
        def main(
            src: T.Tensor((ROWS, ns, COLS), F32),
            dst: T.Tensor((ROWS, COLS), F32),
        ):
            with T.Kernel(ROWS, is_npu=True) as (bid, _):
                prow = T.alloc_shared((ns, COLS), F32)
                tot = T.alloc_shared((1, COLS), F32)
                T.copy(src[bid, :, :], prow)
                T.reduce_sum(prow, tot, dim=0)
                T.copy(tot[0, :], dst[bid, :])

        return main

    return factory()


if __name__ == "__main__":
    torch.manual_seed(0)
    for ns in (4, 8):
        src = torch.randn(ROWS, ns, COLS, dtype=torch.float32).npu()
        ref = src.cpu().sum(1)
        outs = []
        for _ in range(2):
            dst = torch.zeros(ROWS, COLS, dtype=torch.float32).npu()
            _build(ns)(src, dst)
            torch.npu.synchronize()
            outs.append(dst.cpu())
        rel = ((outs[0] - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
        assert rel <= 1e-6, f"ns={ns}: reduce(dim=0) wrong, rel={rel:.2e}"
        assert torch.equal(outs[0], outs[1]), f"ns={ns}: not deterministic"
        print(f"[ok] ns={ns}: reduce_sum(dim=0) rel_err={rel:.1e}, "
              f"bit-identical across runs")
    print("OVERALL: CONST_PASS (timing is free-form; see header)")
