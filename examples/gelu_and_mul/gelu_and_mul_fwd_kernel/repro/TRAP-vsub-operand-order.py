# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: ``T.vsub(src0, src1, dst)`` is ``src0 - src1`` and rejects a scalar ``src0``.

Finding (Stage 3, gelu_and_mul_fwd_kernel, 2026-10-13):
    The DESIGN.md pseudocode for the clamped polynomial erf computes the
    polynomial variable as ``T.vsub(w32, 1.0, w32)  # w = 1 - u^2``.  But
    ``T.vsub`` is documented as ``src0 - src1`` with ``src1`` allowed to be a
    scalar and ``src0`` NOT allowed to be a scalar
    (docs/Tilelang.language/数学操作/T.vsub.md section 2.3).  So that call
    actually computes ``u^2 - 1`` (the negation), silently producing a wrong
    ``w`` in ``[-1, 0]`` instead of ``[0, 1]``.  Every Horner term then came out
    with the wrong sign and the full ``--level L0`` suite failed.

    A scalar-first subtraction (``1 - u^2``) is NOT expressible in one call on
    this front end.  The working emulation is to negate then add::

        T.vmul(u32, u32, w32)      # w = u^2
        T.vmul(w32, neg_one, w32)  # w = -u^2
        T.vadd(w32, one, w32)      # w = 1 - u^2

Assertions below (run on a real NPU) capture the trap:
    1. ``vsub(a, 1.0)`` equals ``a - 1`` for every sample;
    2. it does NOT equal ``1 - a`` (the intended but unavailable form) whenever
       ``a != 1``, so operand order matters and cannot be assumed.

Run:  python TRAP-vsub-operand-order.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)


@tilelang.jit(out_idx=[1], target="npuir")
def _vsub_minus_one(blk):
    @T.prim_func
    def main(x: T.Tensor((blk,), "float32"), y: T.Tensor((blk,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((blk,), "float32")
            b = T.alloc_ub((blk,), "float32")
            T.copy(x[0:blk], a[0:blk])
            T.vsub(a, T.cast(1.0, "float32"), b)  # b = a - 1 (src1 scalar)
            T.copy(b[0:blk], y[0:blk])

    return main


def main():
    blk = 256
    vals = [3.0, 0.0, -2.0, 0.5, 10.0]
    x = torch.zeros(blk, dtype=torch.float32)
    for i, v in enumerate(vals):
        x[i] = v
    y = _vsub_minus_one(blk)(x.npu()).cpu()

    for v in vals:
        i = vals.index(v)
        print(f"vsub({v!r:>5}, 1.0) = {y[i].item():>6}   a-1 = {v - 1.0:>6}   1-a = {1.0 - v:>6}")

    # 1. vsub(tensor, scalar) == tensor - scalar.
    for v in vals:
        i = vals.index(v)
        assert abs(y[i].item() - (v - 1.0)) < 1e-6, f"vsub({v},1) != {v}-1"
    # 2. operand order matters: it is NOT 1 - a for a != 1.
    for v in vals:
        if v == 1.0:
            continue
        i = vals.index(v)
        assert abs(y[i].item() - (1.0 - v)) > 1e-6, f"vsub({v},1) unexpectedly equals 1-{v}"

    print("TRAP-vsub-operand-order: all assertions hold")


if __name__ == "__main__":
    main()
