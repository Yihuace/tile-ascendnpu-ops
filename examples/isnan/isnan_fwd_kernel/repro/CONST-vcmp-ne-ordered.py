# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""repro/CONST-vcmp-ne-ordered.py

Overturns / corrects pattern-library entry ``CONST-vcmp-ne-nan-unordered``
(INDEX.md:116) for this toolchain: on the current npuir build, ``T.vcmp``'s
``"ne"`` mode is lowered as an **ordered** not-equal comparison, so a
self-compare ``vcmp(x, x, mask, "ne")`` is **false** for NaN -- it does NOT
detect NaN (contradicting the design of isnan_fwd_kernel section 0.6 R2 /
section 9.1, which assumes IEEE-unordered ``ne``).

Finding (measured 2026-10-14, tilelang 0.1.2 dev-root
/home/developer/tilelang-mlir-ascend/build + Ascend 910B2C 20 AI Core +
CANN 8.5.2):

    mode  NaN   +inf  -inf  1.0  0.0  -0.0
    ne    0     0     0     0    0    0     <- ordered: NaN self-compare == false
    eq    0     1     1     1    1    1
    le    0     1     1     1    1    1     <- false ONLY for NaN
    ge    0     1     1     1    1    1     <- false ONLY for NaN

Because ``Ascend ne == (lt OR gt)`` (ordered), NaN -> false.  A discriminating
workaround for ``isnan`` (validated exact for fp32/fp16/bfloat16) is the logical
complement of the ``le`` self-compare:  ``isnan(x) == NOT (x <= x)``, since
``le(x, x)`` / ``ge(x, x)`` are true for every non-NaN value (incl. +-inf / +-0 /
subnormal) and false for NaN.

Note: the older repro ``CONST-vcmp-ne-nan-unordered.py`` still *passes* on this
toolchain, but it is non-discriminating -- its hardsigmoid arithmetic chain
(vadd -> vclamp -> vmul) already propagates NaN, so NaN survives to the output
regardless of whether the ``vcmp "ne"`` mask fired.  This repro reads the mask
out directly (vcast(bool)->f16) and therefore actually discriminates.

Run:
    python repro/CONST-vcmp-ne-ordered.py
Expected: all asserts pass (exit 0).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

BS = 256
_VALS = [float("nan"), float("inf"), float("-inf"), 1.0, 0.0, -0.0]


@tilelang.jit(out_idx=[1], target="npuir")
def mask_probe(N, dtype, mode):
    """Read the raw vcmp mask out as f16 (1.0/0.0) for a given cmp mode."""

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((BS,), dtype)
            a_ub = T.alloc_ub((BS,), dtype)
            m_ub = T.alloc_ub((BS,), "bool")
            f_ub = T.alloc_ub((BS,), "float16")
            y_ub = T.alloc_ub((BS,), "float32")
            T.copy(x[0:BS], x_ub[0:BS])
            T.copy(x[0:BS], a_ub[0:BS])  # distinct buffer: rules out same-operand folding
            T.vcmp(x_ub, a_ub, m_ub, mode)
            T.vcast(m_ub, f_ub, round_mode="rint")
            T.vcast(f_ub, y_ub, round_mode="rint")
            T.copy(y_ub[0:BS], y[0:BS])

    return main


@tilelang.jit(out_idx=[1], target="npuir")
def isnan_via_le(N, dtype):
    """isnan = NOT (x <= x), byte output; mirrors the design R3 byte chain."""
    one = T.cast(1.0, "float16")

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), "int8")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((BS,), dtype)
            mask = T.alloc_ub((BS,), "bool")
            f_ub = T.alloc_ub((BS,), "float16")
            one_ub = T.alloc_ub((BS,), "float16")
            o_ub = T.alloc_ub((BS,), "float16")
            y_ub = T.alloc_ub((BS,), "int8")
            T.copy(x[0:BS], x_ub[0:BS])
            T.vcmp(x_ub, x_ub, mask, "le")            # true = non-NaN
            T.vcast(mask, f_ub, round_mode="rint")    # 1.0 non-NaN / 0.0 NaN
            T.fill(one_ub, one)
            T.vsub(one_ub, f_ub, o_ub)                # 0.0 non-NaN / 1.0 NaN
            T.vcast(o_ub, y_ub, round_mode="rint")
            T.copy(y_ub[0:BS], y[0:BS])

    return main


def _input(dtype_str):
    x = torch.tensor(_VALS + [2.0] * (BS - len(_VALS)), dtype=torch.float32)
    return x.to(getattr(torch, dtype_str))


def main():
    for dt in ("float32", "float16", "bfloat16"):
        x = _input(dt)
        ne = mask_probe(BS, dt, "ne")(x.npu()).cpu()
        eq = mask_probe(BS, dt, "eq")(x.npu()).cpu()
        le = mask_probe(BS, dt, "le")(x.npu()).cpu()

        # Ordered "ne": NaN self-compare is FALSE (the overturned behavior).
        assert ne[0].item() == 0.0, f"{dt}: vcmp 'ne' NaN expected 0 (ordered), got {ne[0].item()}"
        # eq / le are the ordered IEEE results: NaN -> false, others -> true.
        assert eq[0].item() == 0.0 and le[0].item() == 0.0
        assert bool((eq[1:6] == 1.0).all()) and bool((le[1:6] == 1.0).all())

        # Workaround: isnan = NOT(x <= x) is exact.
        y = isnan_via_le(BS, dt)(x.npu()).cpu().view(torch.bool)
        assert torch.equal(y, torch.isnan(x.float())), f"{dt}: le-invert isnan mismatch"
        print(f"[PASS] {dt}: vcmp 'ne' NaN self-compare = 0 (ordered); "
              f"NOT(x<=x) isnan exact (repro overturns CONST-vcmp-ne-nan-unordered)")

    print("\033[92mCONST-vcmp-ne-ordered: repro confirms ordered 'ne' + le-invert workaround\033[0m")


if __name__ == "__main__":
    main()
