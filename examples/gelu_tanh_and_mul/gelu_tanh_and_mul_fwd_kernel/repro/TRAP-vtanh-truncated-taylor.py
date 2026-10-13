# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: npuir ``T.vtanh`` is NOT a faithful tanh over the operator's range.

Finding (Stage 1, gelu_tanh_and_mul_fwd_kernel, 2026-10-13):
    ``T.vtanh`` is the npuir vector hyperbolic tangent
    (``docs/Tilelang.language/数学操作/T.vtanh.md``); ``T.tanh`` is only a TIR
    *scalar* expression (``tilelang/language/tir/op.py``).  ``tl.npuir_vtanh`` is
    lowered by ``TILELANG_ASCEND_MODE`` to one of two bodies, NEITHER faithful
    over the whole real line (distribution of the GELU-tanh argument
    ``u = sqrt(2/pi)(g + 0.044715 g^3)`` reaches |u| ~ 10 for a standard-normal
    gate and +inf at the boundary):

      * default / Expert -> ``CodeGenTileLangNPUIRAPI::VtanhCodegen``
        (``tilelang-mlir-ascend/src/target/codegen_npuir_api.cc`` ~2526):
        truncated Maclaurin series ``x - x^3/3 + 2x^5/15 - 17x^7/315``,
        valid only for |x| <~ 0.5, diverges beyond, ``±inf -> NaN``.
      * Developer -> ``CodeGenTileLangNPUIRDEV::VtanhCodegen``
        (``tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc`` ~3365):
        exp identity ``(e^{2x}-1)/(e^{2x}+1)``; exact for finite |x| <~ 44 but
        ``e^{2x}`` overflows (fp32) so ``x >= ~44`` and ``+inf`` give NaN; the
        GELU-tanh boundary needs ``tanh(+inf) = +1``.

    Since ``TILELANG_ASCEND_MODE`` is process-global and read at trace time
    (CG-2026-0013), the mode is not stable, and a faithful tanh is never
    available.  This is the tanh analogue of the ``T.verf`` truncation
    (CG-2026-0021).

Assertions below (run on a real NPU), selected by the mode argument:
    default   : |vtanh-tanh| <= 5e-3 on [-0.5, 0.5], but > 0.5 at |x| = 2 and
                wrong-signed/huge at x = 3; ±inf / nan -> NaN.
    Developer : |vtanh-tanh| <= 5e-3 on [-10, 10], but vtanh(+inf) = NaN and
                vtanh(50) = NaN (overflow), while tanh(+inf) = 1.

Run:  python repro/TRAP-vtanh-truncated-taylor.py            # default mode
      python repro/TRAP-vtanh-truncated-taylor.py Developer  # developer mode
"""

import os
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "default"
if mode != "default":
    os.environ["TILELANG_ASCEND_MODE"] = mode

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402


@tilelang.jit(out_idx=[1], target="npuir")
def _vtanh(blk):
    @T.prim_func
    def main(x: T.Tensor((blk,), "float32"), y: T.Tensor((blk,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((blk,), "float32")
            b = T.alloc_ub((blk,), "float32")
            T.copy(x[0:blk], a[0:blk])
            T.vtanh(a, b)
            T.copy(b[0:blk], y[0:blk])

    return main


def _run(values):
    # Distinct block size per mode so the two runs do not share a compile-cache
    # entry (the codegen is mode-dispatched but the on-disk kernel cache key is
    # not verified to include TILELANG_ASCEND_MODE).
    blk = 512 if mode == "default" else 544
    x = torch.zeros(blk, dtype=torch.float32)
    for i, v in enumerate(values):
        x[i] = v
    y = _vtanh(blk)(x.npu()).cpu()
    return [y[i].item() for i in range(len(values))]


def _is_nan(v):
    return v != v


def main():
    if mode == "default":
        pts = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0,
               float("inf"), float("-inf"), float("nan")]
        res = _run(pts)
        for v, r in zip(pts, res):
            print(f"vtanh({v!r:>8}) = {r!r:>14}   tanh = {torch.tanh(torch.tensor(v)).item()!r}")
        for v, r in zip((-0.5, 0.0, 0.5), (res[2], res[3], res[4])):
            assert abs(r - torch.tanh(torch.tensor(v)).item()) < 5e-3
        for i in (0, 6):  # -2.0, 2.0 -> catastrophic
            assert abs(res[i] - torch.tanh(torch.tensor(pts[i])).item()) > 0.5
        assert res[7] < 0.0, f"vtanh(3.0) = {res[7]} (expected a large negative value)"
        assert _is_nan(res[8]) and _is_nan(res[9]) and _is_nan(res[10])
        print("TRAP-vtanh-truncated-taylor [default]: truncation assertions hold")
    else:
        pts = [-3.0, -2.0, 2.0, 3.0, 10.0, 50.0, float("inf"), float("-inf"), float("nan")]
        res = _run(pts)
        for v, r in zip(pts, res):
            print(f"vtanh({v!r:>8}) = {r!r:>14}   tanh = {torch.tanh(torch.tensor(v)).item()!r}")
        for i in (0, 1, 2, 3, 4):  # finite small |x| -> faithful
            assert abs(res[i] - torch.tanh(torch.tensor(pts[i])).item()) < 5e-3
        # non-faithful tail: overflow at x=50 and +inf (tanh -> 1)
        assert _is_nan(res[5]), f"vtanh(50) = {res[5]} (expected NaN overflow)"
        assert _is_nan(res[6]), f"vtanh(+inf) = {res[6]} (expected NaN)"
        assert res[7] == -1.0
        assert _is_nan(res[8])
        print("TRAP-vtanh-truncated-taylor [Developer]: overflow/NaN-tail assertions hold")


if __name__ == "__main__":
    main()
