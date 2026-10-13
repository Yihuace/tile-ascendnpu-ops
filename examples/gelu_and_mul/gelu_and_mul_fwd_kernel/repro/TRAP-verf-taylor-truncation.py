# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro: npuir ``T.verf`` is a truncated Maclaurin series, NOT a real erf.

Finding (Stage 3, gelu_and_mul_fwd_kernel, 2026-10-13):
    ``T.verf`` is advertised as a native vector error function
    (docs/Tilelang.language/数学操作/T.verf.md, value range (-1, 1), "no
    special shape requirement").  On this toolchain it is actually lowered
    by ``CodeGenTileLangNPUIRAPI::VerfCodegen``
    (tilelang-mlir-ascend/src/target/codegen_npuir_api.cc:2423) to a
    truncated Taylor/Maclaurin series::

        erf(x) ~= (2/sqrt(pi)) * (x - x^3/3 + x^5/10 - x^7/42)

    This series is only usable for |x| <~ 1.  The official test
    testing/npuir/arith_ops/test_verf_dev.py only samples inputs in
    (-1, 1) at rtol=atol=1e-2; it never exercises the saturation domain
    that a GELU-style operator needs (gate/sqrt(2) ~ N(0, 0.5), so ~16%
    of samples have |x| > 1).

Assertions below (run on a real NPU) capture the trap:
    1. |verf(x) - erf(x)| on [-1, 1] is within ~5e-3 (usable there);
    2. |verf(x) - erf(x)| at |x| = 2 is > 0.5  (catastrophic);
    3. verf(3.6) has the wrong sign and huge magnitude (divergence);
    4. verf(+inf), verf(-inf) and verf(nan) are all NaN.

Run:  python TRAP-verf-taylor-truncation.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)


@tilelang.jit(out_idx=[1], target="npuir")
def _verf(blk):
    @T.prim_func
    def main(x: T.Tensor((blk,), "float32"), y: T.Tensor((blk,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((blk,), "float32")
            b = T.alloc_ub((blk,), "float32")
            T.copy(x[0:blk], a[0:blk])
            T.verf(a, b)
            T.copy(b[0:blk], y[0:blk])

    return main


def _verf_on(values):
    blk = 256
    x = torch.zeros(blk, dtype=torch.float32)
    for i, v in enumerate(values):
        x[i] = v
    y = _verf(blk)(x.npu()).cpu()
    return [y[i].item() for i in range(len(values))]


def _is_nan(v):
    return v != v


def main():
    pts = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.6,
           float("inf"), float("-inf"), float("nan")]
    res = _verf_on(pts)

    for v, r in zip(pts, res):
        print(f"verf({v!r:>8}) = {r!r:>12}   erf = {torch.erf(torch.tensor(v)).item()!r}")

    # 1. within [-1, 1] the approximation is (loosely) usable.
    for i, v in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
        err = abs(res[i + 1] - torch.erf(torch.tensor(v)).item())
        assert err < 5e-3, f"|verf-erf| at {v} = {err} unexpectedly large"
    # 2. at |x| = 2 the truncation error is catastrophic (> 0.5).
    for i in (0, 6):
        err = abs(res[i] - torch.erf(torch.tensor(pts[i])).item())
        assert err > 0.5, f"|verf-erf| at {pts[i]} = {err} (expected > 0.5)"
    # 3. at x = 3.6 the series has diverged (sign is even wrong).
    assert res[7] < 0.0, f"verf(3.6) = {res[7]} (expected a large negative value)"
    # 4. +-inf / nan all collapse to nan.
    assert _is_nan(res[8]), "verf(+inf) should be NaN"
    assert _is_nan(res[9]), "verf(-inf) should be NaN"
    assert _is_nan(res[10]), "verf(nan) should be NaN"

    print("TRAP-verf-taylor-truncation: all assertions hold")


if __name__ == "__main__":
    main()
