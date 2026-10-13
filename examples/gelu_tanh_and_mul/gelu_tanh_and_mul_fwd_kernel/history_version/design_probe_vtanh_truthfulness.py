# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Design-time probe (D-3, probe #1): is npuir ``T.vtanh`` a faithful tanh?

Purpose
-------
Stage 1 of the ``gelu_tanh_and_mul_fwd_kernel`` migration (2026-10-13) must
decide whether the source GPU activation ``T.tanh`` can be carried onto npuir
as the native vector ``T.vtanh``, or whether an explicit mechanism is needed
(analogous to the ``T.verf`` truncation found for the erf sibling — CG-2026-0021).

Method
------
Compile and run a minimal ``T.vtanh`` kernel on the attached NPU under both the
default mode and ``TILELANG_ASCEND_MODE=Developer``, and compare the hardware
output against ``torch.tanh`` (fp64) at a spread of points including
``+-inf`` / ``nan``.  Minimal tensor, minimal kernel — this probe only measures
the unknown (is the native op faithful?), not the operator.

Result (2026-10-13, toolchain tilelang 0.1.2 dev-root + CANN 8.5.2 + Ascend910B3)
--------------------------------------------------------------------------------
``tl.npuir_vtanh`` is mode-dispatched and NEITHER path is faithful over the
whole real line:

*  default (``TILELANG_ASCEND_MODE`` unset) -> ``CodeGenTileLangNPUIRAPI::
   VtanhCodegen`` (``codegen_npuir_api.cc``): truncated Maclaurin
   ``x - x^3/3 + 2x^5/15 - 17x^7/315``, usable only for |x| <~ 0.5:

       vtanh( 2.0) = 3.3079  (tanh = 0.9640)   <- wrong sign
       vtanh( 3.0) = -91.63  (tanh = 0.9951)   <- diverged
       vtanh( inf) = nan     (tanh = 1.0)

*  Developer (``TILELANG_ASCEND_MODE=Developer``) -> ``CodeGenTileLangNPUIRDEV::
   VtanhCodegen`` (``codegen_npuir_dev.cc``): exp identity
   ``(e^{2x}-1)/(e^{2x}+1)``, exact for finite |x| <~ 44 but overflows beyond:

       vtanh(50.0) = nan     (tanh = 1.0)      <- e^{100} overflow
       vtanh( inf) = nan     (tanh = 1.0)

Conclusion: no mode gives a faithful tanh over the GELU-tanh argument range
(``|u|`` reaches ~10 for a standard-normal gate and ``+inf`` at the boundary),
so the design uses an explicit sigmoid-based mechanism (probe #2).  Note also
that the two runs must use distinct block sizes: the on-disk kernel cache key is
not verified to include ``TILELANG_ASCEND_MODE``, so a shared key can return the
wrong mode's kernel (observed while probing).

Run:  python history_version/design_probe_vtanh_truthfulness.py [default|Developer]
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


def main():
    pts = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0,
           float("inf"), float("-inf"), float("nan")]
    # Distinct block size per mode avoids a shared compile-cache entry (the
    # codegen is mode-dispatched but the cache key is not verified to include
    # TILELANG_ASCEND_MODE).
    blk = 512 if mode == "default" else 544
    x = torch.zeros(blk, dtype=torch.float32)
    for i, v in enumerate(pts):
        x[i] = v
    y = _vtanh(blk)(x.npu()).cpu()
    print("MODE =", mode)
    for i, v in enumerate(pts):
        print(f"vtanh({v!r:>6}) = {y[i].item()!r:>14}   "
              f"tanh = {torch.tanh(torch.tensor(v)).item()!r}")


if __name__ == "__main__":
    main()
