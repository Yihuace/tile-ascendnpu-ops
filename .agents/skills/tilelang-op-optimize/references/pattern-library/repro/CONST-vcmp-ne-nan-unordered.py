# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-vcmp-ne-nan-unordered.py

Entry: traps-runtime.md candidate  CONST-vcmp-ne-nan-unordered
  (also resolves DESIGN.md §9.1 risk point + REVIEW.md non-blocking risk for
   the hardsigmoid migration, and the source-repo `_nan.py` "self-compare `!=`
   is not a NaN guard" note -- which applies to the GPU backend, NOT NPU).

Phenomenon: On Ascend NPU, `T.vcmp(x, x, mask, "ne")` self-compare implements
  IEEE *unordered* comparison -- NaN != NaN -> true, and +-inf/ finite -> false.
  Therefore `vcmp(x,x,"ne")` + `T.vselect` correctly recovers NaN for
  elementwise activations whose NaN source is the input alone (single-point
  recovery).  This is the inverse of the GPU TileLang backend, where `!=` is
  lowered as an *ordered* compare and a self-compare misses NaN.

Assertion (workaround / behavior passes): a NaN-propagating hardsigmoid-style
  chain (vadd -> vclamp -> vmul -> vbrc(NaN) -> vcmp(x,x,"ne") -> vselect) yields
  NaN out exactly where the input is NaN, and finite/+-inf out elsewhere, for
  float32 / float16 / bfloat16.  assert_close with equal_nan=True must pass.

Secondary observation (compiler/toolchain trap, no assert): broadcasting a NaN
  constant via `T.cast(float("nan"), "float32")` + `T.vbrc` makes the kernel
  cache fail to load its metadata (``Error loading metadata ... Wrong value
  format for field value`` at FloatImmNode::VisitAttrs -> ParseDouble), because
  JSON cannot represent NaN.  The kernel still compiles and runs correctly; the
  cache simply misses on every build (recompilation overhead).  Finite FloatImm
  constants (e.g. ``T.cast(1.0,"float32")``) do not trigger this.

First verified: 2026-10-12, tilelang dev-root /home/developer/tilelang-mlir-ascend/build,
  Ascend 910B2C (20 AI Core), CANN 8.5.0.
origin_task: hardsigmoid-hardsigmoid_fwd_kernel-20261012T132114Z (Stage 3 develop)
Re-verify log:
  2026-10-12 first run PASS (NaN->NaN for float32/float16/bfloat16; cache warn observed)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1], target="npuir")
def hsigmoid_nan_recover(N, dtype, block_size=8192):
    compute_dtype = "float32"
    three = T.cast(3.0, "float32")
    zero = T.cast(0.0, "float32")
    six = T.cast(6.0, "float32")
    one_sixth = T.cast(1.0 / 6.0, "float32")
    nan_scalar = T.cast(float("nan"), "float32")

    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), dtype)
            x_f32 = T.alloc_ub((block_size,), compute_dtype)
            buf1 = T.alloc_ub((block_size,), compute_dtype)
            buf2 = T.alloc_ub((block_size,), compute_dtype)
            mask = T.alloc_ub((block_size,), "bool")
            y_ub = T.alloc_ub((block_size,), dtype)
            offset = cid * block_size
            tail = T.min(block_size, shape - offset)
            T.copy(x[offset : offset + tail], x_ub[0:tail])
            if dtype != compute_dtype:
                T.vcast(x_ub, x_f32, round_mode="rint")
            else:
                T.copy(x_ub, x_f32)
            T.vadd(x_f32, three, buf1)
            T.vclamp(buf1, buf2, zero, six)
            T.vmul(buf2, one_sixth, buf1)
            T.vbrc(nan_scalar, buf2)
            T.vcmp(x_f32, x_f32, mask, "ne")
            T.vselect(mask, buf2, buf1, buf1)
            if dtype != compute_dtype:
                T.vcast(buf1, y_ub, round_mode="rint")
            else:
                T.copy(buf1, y_ub)
            T.copy(y_ub[0:tail], y[offset : offset + tail])

    return main


def _check(dtype_str, torch_dtype, tol):
    N = 4096
    x = torch.randn(N, dtype=torch.float32)
    x[0::8] = float("nan")
    x[1::8] = float("inf")
    x[2::8] = float("-inf")
    x = x.to(torch_dtype).npu()
    y = hsigmoid_nan_recover(N, dtype_str)(x, N).cpu()
    ref = torch.nn.functional.hardsigmoid(x.cpu().float()).to(torch_dtype)
    torch.testing.assert_close(y, ref, atol=tol, rtol=tol, equal_nan=True)
    assert torch.equal(torch.isnan(x.cpu()), torch.isnan(y)), "NaN mask mismatch"
    print(f"[PASS] dtype={dtype_str}: NaN->NaN preserved, nan_count={int(torch.isnan(y).sum())}")


if __name__ == "__main__":
    _check("float32", torch.float32, 1e-5)
    _check("float16", torch.float16, 1e-3)
    _check("bfloat16", torch.bfloat16, 1.6e-2)
    print("\033[92mvcmp(x,x,'ne') IEEE-unordered NaN detection verified on NPU\033[0m")
