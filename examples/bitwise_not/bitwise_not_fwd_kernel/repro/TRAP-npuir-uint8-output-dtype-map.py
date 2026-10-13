# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-npuir-uint8-output-dtype-map

Phenomenon (toolchain trap, kind: trap):
    ``JitKernel_NPU._tvm_dtype_to_torch`` (tilelang/jit/jit_npu.py) maps
    float16/float32/float64/bfloat16/int8/int16/int32/int64/bool to torch
    dtypes but has NO ``uint8`` entry -- it falls back to ``torch.float32``.
    A kernel that declares a ``uint8`` tensor therefore gets its auto-allocated
    (``out_idx``) output as a *float32* tensor, and the returned values are
    garbage (the byte buffer is re-interpreted as fp32).

Workaround (asserted below):
    Carry ``uint8`` through ``int8`` byte storage (identical bit pattern for
    ``vnot``) and view back on the host:
        x_uint8.view(torch.int8) -> kernel(int8) -> out.view(torch.uint8)
    bit-exact vs ``torch.bitwise_not``.

First verified: 2026-10-13, tilelang 0.1.2 (dev root build) + Ascend 910B2C
    (``NPUUtils.get().get_aicore_num()`` = 20) + CANN 8.5.2.
origin_task: examples/bitwise_not/bitwise_not_fwd_kernel (Stage 3 first_impl).

Re-verify records (append):
    2026-10-13 first run: uint8 output dtype is float32 (garbage); int8
    byte-storage form bit-exact.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N = 4096
BS = 8192


@tilelang.jit(out_idx=[1], target="npuir")
def _not_kernel(storage_dtype):
    @T.prim_func
    def main(
        x: T.Tensor((N,), storage_dtype),
        y: T.Tensor((N,), storage_dtype),
        shape: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((BS,), storage_dtype)
            y_ub = T.alloc_ub((BS,), storage_dtype)
            T.copy(x[0:shape], x_ub[0:shape])
            T.vnot(x_ub, y_ub)
            T.copy(y_ub[0:shape], y[0:shape])

    return main


def main():
    xu = torch.tensor([0, 1, 2, 7, 127, 128, 254, 255] * (N // 8), dtype=torch.uint8)
    ref = torch.bitwise_not(xu)

    # uint8-declared auto output is mis-allocated as float32 (runtime gap).
    y_bad = _not_kernel("uint8")(xu.npu(), N)
    print(f"[uint8 storage] output dtype = {y_bad.dtype} (expected corruption)")
    assert y_bad.dtype == torch.float32, "uint8 output dtype map unexpectedly fixed"

    # int8 byte storage works and is bit-exact.
    y_ok = _not_kernel("int8")(xu.view(torch.int8).npu(), N).cpu().view(torch.uint8)
    assert torch.equal(y_ok, ref), "int8 byte storage vnot not bit-exact"
    print("[int8 storage]  bit-exact PASS")


if __name__ == "__main__":
    main()
