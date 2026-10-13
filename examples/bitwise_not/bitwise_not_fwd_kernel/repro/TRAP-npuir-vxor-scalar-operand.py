# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-npuir-vxor-scalar-operand

Phenomenon (toolchain trap, kind: trap):
    ``hivm.hir.vxor`` requires BOTH source operands to be shaped tensors.
    Passing a Python scalar as ``src1`` (``T.vxor(x_ub, 1, y_ub)``) lowers to
    an ``i8`` scalar operand and fails MLIR verification:

        error: 'hivm.hir.vxor' op operand #1 must be variadic of shaped of any
        type values, but got 'i8'

    (``docs/Tilelang.language/逻辑操作/T.vxor.md`` section 2.1 states a Python
    scalar int is promoted and broadcast; that claim is NOT honored by the
    codegen for the XOR op on this toolchain.)

Workaround (asserted below):
    Materialize a shaped constant tile in UB and XOR against it:
        T.fill(c_ub, T.cast(1, "int8"))
        T.vxor(x_ub, c_ub, y_ub)
    This is bit-exact vs ``torch.bitwise_not`` for bool byte values {0,1}.

First verified: 2026-10-13, tilelang 0.1.2 (dev root build) + Ascend 910B2C
    (``NPUUtils.get().get_aicore_num()`` = 20) + CANN 8.5.2.
origin_task: examples/bitwise_not/bitwise_not_fwd_kernel (Stage 3 first_impl).

Re-verify records (append):
    2026-10-13 first run: scalar form raises; shaped form bit-exact.
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
def _xor_kernel(scalar_operand):
    @T.prim_func
    def main(x: T.Tensor((N,), "int8"), y: T.Tensor((N,), "int8"), shape: T.int32):
        with T.Kernel(1, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((BS,), "int8")
            y_ub = T.alloc_ub((BS,), "int8")
            c_ub = T.alloc_ub((BS,), "int8")
            T.copy(x[0:shape], x_ub[0:shape])
            if scalar_operand:
                T.vxor(x_ub, 1, y_ub)
            else:
                T.fill(c_ub, T.cast(1, "int8"))
                T.vxor(x_ub, c_ub, y_ub)
            T.copy(y_ub[0:shape], y[0:shape])

    return main


def main():
    xb = torch.randint(0, 2, (N,), dtype=torch.bool)
    ref = torch.bitwise_not(xb)

    # Scalar src1 is not lowerable on this toolchain.
    raised = False
    try:
        _xor_kernel(True)(xb.view(torch.int8).npu(), N)
    except Exception as e:  # noqa: BLE001
        raised = True
        print(f"[scalar form] rejected as expected: {type(e).__name__}")
    assert raised, "scalar src1 unexpectedly compiled"

    # Shaped constant tile works and is bit-exact.
    y = _xor_kernel(False)(xb.view(torch.int8).npu(), N).cpu().view(torch.bool)
    assert torch.equal(y, ref), "shaped-constant vxor not bit-exact"
    print("[shaped form] bit-exact PASS")


if __name__ == "__main__":
    main()
