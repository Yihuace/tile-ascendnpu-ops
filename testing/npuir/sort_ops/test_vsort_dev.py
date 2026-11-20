# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("sort"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16", "float32"]


def vsort_kernel(M, N, dtype, descending, sort_axis):
    BLOCK_SIZE = 1

    @T.prim_func
    def vsortDev(
        src: T.Tensor((M, N), dtype),
        dst_value: T.Tensor((M, N), dtype),
        dst_index: T.Tensor((M, N), "int32"),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((M, N), dtype)
            val_ub = T.alloc_shared((M, N), dtype)
            idx_ub = T.alloc_shared((M, N), "int32")
            T.copy(src, src_ub)
            T.vsort(
                src_ub,
                val_ub,
                idx_ub,
                descending=descending,
                sort_axis=sort_axis,
            )
            T.copy(val_ub, dst_value)
            T.copy(idx_ub, dst_index)

    return vsortDev


def golden_sort(src, descending):
    values, indices = torch.sort(src, dim=-1, descending=descending)
    return values, indices.to(torch.int32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("descending", [False, True])
def test_vsort_dev(dtype, descending):
    M, N = 4, 1024
    sort_axis = -1
    src = gen_tensor((M, N), dtype, kind="randn")
    dst_value = gen_tensor((M, N), dtype, kind="zeros")
    dst_index = gen_tensor((M, N), "int32", kind="zeros")

    ref_value, ref_index = golden_sort(src.cpu(), descending=descending)

    func = vsort_kernel(
        M=M, N=N, dtype=dtype, descending=descending, sort_axis=sort_axis
    )
    compiled = tilelang.compile(func, target="npuir")
    compiled(src, dst_value, dst_index)

    assert_close(dst_value.cpu(), ref_value, rtol=1e-3, atol=1e-3)
    for r in range(M):
        assert sorted(dst_index[r].cpu().tolist()) == list(range(N)), (
            f"row {r} indices not a permutation of 0..{N - 1}"
        )
        assert torch.all(
            src.cpu()[r].gather(0, dst_index[r].cpu().long()) == ref_value[r].cpu()
        ), f"row {r} index-value mismatch"
