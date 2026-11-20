# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("flip"),
    pytest.mark.mode("Developer"),
]

DTYPES = ["float16"]


def vec_flip(block_M, block_N, flip_axis, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceFlipDev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A, A_VEC)
            T.npuir_flip(
                A_VEC[:block_M, :block_N],
                B_VEC[:block_M, :block_N],
                flip_axis,
            )
            T.copy(B_VEC, B)

    return sliceFlipDev


def vec_flip_partial_dst(block_M, block_N, flip_axis, dtype="float16"):
    BLOCK_SIZE = 1

    @T.prim_func
    def sliceFlipPartialDstDev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, 2 * block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            B_VEC = T.alloc_shared((block_M, 2 * block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, B_VEC)
            T.npuir_flip(
                A_VEC[:block_M, :block_N],
                B_VEC[:block_M, block_N : 2 * block_N],
                flip_axis,
            )
            T.copy(B_VEC, B)

    return sliceFlipPartialDstDev


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_flip_dev(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), dtype, kind="zeros")
    ref_B = torch.flip(A.cpu(), [1])

    func = vec_flip(32, 32, flip_axis=1)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B)

    assert_close(B.cpu(), ref_B, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vec_flip_partial_dst_dev(dtype):
    M, N = 32, 32
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, 2 * N), dtype, kind="zeros")
    ref_B = torch.zeros_like(B.cpu())
    ref_B[:, N : 2 * N] = torch.flip(A.cpu(), [1])

    func = vec_flip_partial_dst(32, 32, flip_axis=1)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B)

    assert_close(B.cpu(), ref_B, dtype=dtype, rtol=1e-2, atol=1e-2)
