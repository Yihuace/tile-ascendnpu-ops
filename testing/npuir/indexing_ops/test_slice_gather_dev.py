# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from testcommon import assert_close, gen_tensor

pytestmark = [
    pytest.mark.op("gather"),
    pytest.mark.mode("Developer"),
]

GATHER_CASES = [(32, 32)]
DTYPES = ["float16"]


def vec_gather(block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 1
    itype = "int32"

    @T.prim_func
    def sliceGatherDev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), itype),
        C: T.Tensor((block_M, block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            index_VEC = T.alloc_shared((block_M, block_N), itype)
            C_VEC = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, index_VEC)
            T.npuir_gather(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_M, :block_N],
                index_VEC[:block_M, :block_N],
            )
            T.copy(C_VEC, C)

    return sliceGatherDev


def vec_gather_partial_dst(block_M, block_N, dtype="float16"):
    BLOCK_SIZE = 1
    itype = "int32"

    @T.prim_func
    def sliceGatherPartialDstDev(
        A: T.Tensor((block_M, block_N), dtype),
        B: T.Tensor((block_M, block_N), itype),
        C: T.Tensor((block_M, 2 * block_N), dtype),
    ):
        with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
            A_VEC = T.alloc_shared((block_M, block_N), dtype)
            index_VEC = T.alloc_shared((block_M, block_N), itype)
            C_VEC = T.alloc_shared((block_M, 2 * block_N), dtype)
            T.copy(A, A_VEC)
            T.copy(B, index_VEC)
            T.copy(C, C_VEC)
            T.npuir_gather(
                A_VEC[:block_M, :block_N],
                C_VEC[:block_M, block_N : 2 * block_N],
                index_VEC[:block_M, :block_N],
            )
            T.copy(C_VEC, C)

    return sliceGatherPartialDstDev


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", GATHER_CASES)
def test_vec_gather_dev(dtype, M, N):
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), "int32", kind="randint", low=0, high=N)
    C = gen_tensor((M, N), dtype, kind="zeros")
    ref_C = torch.gather(A.cpu(), dim=-1, index=B.cpu().long())

    func = vec_gather(M, N)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M, N", GATHER_CASES)
def test_vec_gather_partial_dst_dev(dtype, M, N):
    torch.manual_seed(42)
    A = gen_tensor((M, N), dtype, kind="randn")
    B = gen_tensor((M, N), "int32", kind="randint", low=0, high=N)
    C = gen_tensor((M, 2 * N), dtype, kind="zeros")
    ref_C = torch.zeros_like(C.cpu())
    ref_C[:, N : 2 * N] = torch.gather(A.cpu(), dim=-1, index=B.cpu().long())

    func = vec_gather_partial_dst(M, N)
    compiled = tilelang.compile(func, target="npuir")
    compiled(A, B, C)

    assert_close(C.cpu(), ref_C, dtype=dtype, rtol=1e-2, atol=1e-2)
