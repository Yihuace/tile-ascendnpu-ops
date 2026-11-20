import os
from typing import Tuple

if __package__:
    from .kernels import common as _common
else:
    from kernels import common as _common

_FP32_SPLIT_D_MIN_N = 10240
_LOW_PRECISION_SPLIT_D_MIN_N = 12288
_ROW_ONE_MIN_M = 256
_SPLIT_ALIGNMENT = 2048
_BF16_BALANCED_N = 4096
_MERGE_N_MAX_COLS = 2000


def split_d_ub_bytes(block_m: int, block_n: int, dtype: str) -> int:
    if block_m <= 0 or block_n <= 0:
        return _common.DEFAULT_ASCENDC_UB_BYTES + 1
    dtype_bytes = _common.dtype_nbytes(dtype)
    fp32_tile_cols = 1 if dtype == "float32" else block_n
    weight_f32_cols = block_n if dtype == "bfloat16" else 1
    allocations = [
        _common.ub_align_bytes(block_m * block_n * dtype_bytes) for _ in range(3)
    ]
    allocations += [_common.ub_align_bytes(block_n * dtype_bytes)]
    allocations += [_common.ub_align_bytes(block_m * fp32_tile_cols * dtype_bytes)]
    allocations += [_common.ub_align_bytes(block_m * fp32_tile_cols * 4)]
    allocations += [_common.ub_align_bytes(block_m * block_n * 4)]
    allocations += [_common.ub_align_bytes(weight_f32_cols * 4)]
    allocations += [_common.ub_align_bytes(block_m * 4) for _ in range(2)]
    return (
        _common.ASCENDC_UB_USED
        + _common.TILELANG_LOWERING_RESERVE_BYTES
        + sum(allocations)
    )


def aiv_core_count() -> int:
    env_value = os.environ.get("TILELANG_ASCEND_AIV_CORES")
    if env_value:
        return max(1, int(env_value))
    try:
        from tilelang.utils import NPUUtils

        return max(1, int(NPUUtils.get().get_aivector_core_num()))
    except Exception:
        return _common.DEFAULT_ASCENDC_AIV_CORES


def rows_per_core(num_rows: int) -> int:
    return _common.ceildiv(num_rows, aiv_core_count())


def _should_multibuffer_normal(dtype: str, block_m: int, row_factor: int) -> bool:
    if row_factor < 2:
        return False
    iterations = _common.ceildiv(block_m, row_factor)
    if dtype == "bfloat16":
        return iterations >= 3
    return iterations != 2


def _select_split_d(
    M: int,
    N: int,
    dtype: str,
    ub_budget: int = _common.DEFAULT_ASCENDC_UB_BYTES,
) -> Tuple[str, int, int]:
    def fits(bm: int, bn: int) -> bool:
        return (
            bm > 0
            and bn > 0
            and bm <= M
            and bn <= N
            and split_d_ub_bytes(bm, bn, dtype) <= ub_budget
        )

    if dtype in {"bfloat16", "float32"}:
        balanced_bm = rows_per_core(M)
        if balanced_bm < 32:
            balanced_bns = (
                (8192, 4096, 2048, 1024, 512, 256, 128, 64)
                if dtype == "bfloat16"
                else (2048, 1024, 512, 256, 128, 64)
            )
            for balanced_bn in balanced_bns:
                if fits(balanced_bm, balanced_bn):
                    return "split_d", balanced_bm, balanced_bn

    if dtype == "float32":
        preferred = [(4, 1024), (16, 256), (8, 512), (32, 128), (2, 2048), (1, 4096)]
        fallback_bn = [2048, 1024, 512, 256, 128, 64]
    elif N % 2048 == 0:
        preferred = [
            (4, 2048),
            (16, 512),
            (8, 1024),
            (32, 256),
            (64, 128),
            (2, 4096),
            (1, 8192),
        ]
        fallback_bn = [4096, 2048, 1024, 512, 256, 128, 64]
    else:
        preferred = [
            (16, 512),
            (32, 256),
            (8, 1024),
            (64, 128),
            (4, 2048),
            (2, 4096),
            (1, 8192),
        ]
        fallback_bn = [4096, 2048, 1024, 512, 256, 128, 64]

    fallback_bm = [32, 16, 8, 4, 2, 1]
    candidates = preferred + [(bm, bn) for bm in fallback_bm for bn in fallback_bn]
    for bm, bn in candidates:
        if fits(bm, bn):
            return "split_d", bm, bn
    fallback_bn = min(N, 64)
    if fits(1, fallback_bn):
        return "split_d", 1, fallback_bn
    return "split_d", 1, max(1, min(N, _common.dtype_align_elems(dtype)))


def _select_row_group(
    M: int,
    N: int,
    dtype: str = "float16",
    ub_budget: int = _common.DEFAULT_ASCENDC_UB_BYTES,
) -> Tuple[str, int, int]:
    core_rows = rows_per_core(M)
    row_factor = min(core_rows, _common.ASCENDC_SPLIT_D_ROW_FACTOR)
    max_cols = _common.max_row_stream_cols(
        N,
        dtype,
        ub_budget,
        row_factor=row_factor,
    )
    if max_cols <= 0:
        return _select_split_d(M, N, dtype, ub_budget=ub_budget)
    target_cols = min(max_cols, _common.balanced_split_cols(N))
    segment_count = _common.ceildiv(N, target_cols)
    block_n = _common.align_up(_common.ceildiv(N, segment_count), 16)
    if _common.row_group_ub_bytes(row_factor, block_n) > ub_budget:
        return _select_split_d(M, N, dtype, ub_budget=ub_budget)
    return "split_d_row_group", core_rows, block_n


def _select_auto(
    M: int,
    N: int,
    dtype: str,
    ub_budget: int = _common.DEFAULT_ASCENDC_UB_BYTES,
) -> Tuple[str, int, int]:
    core_count = aiv_core_count()
    block_factor = _common.ceildiv(M, core_count)
    split_d_min_n = (
        _FP32_SPLIT_D_MIN_N if dtype == "float32" else _LOW_PRECISION_SPLIT_D_MIN_N
    )
    if split_d_min_n < N:
        if dtype == "float16" and N % _SPLIT_ALIGNMENT != 0:
            return _select_row_group(M, N, dtype, ub_budget)
        if dtype == "bfloat16" and M >= _ROW_ONE_MIN_M and N % _SPLIT_ALIGNMENT != 0:
            return _select_row_group(M, N, dtype, ub_budget)
        return _select_split_d(M, N, dtype, ub_budget=ub_budget)
    if core_count >= M and _common.single_n_ub_bytes(N, dtype) <= ub_budget:
        return "single_n", 1, N
    if (
        _common.align_up(N, _common.dtype_align_elems(dtype)) <= _MERGE_N_MAX_COLS
        and _common.max_merge_n_rows(N, dtype, ub_budget) > 0
    ):
        return "merge_n", block_factor, N
    if _common.max_multi_n_rows(N, dtype, ub_budget) > 0:
        return "multi_n", block_factor, N

    normal_row_factor = _common.max_normal_rows(N, dtype, ub_budget)
    if (
        M >= _ROW_ONE_MIN_M
        and (normal_row_factor <= 1 or (dtype == "bfloat16" and N == _BF16_BALANCED_N))
        and _common.row_one_mb_fits(N, dtype, ub_budget)
    ):
        return "normal_row_one_auto_multibuffer", block_factor, N
    if normal_row_factor > 0:
        if _should_multibuffer_normal(
            dtype,
            block_factor,
            _common.max_normal_mb_rows(N, dtype, ub_budget),
        ):
            return "normal_auto_multibuffer", block_factor, N
        if dtype == "bfloat16" and N == _BF16_BALANCED_N:
            return "normal", min(2, M), N
        return "normal", block_factor, N

    return _select_split_d(M, N, dtype, ub_budget=ub_budget)


def select_config(
    M: int,
    N: int,
    dtype: str,
    ub_budget: int = _common.DEFAULT_ASCENDC_UB_BYTES,
) -> Tuple[str, int, int]:
    return _select_auto(M, N, dtype, ub_budget)
