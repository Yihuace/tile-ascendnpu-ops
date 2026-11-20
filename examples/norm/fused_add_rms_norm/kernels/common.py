try:
    import tilelang
    import tilelang.language as T
except Exception as exc:
    tilelang = None
    T = None
    _TILELANG_IMPORT_ERROR = exc
else:
    _TILELANG_IMPORT_ERROR = None

FULL_OUT_IDX = [-3, -2, -1]
DEFAULT_ASCENDC_UB_BYTES = 192 * 1024
DEFAULT_ASCENDC_AIV_CORES = 48
ASCENDC_UB_USED = 1024
ASCENDC_BLOCK_ALIGN_NUM = 16
ASCENDC_NORMAL_ROW_FACTOR = 64
ASCENDC_SPLIT_D_ROW_FACTOR = 64
MULTIBUFFER_STAGES = 2
TILELANG_LOWERING_RESERVE_BYTES = 16 * 1024
ASCENDC_SPLIT_D_MAX_COLS = 12096


def require_tilelang() -> None:
    if tilelang is None or T is None:
        raise RuntimeError(
            "TileLang import failed; fix the TileLang environment before running this kernel"
        ) from (_TILELANG_IMPORT_ERROR)


def align_up(value: int, factor: int) -> int:
    return ((value + factor - 1) // factor) * factor


def dtype_align_elems(dtype: str) -> int:
    return 8 if dtype == "float32" else 16


def aligned_cols(n: int, dtype: str) -> int:
    return align_up(n, dtype_align_elems(dtype))


def dtype_nbytes(dtype: str) -> int:
    if dtype in {"float16", "bfloat16"}:
        return 2
    if dtype == "float32":
        return 4
    raise ValueError(f"Unsupported dtype for AddRmsNorm kernel: {dtype}")


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def ub_align_bytes(size: int) -> int:
    return align_up(int(size), 32)


def single_n_ub_bytes(n: int, dtype: str) -> int:
    n_align = aligned_cols(n, dtype)
    if dtype == "float16":
        allocations = [
            ub_align_bytes(n_align * 2),
            ub_align_bytes(n_align * 2),
            ub_align_bytes(n_align * 4),
            ub_align_bytes(n_align * 4),
            ub_align_bytes(4),
        ]
    else:
        dtype_bytes = dtype_nbytes(dtype)
        allocations = [ub_align_bytes(n_align * dtype_bytes) for _ in range(5)]
        allocations += [ub_align_bytes(n_align * 4) for _ in range(3)]
        allocations += [ub_align_bytes(4), ub_align_bytes(4)]
    return ASCENDC_UB_USED + TILELANG_LOWERING_RESERVE_BYTES + sum(allocations)


def merge_n_db_ub_bytes(rows: int, n: int, dtype: str, stages: int = 2) -> int:
    if rows <= 0:
        return DEFAULT_ASCENDC_UB_BYTES + 1
    n_align = aligned_cols(n, dtype)
    dtype_bytes = dtype_nbytes(dtype)
    weight_f32_cols = 1 if dtype == "float16" else n_align
    static_bytes = ub_align_bytes(n_align * dtype_bytes) + ub_align_bytes(
        weight_f32_cols * 4
    )
    stage_bytes = (
        3 * ub_align_bytes(rows * n_align * dtype_bytes)
        + 2 * ub_align_bytes(rows * n_align * 4)
        + 2 * ub_align_bytes(rows * 4)
    )
    return (
        ASCENDC_UB_USED
        + TILELANG_LOWERING_RESERVE_BYTES
        + static_bytes
        + stages * stage_bytes
    )


def max_row_stream_cols(
    n: int,
    dtype: str,
    ub_budget: int = DEFAULT_ASCENDC_UB_BYTES,
    *,
    row_factor: int = 1,
) -> int:
    if dtype not in {"float16", "bfloat16"}:
        return 0
    row_scalar_bytes = ub_align_bytes(max(1, int(row_factor)) * 4)
    available = (
        int(ub_budget)
        - ASCENDC_UB_USED
        - TILELANG_LOWERING_RESERVE_BYTES
        - 2 * row_scalar_bytes
    )
    max_cols = min(available // (3 * 2 + 2 * 4), ASCENDC_SPLIT_D_MAX_COLS, int(n))
    return (max_cols // ASCENDC_BLOCK_ALIGN_NUM) * ASCENDC_BLOCK_ALIGN_NUM


def row_group_ub_bytes(row_factor: int, block_n: int) -> int:
    row_scalar_bytes = ub_align_bytes(max(1, int(row_factor)) * 4)
    wide_bytes = 3 * int(block_n) * 2 + 2 * int(block_n) * 4
    return (
        ASCENDC_UB_USED
        + TILELANG_LOWERING_RESERVE_BYTES
        + wide_bytes
        + 2 * row_scalar_bytes
    )


def balanced_split_cols(n: int) -> int:
    if n <= ASCENDC_SPLIT_D_MAX_COLS:
        return align_up(n, ASCENDC_BLOCK_ALIGN_NUM)
    tile_count = ceildiv(n, ASCENDC_SPLIT_D_MAX_COLS)
    return align_up(ceildiv(n, tile_count), ASCENDC_BLOCK_ALIGN_NUM)


def max_merge_n_rows(
    n: int, dtype: str, ub_budget: int = DEFAULT_ASCENDC_UB_BYTES
) -> int:
    n_align = aligned_cols(n, dtype)
    weight = 24 if dtype == "float32" else 18
    ub_size = int(ub_budget) - ASCENDC_UB_USED
    denominator = n_align * weight + 260
    if denominator <= 0 or ub_size <= 0:
        return 0
    return ub_size // denominator


def max_merge_n_db_rows(
    n: int,
    dtype: str,
    ub_budget: int = DEFAULT_ASCENDC_UB_BYTES,
    stages: int = MULTIBUFFER_STAGES,
) -> int:
    n_align = aligned_cols(n, dtype)
    row_bytes = 3 * n_align * dtype_nbytes(dtype) + 2 * n_align * 4 + 2 * 4
    available = int(ub_budget) - ASCENDC_UB_USED - TILELANG_LOWERING_RESERVE_BYTES
    if row_bytes <= 0 or available <= 0:
        return 0
    rows = available // (int(stages) * row_bytes)
    while rows > 0 and merge_n_db_ub_bytes(rows, n, dtype, stages) > ub_budget:
        rows -= 1
    return rows


def merge_n_mb_ub_bytes(rows: int, n: int, dtype: str, stages: int = 2) -> int:
    if rows <= 0:
        return DEFAULT_ASCENDC_UB_BYTES + 1
    n_align = aligned_cols(n, dtype)
    dtype_bytes = dtype_nbytes(dtype)
    weight_f32_cols = 1 if dtype == "float16" else n_align
    static_bytes = ub_align_bytes(n_align * dtype_bytes) + ub_align_bytes(
        weight_f32_cols * 4
    )
    stage_bytes = (
        4 * ub_align_bytes(rows * n_align * dtype_bytes)
        + 2 * ub_align_bytes(rows * n_align * 4)
        + ub_align_bytes(rows * 4)
    )
    return (
        ASCENDC_UB_USED
        + TILELANG_LOWERING_RESERVE_BYTES
        + static_bytes
        + stages * stage_bytes
    )


def max_merge_n_mb_rows(
    n: int,
    dtype: str,
    ub_budget: int = DEFAULT_ASCENDC_UB_BYTES,
    stages: int = MULTIBUFFER_STAGES,
) -> int:
    n_align = aligned_cols(n, dtype)
    row_bytes = 4 * n_align * dtype_nbytes(dtype) + 2 * n_align * 4 + 4
    available = int(ub_budget) - ASCENDC_UB_USED - TILELANG_LOWERING_RESERVE_BYTES
    if row_bytes <= 0 or available <= 0:
        return 0
    rows = available // (int(stages) * row_bytes)
    while rows > 0 and merge_n_mb_ub_bytes(rows, n, dtype, stages) > ub_budget:
        rows -= 1
    return rows


def max_normal_rows(
    n: int, dtype: str, ub_budget: int = DEFAULT_ASCENDC_UB_BYTES
) -> int:
    n_align = aligned_cols(n, dtype)
    row_bytes = 2 * n_align * dtype_nbytes(dtype) + 2 * n_align * 4 + 2 * 4
    available = int(ub_budget) - ASCENDC_UB_USED - TILELANG_LOWERING_RESERVE_BYTES
    if row_bytes <= 0 or available <= 0:
        return 0
    rows = min(ASCENDC_NORMAL_ROW_FACTOR, available // row_bytes)
    while rows > 0 and normal_ub_bytes(rows, n, dtype) > ub_budget:
        rows -= 1
    return rows


def normal_ub_bytes(rows: int, n: int, dtype: str, stages: int = 1) -> int:
    if rows <= 0:
        return DEFAULT_ASCENDC_UB_BYTES + 1
    n_align = aligned_cols(n, dtype)
    dtype_bytes = dtype_nbytes(dtype)
    static_bytes = ub_align_bytes(n_align * dtype_bytes) + ub_align_bytes(n_align * 4)
    stage_bytes = (
        2 * ub_align_bytes(rows * n_align * dtype_bytes)
        + 2 * ub_align_bytes(rows * n_align * 4)
        + 2 * ub_align_bytes(rows * 4)
    )
    fp16_add_workspace = 0
    if dtype == "float16" and int(stages) == 1:
        if rows >= 6:
            fp16_add_workspace = ASCENDC_UB_USED
        if n_align == n:
            fp16_add_workspace += ub_align_bytes(rows * n_align * dtype_bytes)
    return (
        ASCENDC_UB_USED
        + TILELANG_LOWERING_RESERVE_BYTES
        + static_bytes
        + int(stages) * stage_bytes
        + fp16_add_workspace
    )


def max_normal_mb_rows(
    n: int,
    dtype: str,
    ub_budget: int = DEFAULT_ASCENDC_UB_BYTES,
    stages: int = MULTIBUFFER_STAGES,
) -> int:
    n_align = aligned_cols(n, dtype)
    row_bytes = 2 * n_align * dtype_nbytes(dtype) + 2 * n_align * 4 + 2 * 4
    available = int(ub_budget) - ASCENDC_UB_USED - TILELANG_LOWERING_RESERVE_BYTES
    if available <= 0 or row_bytes <= 0:
        return 0
    rows = min(ASCENDC_NORMAL_ROW_FACTOR, available // (int(stages) * row_bytes))
    while rows > 0 and normal_ub_bytes(rows, n, dtype, stages) > ub_budget:
        rows -= 1
    return rows


def row_one_mb_ub_bytes(
    n: int,
    dtype: str,
    *,
    stages: int = MULTIBUFFER_STAGES,
) -> int:
    if n <= 0 or stages <= 0:
        return DEFAULT_ASCENDC_UB_BYTES + 1

    n_align = aligned_cols(n, dtype)
    if int(stages) != 2:
        return DEFAULT_ASCENDC_UB_BYTES + 1

    if dtype == "float16":
        bytes_per_col = 18
        scalar_bytes = 2 * ub_align_bytes(4)
    elif dtype == "bfloat16":
        bytes_per_col = 20 if int(n) == n_align else 22
        scalar_bytes = 2 * ub_align_bytes(4)
    elif dtype == "float32":
        bytes_per_col = 18 if int(n) == n_align else 26
        scalar_bytes = ub_align_bytes(4) if int(n) != n_align else 2 * ub_align_bytes(4)
    else:
        raise ValueError(f"Unsupported dtype for row-one UB model: {dtype}")
    return bytes_per_col * n_align + scalar_bytes


def row_one_mb_fits(
    n: int,
    dtype: str,
    ub_budget: int = DEFAULT_ASCENDC_UB_BYTES,
    *,
    stages: int = MULTIBUFFER_STAGES,
) -> bool:
    return row_one_mb_ub_bytes(
        n,
        dtype,
        stages=stages,
    ) <= int(ub_budget)


def max_multi_n_rows(
    n: int, dtype: str, ub_budget: int = DEFAULT_ASCENDC_UB_BYTES
) -> int:
    if dtype != "float16":
        return 0
    n_align = aligned_cols(n, dtype)
    if n != n_align:
        return 0
    ub_size = int(ub_budget) - ASCENDC_UB_USED
    numerator = ub_size - 256 - n_align * 2
    denominator = n_align * ASCENDC_BLOCK_ALIGN_NUM + 64
    if denominator <= 0 or numerator <= 0:
        return 0
    return numerator // denominator
