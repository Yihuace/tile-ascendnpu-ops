from .common import (
    FULL_OUT_IDX,
    T,
    aligned_cols,
    max_merge_n_db_rows,
    max_merge_n_mb_rows,
    max_merge_n_rows,
    require_tilelang,
    tilelang,
)


def _build_merge_n_fp32_fallback(M: int, N: int, eps: float, scale: float):
    dtype = "float32"
    n_align = aligned_cols(N, dtype)
    row_factor = max_merge_n_rows(N, dtype)
    db_row_factor = max_merge_n_db_rows(N, dtype)
    is_aligned = n_align == N
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0

    @tilelang.jit(
        out_idx=FULL_OUT_IDX,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
    def _kernel(block_m, block_n):
        tile_rows = min(block_m, row_factor, db_row_factor)

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            residual: T.Tensor((M, N), dtype),
            weight: T.Tensor((N,), dtype),
            out: T.Tensor((M, N), dtype),
            rstd: T.Tensor((M,), "float32"),
            residual_out: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                offset_m = pid_m * block_m
                row_work = T.min(block_m, M - offset_m)

                x_tile = T.alloc_ub((tile_rows, n_align), dtype)
                residual_tile = T.alloc_ub((tile_rows, n_align), dtype)
                sum_tile = T.alloc_ub((tile_rows, n_align), dtype)
                weight_tile = T.alloc_ub((1, n_align), dtype)
                sq_f32 = T.alloc_ub((tile_rows, n_align), "float32")
                rstd_block = T.alloc_ub((tile_rows, 1), "float32")
                rstd_tile = T.alloc_ub((tile_rows,), "float32")

                if not is_aligned:
                    T.clear(weight_tile)
                T.copy(weight[0:N], weight_tile[0, 0:N])

                for row_outer in T.Pipelined(
                    T.ceildiv(row_work, tile_rows), num_stages=2
                ):
                    local_row = row_outer * tile_rows
                    row_size = T.min(tile_rows, row_work - local_row)
                    row_start = offset_m + local_row

                    if is_aligned and row_size == tile_rows:
                        T.copy(x[row_start, 0], x_tile)
                        T.copy(residual[row_start, 0], residual_tile)
                    else:
                        T.clear(x_tile)
                        T.clear(residual_tile)
                        T.copy(
                            x[row_start : row_start + row_size, 0:N],
                            x_tile[0:row_size, 0:N],
                        )
                        T.copy(
                            residual[row_start : row_start + row_size, 0:N],
                            residual_tile[0:row_size, 0:N],
                        )

                    if has_scale:
                        T.vmul(x_tile, tmp_scale, x_tile)
                    T.vadd(x_tile, residual_tile, sum_tile)
                    if is_aligned and row_size == tile_rows:
                        T.copy(sum_tile, residual_out[row_start, 0])
                    else:
                        T.copy(
                            sum_tile[0:row_size, 0:N],
                            residual_out[row_start : row_start + row_size, 0:N],
                        )
                    T.vmul(sum_tile, sum_tile, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)
                    T.reduce(
                        sq_f32,
                        rstd_block,
                        dims=1,
                        reduce_mode="sum",
                        size=[tile_rows, n_align],
                    )
                    T.vadd(rstd_block, tmp_eps, rstd_block)
                    T.vrsqrt(rstd_block, rstd_block)

                    for i in T.Parallel(tile_rows):
                        if i < row_size:
                            rstd_tile[i] = rstd_block[i, 0]
                    T.copy(
                        rstd_tile[0:row_size], rstd[row_start : row_start + row_size]
                    )

                    T.vmul(sum_tile, rstd_block, sum_tile)
                    T.vmul(sum_tile, weight_tile, sum_tile)
                    if is_aligned and row_size == tile_rows:
                        T.copy(sum_tile, out[row_start, 0])
                    else:
                        T.copy(
                            sum_tile[0:row_size, 0:N],
                            out[row_start : row_start + row_size, 0:N],
                        )

        return main

    return _kernel


def build_merge_n(M: int, N: int, eps: float, dtype: str, scale: float = 1.0):
    require_tilelang()
    n_align = aligned_cols(N, dtype)
    row_factor = max_merge_n_rows(N, dtype)
    tile_limit = max_merge_n_mb_rows(N, dtype)
    if dtype == "float32" and tile_limit < 2:
        return _build_merge_n_fp32_fallback(M, N, eps, scale)
    is_aligned = n_align == N
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0

    @tilelang.jit(
        out_idx=FULL_OUT_IDX,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
    def _kernel(block_m, block_n):
        tile_rows = min(block_m, row_factor, tile_limit)
        weight_f32_cols = n_align if dtype != "float16" else 1

        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            residual: T.Tensor((M, N), dtype),
            weight: T.Tensor((N,), dtype),
            out: T.Tensor((M, N), dtype),
            rstd: T.Tensor((M,), "float32"),
            residual_out: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                offset_m = pid_m * block_m
                row_work = T.min(block_m, M - offset_m)

                x_tile = T.alloc_ub((tile_rows, n_align), dtype)
                residual_tile = T.alloc_ub((tile_rows, n_align), dtype)
                residual_sum_tile = T.alloc_ub((tile_rows, n_align), dtype)
                out_tile = T.alloc_ub((tile_rows, n_align), dtype)
                weight_tile = T.alloc_ub((1, n_align), dtype)
                sum_f32 = T.alloc_ub((tile_rows, n_align), "float32")
                sq_f32 = T.alloc_ub((tile_rows, n_align), "float32")
                rstd_block = T.alloc_ub((tile_rows, 1), "float32")
                weight_f32 = T.alloc_ub((1, weight_f32_cols), "float32")
                T.copy(weight[0:N], weight_tile[0, 0:N])
                if dtype == "bfloat16":
                    T.vcast(weight_tile, weight_f32)

                for row_outer in T.Pipelined(
                    T.ceildiv(row_work, tile_rows), num_stages=2
                ):
                    local_row = row_outer * tile_rows
                    row_size = T.min(tile_rows, row_work - local_row)
                    row_start = offset_m + local_row

                    if is_aligned and row_size == tile_rows:
                        T.copy(x[row_start, 0], x_tile)
                        T.copy(residual[row_start, 0], residual_tile)
                    else:
                        T.clear(x_tile)
                        T.clear(residual_tile)
                        T.copy(
                            x[row_start : row_start + row_size, 0:N],
                            x_tile[0:row_size, 0:N],
                        )
                        T.copy(
                            residual[row_start : row_start + row_size, 0:N],
                            residual_tile[0:row_size, 0:N],
                        )

                    if dtype == "float16":
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, residual_sum_tile, round_mode="rint")
                    elif dtype == "float32":
                        if has_scale:
                            T.vmul(x_tile, tmp_scale, x_tile)
                        T.vadd(x_tile, residual_tile, residual_sum_tile)
                    else:
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, residual_sum_tile, round_mode="rint")

                    if dtype != "float32":
                        if is_aligned and row_size == tile_rows:
                            T.copy(residual_sum_tile, residual_out[row_start, 0])
                        else:
                            T.copy(
                                residual_sum_tile[0:row_size, 0:N],
                                residual_out[row_start : row_start + row_size, 0:N],
                            )

                    if dtype == "float32":
                        T.vmul(residual_sum_tile, residual_sum_tile, sq_f32)
                    else:
                        T.vmul(sum_f32, sum_f32, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)
                    T.reduce(
                        sq_f32,
                        rstd_block,
                        dims=1,
                        reduce_mode="sum",
                        size=[tile_rows, n_align],
                    )
                    T.vadd(rstd_block, tmp_eps, rstd_block)
                    T.vrsqrt(rstd_block, rstd_block)

                    if dtype == "float32":
                        T.vmul(residual_sum_tile, rstd_block, out_tile)
                        if is_aligned and row_size == tile_rows:
                            T.copy(residual_sum_tile, residual_out[row_start, 0])
                        else:
                            T.copy(
                                residual_sum_tile[0:row_size, 0:N],
                                residual_out[row_start : row_start + row_size, 0:N],
                            )
                        T.copy(
                            rstd_block[0:row_size, 0],
                            rstd[row_start : row_start + row_size],
                        )
                        T.vmul(out_tile, weight_tile, out_tile)
                    else:
                        T.copy(
                            rstd_block[0:row_size, 0],
                            rstd[row_start : row_start + row_size],
                        )
                        if dtype == "float16":
                            T.vmul(sum_f32, rstd_block, sum_f32)
                            T.vcast(sum_f32, out_tile)
                            T.vmul(out_tile, weight_tile, out_tile)
                        else:
                            T.vmul(sum_f32, rstd_block, sum_f32)
                            T.vcast(sum_f32, out_tile, round_mode="rint")
                            T.vcast(out_tile, sum_f32)
                            T.vmul(sum_f32, weight_f32, sum_f32)
                            T.vcast(sum_f32, out_tile, round_mode="rint")

                    if is_aligned and row_size == tile_rows:
                        T.copy(out_tile, out[row_start, 0])
                    else:
                        T.copy(
                            out_tile[0:row_size, 0:N],
                            out[row_start : row_start + row_size, 0:N],
                        )

        return main

    return _kernel
