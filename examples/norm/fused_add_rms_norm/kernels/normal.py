from .common import (
    FULL_OUT_IDX,
    T,
    aligned_cols,
    max_normal_mb_rows,
    max_normal_rows,
    require_tilelang,
    tilelang,
)


def _build_normal(
    M: int,
    N: int,
    eps: float,
    dtype: str,
    scale: float,
    *,
    auto_multibuffer: bool,
):
    require_tilelang()
    n_align = aligned_cols(N, dtype)
    row_factor = (
        max_normal_mb_rows(N, dtype) if auto_multibuffer else max_normal_rows(N, dtype)
    )
    if not auto_multibuffer and row_factor == 1:
        from .row_one import _build_row_one

        return _build_row_one(M, N, eps, dtype, scale, auto_multibuffer=False)
    is_aligned = n_align == N
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0
    pipeline_stages = 2 if auto_multibuffer else 1

    @tilelang.jit(
        out_idx=FULL_OUT_IDX,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: auto_multibuffer,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
    def _kernel(block_m, block_n):
        tile_rows = min(block_m, row_factor)

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
                weight_tile = T.alloc_ub((1, n_align), dtype)
                x_fp32 = T.alloc_ub((tile_rows, n_align), "float32")
                sq_f32 = T.alloc_ub((tile_rows, n_align), "float32")
                gamma_fp32 = T.alloc_ub((1, n_align), "float32")
                rstd_block = T.alloc_ub((tile_rows, 1), "float32")
                rstd_tile = T.alloc_ub((tile_rows,), "float32")

                if not is_aligned:
                    T.clear(weight_tile)
                T.copy(weight[0:N], weight_tile[0, 0:N])

                for row_outer in T.Pipelined(
                    T.ceildiv(row_work, tile_rows), num_stages=pipeline_stages
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

                    if dtype == "float32":
                        if has_scale:
                            T.vmul(x_tile, tmp_scale, x_tile)
                        T.vadd(x_tile, residual_tile, x_tile)
                    elif dtype == "float16":
                        T.vcast(x_tile, x_fp32)
                        if has_scale:
                            T.vmul(x_fp32, tmp_scale, x_fp32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(x_fp32, sq_f32, x_fp32)
                        T.vcast(x_fp32, x_tile, round_mode="rint")
                    else:
                        T.vcast(x_tile, x_fp32)
                        if has_scale:
                            T.vmul(x_fp32, tmp_scale, x_fp32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(x_fp32, sq_f32, x_fp32)
                        T.vcast(x_fp32, x_tile, round_mode="rint")

                    if is_aligned and row_size == tile_rows:
                        T.copy(x_tile, residual_out[row_start, 0])
                    else:
                        T.copy(
                            x_tile[0:row_size, 0:N],
                            residual_out[row_start : row_start + row_size, 0:N],
                        )
                    if dtype == "float32":
                        T.vmul(x_tile, x_tile, sq_f32)
                    else:
                        T.vmul(x_fp32, x_fp32, sq_f32)
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

                    if dtype == "float32":
                        T.vmul(x_tile, rstd_block, x_tile)
                        T.vmul(x_tile, weight_tile, x_tile)
                    elif dtype == "float16":
                        T.vmul(x_fp32, rstd_block, x_fp32)
                        T.vcast(x_fp32, x_tile)
                        T.vmul(x_tile, weight_tile, x_tile)
                    else:
                        T.vmul(x_fp32, rstd_block, x_fp32)
                        T.vcast(x_fp32, x_tile, round_mode="rint")
                        T.vcast(x_tile, x_fp32)
                        T.vcast(weight_tile, gamma_fp32)
                        T.vmul(x_fp32, gamma_fp32, x_fp32)
                        T.vcast(x_fp32, x_tile, round_mode="rint")

                    if is_aligned and row_size == tile_rows:
                        T.copy(x_tile, out[row_start, 0])
                    else:
                        T.copy(
                            x_tile[0:row_size, 0:N],
                            out[row_start : row_start + row_size, 0:N],
                        )

        return main

    return _kernel


def build_normal(M: int, N: int, eps: float, dtype: str, scale: float = 1.0):
    return _build_normal(M, N, eps, dtype, scale, auto_multibuffer=False)


def build_normal_multibuffer(
    M: int, N: int, eps: float, dtype: str, scale: float = 1.0
):
    return _build_normal(M, N, eps, dtype, scale, auto_multibuffer=True)
