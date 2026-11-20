from .common import (
    FULL_OUT_IDX,
    T,
    aligned_cols,
    max_multi_n_rows,
    require_tilelang,
    tilelang,
)


def build_multi_n(M: int, N: int, eps: float, dtype: str, scale: float = 1.0):
    require_tilelang()
    n_align = aligned_cols(N, dtype)
    row_factor = max_multi_n_rows(N, dtype)
    if row_factor == 1:
        from .row_one import _build_row_one

        return _build_row_one(M, N, eps, dtype, scale, auto_multibuffer=True)
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

                x_tile = T.alloc_ub((row_factor, n_align), dtype)
                residual_tile = T.alloc_ub((row_factor, n_align), dtype)
                sum_tile = T.alloc_ub((row_factor, n_align), dtype)
                weight_tile = T.alloc_ub((1, n_align), dtype)
                x_fp32 = T.alloc_ub((row_factor, n_align), "float32")
                sq_f32 = T.alloc_ub((row_factor, n_align), "float32")
                rstd_block = T.alloc_ub((row_factor, 1), "float32")
                rstd_tile = T.alloc_ub((row_factor,), "float32")

                T.copy(weight[0:N], weight_tile[0, 0:N])

                for row_outer in T.Pipelined(
                    T.ceildiv(row_work, row_factor), num_stages=2
                ):
                    local_row = row_outer * row_factor
                    row_size = T.min(row_factor, row_work - local_row)
                    row_start = offset_m + local_row

                    if row_size < row_factor:
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

                    T.vcast(x_tile, x_fp32)
                    if has_scale:
                        T.vmul(x_fp32, tmp_scale, x_fp32)
                    T.vcast(residual_tile, sq_f32)
                    T.vadd(x_fp32, sq_f32, x_fp32)
                    T.vcast(x_fp32, sum_tile, round_mode="rint")
                    T.copy(
                        sum_tile[0:row_size, 0:N],
                        residual_out[row_start : row_start + row_size, 0:N],
                    )
                    T.vmul(x_fp32, x_fp32, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)

                    T.reduce(
                        sq_f32,
                        rstd_block,
                        dims=1,
                        reduce_mode="sum",
                        size=[row_factor, N],
                    )
                    T.vadd(rstd_block, tmp_eps, rstd_block)
                    T.vrsqrt(rstd_block, rstd_block)

                    for i in T.Parallel(row_factor):
                        if i < row_size:
                            rstd_tile[i] = rstd_block[i, 0]
                    T.copy(
                        rstd_tile[0:row_size], rstd[row_start : row_start + row_size]
                    )

                    T.vmul(x_fp32, rstd_block, x_fp32)
                    T.vcast(x_fp32, sum_tile)
                    T.vmul(sum_tile, weight_tile, sum_tile)

                    T.copy(
                        sum_tile[0:row_size, 0:N],
                        out[row_start : row_start + row_size, 0:N],
                    )

        return main

    return _kernel
