from .common import (
    ASCENDC_SPLIT_D_ROW_FACTOR,
    FULL_OUT_IDX,
    T,
    require_tilelang,
    tilelang,
)


def build_split_d_row_group(
    M: int,
    N: int,
    eps: float,
    dtype: str,
    scale: float = 1.0,
):
    require_tilelang()
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0

    @tilelang.jit(
        out_idx=FULL_OUT_IDX,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
    def _kernel(block_m, block_n):
        row_factor = min(block_m, ASCENDC_SPLIT_D_ROW_FACTOR)
        block_count = N // block_n
        tail_cols = N - block_count * block_n

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
                core_start = pid_m * block_m
                core_rows = T.min(block_m, M - core_start)
                group_count = T.ceildiv(core_rows, row_factor)

                x_tile = T.alloc_ub((1, block_n), dtype)
                residual_or_weight_tile = T.alloc_ub((1, block_n), dtype)
                sum_tile = T.alloc_ub((1, block_n), dtype)
                sum_f32 = T.alloc_ub((1, block_n), "float32")
                aux_f32 = T.alloc_ub((1, block_n), "float32")
                rstd_block = T.alloc_ub((row_factor, 1), "float32")
                partial_rows = T.alloc_ub((row_factor, 1), "float32")

                for group_id in T.serial(group_count):
                    group_start = group_id * row_factor
                    group_rows = T.min(row_factor, core_rows - group_start)
                    row_base = core_start + group_start
                    T.clear(rstd_block)

                    for no in T.serial(block_count):
                        n_start = no * block_n
                        for local_row in T.serial(row_factor):
                            if local_row < group_rows:
                                row = row_base + local_row
                                T.copy(
                                    x[row, n_start : n_start + block_n],
                                    x_tile[0, 0:block_n],
                                )
                                T.copy(
                                    residual[row, n_start : n_start + block_n],
                                    residual_or_weight_tile[0, 0:block_n],
                                )
                                if dtype == "float16":
                                    T.vcast(x_tile, sum_f32)
                                    if has_scale:
                                        T.vmul(sum_f32, tmp_scale, sum_f32)
                                    T.vcast(residual_or_weight_tile, aux_f32)
                                    T.vadd(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, sum_tile, round_mode="rint")
                                else:
                                    T.vcast(x_tile, sum_f32)
                                    if has_scale:
                                        T.vmul(sum_f32, tmp_scale, sum_f32)
                                    T.vcast(residual_or_weight_tile, aux_f32)
                                    T.vadd(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, sum_tile, round_mode="rint")
                                T.copy(
                                    sum_tile[0, 0:block_n],
                                    residual_out[row, n_start : n_start + block_n],
                                )
                                if dtype == "float16":
                                    T.vmul(sum_f32, sum_f32, aux_f32)
                                    T.vmul(aux_f32, avg_factor, aux_f32)
                                    T.reduce(
                                        aux_f32,
                                        partial_rows[local_row : local_row + 1, 0:1],
                                        dims=1,
                                        reduce_mode="sum",
                                        size=[1, block_n],
                                    )
                                else:
                                    T.vmul(sum_f32, sum_f32, sum_f32)
                                    T.vmul(sum_f32, avg_factor, sum_f32)
                                    T.reduce(
                                        sum_f32,
                                        partial_rows[local_row : local_row + 1, 0:1],
                                        dims=1,
                                        reduce_mode="sum",
                                        size=[1, block_n],
                                    )
                        T.vadd(rstd_block, partial_rows, rstd_block)

                    if tail_cols > 0:
                        for local_row in T.serial(row_factor):
                            if local_row < group_rows:
                                row = row_base + local_row
                                n_start = block_count * block_n
                                T.clear(x_tile)
                                T.clear(residual_or_weight_tile)
                                T.copy(
                                    x[row, n_start : n_start + tail_cols],
                                    x_tile[0, 0:tail_cols],
                                )
                                T.copy(
                                    residual[row, n_start : n_start + tail_cols],
                                    residual_or_weight_tile[0, 0:tail_cols],
                                )
                                if dtype == "float16":
                                    T.vcast(x_tile, sum_f32)
                                    if has_scale:
                                        T.vmul(sum_f32, tmp_scale, sum_f32)
                                    T.vcast(residual_or_weight_tile, aux_f32)
                                    T.vadd(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, sum_tile, round_mode="rint")
                                else:
                                    T.vcast(x_tile, sum_f32)
                                    if has_scale:
                                        T.vmul(sum_f32, tmp_scale, sum_f32)
                                    T.vcast(residual_or_weight_tile, aux_f32)
                                    T.vadd(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, sum_tile, round_mode="rint")
                                T.copy(
                                    sum_tile[0, 0:tail_cols],
                                    residual_out[row, n_start : n_start + tail_cols],
                                )
                                if dtype == "float16":
                                    T.vmul(sum_f32, sum_f32, aux_f32)
                                    T.vmul(aux_f32, avg_factor, aux_f32)
                                    T.reduce(
                                        aux_f32,
                                        partial_rows[local_row : local_row + 1, 0:1],
                                        dims=1,
                                        reduce_mode="sum",
                                        size=[1, tail_cols],
                                    )
                                else:
                                    T.vmul(sum_f32, sum_f32, sum_f32)
                                    T.vmul(sum_f32, avg_factor, sum_f32)
                                    T.reduce(
                                        sum_f32,
                                        partial_rows[local_row : local_row + 1, 0:1],
                                        dims=1,
                                        reduce_mode="sum",
                                        size=[1, tail_cols],
                                    )
                        T.vadd(rstd_block, partial_rows, rstd_block)

                    T.vadd(rstd_block, tmp_eps, rstd_block)
                    T.vrsqrt(rstd_block, rstd_block)

                    T.copy(
                        rstd_block[0:group_rows, 0],
                        rstd[row_base : row_base + group_rows],
                    )

                    for no in T.serial(block_count):
                        n_start = no * block_n
                        T.copy(
                            weight[n_start : n_start + block_n],
                            residual_or_weight_tile[0, 0:block_n],
                        )
                        if dtype == "float16":
                            for local_row in T.serial(row_factor):
                                if local_row < group_rows:
                                    row = row_base + local_row
                                    T.copy(
                                        residual_out[row, n_start : n_start + block_n],
                                        sum_tile[0, 0:block_n],
                                    )
                                    T.vcast(sum_tile, sum_f32)
                                    T.vmul(
                                        sum_f32,
                                        rstd_block[local_row : local_row + 1, 0:1],
                                        sum_f32,
                                    )
                                    T.vcast(sum_f32, x_tile)
                                    T.vmul(x_tile, residual_or_weight_tile, x_tile)
                                    T.copy(
                                        x_tile[0, 0:block_n],
                                        out[row, n_start : n_start + block_n],
                                    )
                        else:
                            T.vcast(residual_or_weight_tile, aux_f32)
                            for local_row in T.Pipelined(row_factor, num_stages=1):
                                if local_row < group_rows:
                                    row = row_base + local_row
                                    T.copy(
                                        residual_out[row, n_start : n_start + block_n],
                                        sum_tile[0, 0:block_n],
                                    )
                                    T.vcast(sum_tile, sum_f32)
                                    T.vmul(
                                        sum_f32,
                                        rstd_block[local_row : local_row + 1, 0:1],
                                        sum_f32,
                                    )
                                    T.vcast(sum_f32, x_tile, round_mode="rint")
                                    T.vcast(x_tile, sum_f32)
                                    T.vmul(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, x_tile, round_mode="rint")
                                    T.copy(
                                        x_tile[0, 0:block_n],
                                        out[row, n_start : n_start + block_n],
                                    )

                    if tail_cols > 0:
                        n_start = block_count * block_n
                        T.clear(residual_or_weight_tile)
                        T.copy(
                            weight[n_start : n_start + tail_cols],
                            residual_or_weight_tile[0, 0:tail_cols],
                        )
                        if dtype == "float16":
                            for local_row in T.serial(row_factor):
                                if local_row < group_rows:
                                    row = row_base + local_row
                                    T.clear(sum_tile)
                                    T.copy(
                                        residual_out[
                                            row, n_start : n_start + tail_cols
                                        ],
                                        sum_tile[0, 0:tail_cols],
                                    )
                                    T.vcast(sum_tile, sum_f32)
                                    T.vmul(
                                        sum_f32,
                                        rstd_block[local_row : local_row + 1, 0:1],
                                        sum_f32,
                                    )
                                    T.vcast(sum_f32, x_tile)
                                    T.vmul(x_tile, residual_or_weight_tile, x_tile)
                                    T.copy(
                                        x_tile[0, 0:tail_cols],
                                        out[row, n_start : n_start + tail_cols],
                                    )
                        else:
                            T.vcast(residual_or_weight_tile, aux_f32)
                            for local_row in T.Pipelined(row_factor, num_stages=1):
                                if local_row < group_rows:
                                    row = row_base + local_row
                                    T.clear(sum_tile)
                                    T.copy(
                                        residual_out[
                                            row, n_start : n_start + tail_cols
                                        ],
                                        sum_tile[0, 0:tail_cols],
                                    )
                                    T.vcast(sum_tile, sum_f32)
                                    T.vmul(
                                        sum_f32,
                                        rstd_block[local_row : local_row + 1, 0:1],
                                        sum_f32,
                                    )
                                    T.vcast(sum_f32, x_tile, round_mode="rint")
                                    T.vcast(x_tile, sum_f32)
                                    T.vmul(sum_f32, aux_f32, sum_f32)
                                    T.vcast(sum_f32, x_tile, round_mode="rint")
                                    T.copy(
                                        x_tile[0, 0:tail_cols],
                                        out[row, n_start : n_start + tail_cols],
                                    )

        return main

    return _kernel
