from .common import FULL_OUT_IDX, T, require_tilelang, tilelang


def build_split_d(M: int, N: int, eps: float, dtype: str, scale: float = 1.0):
    require_tilelang()
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0
    pipeline_stages = 2 if dtype == "float16" else 1

    @tilelang.jit(
        out_idx=FULL_OUT_IDX,
        target="npuir",
        pass_configs={
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: dtype == "float16",
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    )
    def _kernel(block_m, block_n):
        block_count = N // block_n
        tail_cols = N - block_count * block_n
        out_tile_cols = block_n if dtype != "float32" else 1
        sum_f32_cols = block_n if dtype != "float32" else 1
        weight_f32_cols = block_n if dtype == "bfloat16" else 1

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
                row_size = T.min(block_m, M - offset_m)

                x_tile = T.alloc_ub((block_m, block_n), dtype)
                residual_tile = T.alloc_ub((block_m, block_n), dtype)
                sum_tile = T.alloc_ub((block_m, block_n), dtype)
                weight_tile = T.alloc_ub((1, block_n), dtype)
                out_tile = T.alloc_ub((block_m, out_tile_cols), dtype)
                sum_f32 = T.alloc_ub((block_m, sum_f32_cols), "float32")
                sq_f32 = T.alloc_ub((block_m, block_n), "float32")
                weight_f32 = T.alloc_ub((1, weight_f32_cols), "float32")
                local_sum = T.alloc_ub((block_m, 1), "float32")
                rstd_block = T.alloc_ub((block_m, 1), "float32")

                T.clear(rstd_block)

                for no in T.Pipelined(block_count, num_stages=pipeline_stages):
                    n_start = no * block_n
                    if row_size < block_m:
                        T.clear(x_tile)
                        T.clear(residual_tile)
                    T.copy(
                        x[offset_m : offset_m + row_size, n_start : n_start + block_n],
                        x_tile[0:row_size, 0:block_n],
                    )
                    T.copy(
                        residual[
                            offset_m : offset_m + row_size, n_start : n_start + block_n
                        ],
                        residual_tile[0:row_size, 0:block_n],
                    )

                    if dtype == "float32":
                        if has_scale:
                            T.vmul(x_tile, tmp_scale, x_tile)
                        T.vadd(x_tile, residual_tile, sum_tile)
                        T.copy(
                            sum_tile[0:row_size, 0:block_n],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + block_n,
                            ],
                        )
                        T.vmul(sum_tile, sum_tile, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)
                    elif dtype == "float16":
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")
                        T.copy(
                            sum_tile[0:row_size, 0:block_n],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + block_n,
                            ],
                        )
                        T.vmul(sum_f32, sum_f32, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)
                    else:
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")
                        T.copy(
                            sum_tile[0:row_size, 0:block_n],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + block_n,
                            ],
                        )
                        T.vmul(sum_f32, sum_f32, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)

                    T.reduce(
                        sq_f32,
                        local_sum,
                        dims=1,
                        reduce_mode="sum",
                        size=[block_m, block_n],
                    )
                    T.vadd(rstd_block, local_sum, rstd_block)

                if tail_cols > 0:
                    n_start = block_count * block_n
                    T.clear(x_tile)
                    T.clear(residual_tile)
                    T.copy(
                        x[
                            offset_m : offset_m + row_size,
                            n_start : n_start + tail_cols,
                        ],
                        x_tile[0:row_size, 0:tail_cols],
                    )
                    T.copy(
                        residual[
                            offset_m : offset_m + row_size,
                            n_start : n_start + tail_cols,
                        ],
                        residual_tile[0:row_size, 0:tail_cols],
                    )

                    if dtype == "float32":
                        if has_scale:
                            T.vmul(x_tile, tmp_scale, x_tile)
                        T.vadd(x_tile, residual_tile, sum_tile)
                        T.copy(
                            sum_tile[0:row_size, 0:tail_cols],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + tail_cols,
                            ],
                        )
                        T.vmul(sum_tile, sum_tile, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)
                    elif dtype == "float16":
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")
                        T.copy(
                            sum_tile[0:row_size, 0:tail_cols],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + tail_cols,
                            ],
                        )
                        T.vmul(sum_f32, sum_f32, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)
                    else:
                        T.vcast(x_tile, sum_f32)
                        if has_scale:
                            T.vmul(sum_f32, tmp_scale, sum_f32)
                        T.vcast(residual_tile, sq_f32)
                        T.vadd(sum_f32, sq_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")
                        T.copy(
                            sum_tile[0:row_size, 0:tail_cols],
                            residual_out[
                                offset_m : offset_m + row_size,
                                n_start : n_start + tail_cols,
                            ],
                        )
                        T.vmul(sum_f32, sum_f32, sq_f32)
                        T.vmul(sq_f32, avg_factor, sq_f32)

                    T.reduce(
                        sq_f32,
                        local_sum,
                        dims=1,
                        reduce_mode="sum",
                        size=[block_m, tail_cols],
                    )
                    T.vadd(rstd_block, local_sum, rstd_block)

                T.vadd(rstd_block, tmp_eps, rstd_block)
                T.vrsqrt(rstd_block, rstd_block)

                T.copy(rstd_block[0:row_size, 0], rstd[offset_m : offset_m + row_size])

                for no in T.Pipelined(block_count, num_stages=pipeline_stages):
                    n_start = no * block_n
                    T.copy(
                        residual_out[
                            offset_m : offset_m + row_size, n_start : n_start + block_n
                        ],
                        sum_tile[0:row_size, 0:block_n],
                    )
                    T.copy(
                        weight[n_start : n_start + block_n], weight_tile[0, 0:block_n]
                    )

                    if dtype == "float32":
                        T.vmul(sum_tile, rstd_block, sum_tile)
                        T.vmul(sum_tile, weight_tile, sum_tile)
                    elif dtype == "float16":
                        T.vcast(sum_tile, sum_f32)
                        T.vmul(sum_f32, rstd_block, sum_f32)
                        T.vcast(sum_f32, out_tile)
                        T.vmul(out_tile, weight_tile, sum_tile)
                    else:
                        T.vcast(sum_tile, sum_f32)
                        T.vmul(sum_f32, rstd_block, sum_f32)
                        T.vcast(sum_f32, out_tile, round_mode="rint")
                        T.vcast(out_tile, sum_f32)
                        T.vcast(weight_tile, weight_f32)
                        T.vmul(sum_f32, weight_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")

                    T.copy(
                        sum_tile[0:row_size, 0:block_n],
                        out[
                            offset_m : offset_m + row_size, n_start : n_start + block_n
                        ],
                    )

                if tail_cols > 0:
                    n_start = block_count * block_n
                    T.clear(sum_tile)
                    T.clear(weight_tile)
                    T.copy(
                        residual_out[
                            offset_m : offset_m + row_size,
                            n_start : n_start + tail_cols,
                        ],
                        sum_tile[0:row_size, 0:tail_cols],
                    )
                    T.copy(
                        weight[n_start : n_start + tail_cols],
                        weight_tile[0, 0:tail_cols],
                    )

                    if dtype == "float32":
                        T.vmul(sum_tile, rstd_block, sum_tile)
                        T.vmul(sum_tile, weight_tile, sum_tile)
                    elif dtype == "float16":
                        T.vcast(sum_tile, sum_f32)
                        T.vmul(sum_f32, rstd_block, sum_f32)
                        T.vcast(sum_f32, out_tile)
                        T.vmul(out_tile, weight_tile, sum_tile)
                    else:
                        T.vcast(sum_tile, sum_f32)
                        T.vmul(sum_f32, rstd_block, sum_f32)
                        T.vcast(sum_f32, out_tile, round_mode="rint")
                        T.vcast(out_tile, sum_f32)
                        T.vcast(weight_tile, weight_f32)
                        T.vmul(sum_f32, weight_f32, sum_f32)
                        T.vcast(sum_f32, sum_tile, round_mode="rint")

                    T.copy(
                        sum_tile[0:row_size, 0:tail_cols],
                        out[
                            offset_m : offset_m + row_size,
                            n_start : n_start + tail_cols,
                        ],
                    )

        return main

    return _kernel
