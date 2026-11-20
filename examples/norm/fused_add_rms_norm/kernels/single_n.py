from .common import FULL_OUT_IDX, T, aligned_cols, require_tilelang, tilelang


def build_single_n(M: int, N: int, eps: float, dtype: str, scale: float = 1.0):
    require_tilelang()
    n_align = aligned_cols(N, dtype)
    tail_cols = n_align - N
    avg_factor = 1.0 / float(N)
    tmp_eps = eps
    tmp_scale = scale
    has_scale = scale != 1.0

    if dtype == "float16":

        @tilelang.jit(out_idx=FULL_OUT_IDX, target="npuir")
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
                with T.Kernel(M, is_npu=True) as (pid_m, _):
                    x_tile = T.alloc_ub((1, n_align), dtype)
                    residual_tile = T.alloc_ub((1, n_align), dtype)
                    sum_f32 = T.alloc_ub((1, n_align), "float32")
                    sq_f32 = T.alloc_ub((1, n_align), "float32")
                    rstd_block = T.alloc_ub((1, 1), "float32")

                    if tail_cols > 0:
                        T.clear(x_tile)
                        T.clear(residual_tile)
                    T.copy(x[pid_m : pid_m + 1, 0:N], x_tile[0:1, 0:N])
                    T.copy(residual[pid_m : pid_m + 1, 0:N], residual_tile[0:1, 0:N])

                    T.vcast(x_tile, sum_f32)
                    if has_scale:
                        T.vmul(sum_f32, tmp_scale, sum_f32)
                    T.vcast(residual_tile, sq_f32)
                    T.vadd(sum_f32, sq_f32, sum_f32)
                    T.vcast(sum_f32, x_tile, round_mode="rint")
                    T.copy(x_tile[0:1, 0:N], residual_out[pid_m : pid_m + 1, 0:N])
                    T.copy(weight[0:N], residual_tile[0, 0:N])

                    T.vmul(sum_f32, sum_f32, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)

                    T.reduce(
                        sq_f32,
                        rstd_block,
                        dims=1,
                        reduce_mode="sum",
                        size=[1, n_align],
                    )
                    T.vadd(rstd_block, tmp_eps, rstd_block)
                    T.vrsqrt(rstd_block, rstd_block)

                    T.copy(rstd_block[0:1, 0:1], rstd[pid_m : pid_m + 1])

                    T.vmul(sum_f32, rstd_block[0, 0], sum_f32)
                    T.vcast(sum_f32, x_tile)
                    T.vmul(x_tile, residual_tile, x_tile)
                    T.copy(x_tile[0:1, 0:N], out[pid_m : pid_m + 1, 0:N])

            return main

        return _kernel

    @tilelang.jit(out_idx=FULL_OUT_IDX, target="npuir")
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
            with T.Kernel(M, is_npu=True) as (pid_m, _):
                x_tile = T.alloc_ub((1, n_align), dtype)
                residual_tile = T.alloc_ub((1, n_align), dtype)
                sum_tile = T.alloc_ub((1, n_align), dtype)
                out_tile = T.alloc_ub((1, n_align), dtype)
                weight_tile = T.alloc_ub((1, n_align), dtype)
                sum_f32 = T.alloc_ub((1, n_align), "float32")
                sq_f32 = T.alloc_ub((1, n_align), "float32")
                weight_f32 = T.alloc_ub((1, n_align), "float32")
                rstd_block = T.alloc_ub((1, 1), "float32")
                rstd_tile = T.alloc_ub((1,), "float32")

                if tail_cols > 0:
                    T.clear(x_tile)
                    T.clear(residual_tile)
                    T.clear(weight_tile)
                T.copy(x[pid_m : pid_m + 1, 0:N], x_tile[0:1, 0:N])
                T.copy(residual[pid_m : pid_m + 1, 0:N], residual_tile[0:1, 0:N])
                T.copy(weight[0:N], weight_tile[0, 0:N])

                if dtype == "float32":
                    if has_scale:
                        T.vmul(x_tile, tmp_scale, x_tile)
                    T.vadd(x_tile, residual_tile, sum_tile)
                    T.copy(sum_tile[0:1, 0:N], residual_out[pid_m : pid_m + 1, 0:N])
                    T.vmul(sum_tile, sum_tile, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)
                else:
                    T.vcast(x_tile, sum_f32)
                    if has_scale:
                        T.vmul(sum_f32, tmp_scale, sum_f32)
                    T.vcast(residual_tile, sq_f32)
                    T.vadd(sum_f32, sq_f32, sum_f32)
                    T.vcast(sum_f32, sum_tile, round_mode="rint")
                    T.copy(sum_tile[0:1, 0:N], residual_out[pid_m : pid_m + 1, 0:N])
                    T.vmul(sum_f32, sum_f32, sq_f32)
                    T.vmul(sq_f32, avg_factor, sq_f32)

                T.reduce(
                    sq_f32,
                    rstd_block,
                    dims=1,
                    reduce_mode="sum",
                    size=[1, n_align],
                )
                T.vadd(rstd_block, tmp_eps, rstd_block)
                T.vrsqrt(rstd_block, rstd_block)

                rstd_tile[0] = rstd_block[0, 0]
                T.copy(rstd_tile[0:1], rstd[pid_m : pid_m + 1])

                if dtype == "float32":
                    T.vmul(sum_tile, rstd_block, sum_tile)
                    T.vmul(sum_tile, weight_tile, out_tile)
                else:
                    T.vmul(sum_f32, rstd_block, sum_f32)
                    T.vcast(sum_f32, out_tile, round_mode="rint")
                    T.vcast(out_tile, sum_f32)
                    T.vcast(weight_tile, weight_f32)
                    T.vmul(sum_f32, weight_f32, sum_f32)
                    T.vcast(sum_f32, out_tile, round_mode="rint")
                T.copy(out_tile[0:1, 0:N], out[pid_m : pid_m + 1, 0:N])

        return main

    return _kernel
