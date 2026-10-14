"""Extracted TileLang kernel implementation for Conv1dFwdOp (NPU reference).

TileLang kernels (Pattern B -- 5 raw builders):
    _conv1d_pointwise_kernel, _conv1d_kernel, _conv1d_unit_stride_kernel,
    _conv1d_group_kernel, _conv1d_direct_kernel

Hand-merged from ``extract_tl_kernel.py`` (pattern B) output, from repo root
``/mnt/workspace/TileOPs-main``:

    src/tileops/kernels/convolution/conv1d.py

The GPU ``Conv1dFwdOp`` dispatches among five kernel classes
(``Conv1dPointwiseKernel`` / ``Conv1dKernel`` / ``Conv1dUnitStrideKernel`` /
``DepthwiseConv1dKernel`` / ``GroupConv1dKernel``) whose ``@tilelang.jit``
builders this file holds.  The extraction emits the raw builders plus the
GPU-only wrapper helpers (``launch`` / ``conv_autotune_configs`` /
``get_sm_version`` / ``grid_refusal`` / ...) that the Kernel *classes* -- not
the builders -- reference; the hand-merge strips those so the module is a
self-contained reference for the 5 builders alone (only ``CONV_SWIZZLE_PANEL``
is shared).  The one non-builder dependency, ``torch``, is preserved.

These five factories are the GPU (CUDA-target) implementations and are
imported by ``conv1d.py`` as the reference the NPU kernel component rewrites
for ``target="npuir"``.  Every builder is an implicit-GEMM conv (weights staged
as ``A`` [C_out, K] and the im2col window staged as ``B`` [K, L], contracted by
``T.gemm``), or -- for the depthwise-direct tap -- a serial MAC loop.  The
feasibility probe ``examples/conv1d/probe/conv1d_geomm_probe.py`` confirmed this
GEMM form compiles and runs on the npuir Cube path in Expert mode
(``T.Scope("Cube")`` + ``T.alloc_L1`` / ``T.alloc_L0C`` / ``T.gemm``, with the
slice-form ``T.copy`` for the strided/overlapping im2col stage; avoid
``load_nd2nz`` per CG-2026-0004).  The NPU kernel component is expected to
re-express each factory as that Cube/Expert form (K1-K4), dropping the CUDA
``threads`` execution parameter (K9).

GPU-only constructs that must be adapted when re-implementing for npuir:
``@tilelang.jit`` (CUDA target), ``T.Kernel(..., threads=threads)``,
``T.alloc_shared`` / ``T.alloc_fragment``, ``T.Pipelined`` / ``T.Parallel`` /
``T.use_swizzle``, and the raw-CUDA ``compile_flags``.
"""

import torch
import tilelang
import tilelang.language as T

# Module-level constant referenced by every builder's ``T.use_swizzle`` panel.
CONV_SWIZZLE_PANEL = 2


def _conv1d_pointwise_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_pointwise_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_pointwise_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(l_in, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                tile_l_end = bx * block_n + block_n - 1
                tile_spatial_full = tile_l_end < l_in
                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)

                    if tile_spatial_full & ((k_iter + 1) * block_k <= c_in):
                        T.copy(x[bz, k_iter * block_k, bx * block_n], data_shared)
                    else:
                        for i, j in T.Parallel(block_k, block_n):
                            ci = k_iter * block_k + i
                            l_idx = bx * block_n + j
                            data_shared[i, j] = T.if_then_else(
                                (ci < c_in) & (l_idx < l_in),
                                x[bz, ci, l_idx],
                                T.cast(0.0, dtype),
                            )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    l_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (l_idx < l_in),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (l_idx < l_in),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv1d_pointwise_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
                out: T.Tensor((n, c_out, l_in), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_pointwise_body(x, weight, out, bias)

            return _conv1d_pointwise_bias_main

        @T.prim_func
        def _conv1d_pointwise_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, c_out, l_in), dtype),  # type: ignore
        ):
            _conv1d_pointwise_body(x, weight, out, None)

        return _conv1d_pointwise_main

    return _conv1d_pointwise_func


def _conv1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    k_total = c_in * kernel_l

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_body(x, weight_flat, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                tile_ol_start = bx * block_n
                tile_ol_end = tile_ol_start + block_n - 1
                tile_input_start = tile_ol_start * stride_l - pad_left
                tile_input_end = tile_ol_end * stride_l + (kernel_l - 1) * dilation_l - pad_left
                tile_spatial_full = (
                    (tile_ol_end < out_l) & (tile_input_start >= 0) & (tile_input_end < l_in)
                )

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)

                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        ol = bx * block_n + j
                        # k runs over (kernel_l, c_in): one k tile then covers a single
                        # tap across every input channel, which is one rectangle of x.
                        # The other order splits each tap across k tiles.
                        kw = k_idx // c_in
                        ci = k_idx % c_in
                        il = ol * stride_l + kw * dilation_l - pad_left
                        if tile_spatial_full & ((k_iter + 1) * block_k <= k_total):
                            data_shared[i, j] = x[bz, ci, il]
                        else:
                            in_bound = (k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in)
                            data_shared[i, j] = T.if_then_else(
                                in_bound,
                                x[bz, ci, il],
                                T.cast(0.0, dtype),
                            )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv1d_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_body(x, weight_flat, out, bias)

            return _conv1d_bias_main

        @T.prim_func
        def _conv1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_body(x, weight_flat, out, None)

        return _conv1d_main

    return _conv1d_func


def _conv1d_unit_stride_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    if stride_l != 1:
        raise ValueError(f"_conv1d_unit_stride_kernel serves stride 1, got {stride_l}")
    accum_dtype = "float"
    element_bytes = torch.tensor([], dtype=getattr(torch, dtype)).element_size()
    out_l = l_in + pad_left + pad_right - dilation_l * (kernel_l - 1)
    k_total = c_in * kernel_l

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_unit_stride_func(
        block_m: int,
        block_n: int,
        block_k: int,
        taps: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        if taps not in (1, 2) or kernel_l % taps:
            raise ValueError(f"taps={taps} must be 1 or 2 and divide kernel_l={kernel_l}")
        channel_blocks = (c_in + block_k - 1) // block_k

        @T.macro
        def _load_tap(
            x, weight_flat, weight_shared, data_shared, row0, kw, c0, by, bz, window_start, interior
        ):
            """Load tap ``kw``'s weight and x tiles into rows ``row0`` of the k tile."""
            # A tap's weight columns start kw * c_in elements in, a TMA source only
            # where that offset is 16-byte aligned.
            T.copy(
                weight_flat[by * block_m, kw * c_in + c0],
                weight_shared[0:block_m, row0 : row0 + block_k],
                disable_tma=c_in * element_bytes % 16 != 0,
            )
            if interior:
                # A tap shifts the window by one element, where no TMA box starts.
                T.copy(
                    x[bz, c0, window_start + kw * dilation_l],
                    data_shared[row0 : row0 + block_k, 0:block_n],
                    disable_tma=True,
                )
            else:
                for i, j in T.Parallel(block_k, block_n):
                    ci = c0 + i
                    il = window_start + kw * dilation_l + j
                    data_shared[row0 + i, j] = T.if_then_else(
                        (ci < c_in) & (il >= 0) & (il < l_in),
                        x[bz, ci, il],
                        T.cast(0.0, dtype),
                    )

        @T.macro
        def _conv1d_unit_stride_body(x, weight_flat, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, taps * block_k), dtype)
                data_shared = T.alloc_shared((taps * block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                # A k tile is `taps` taps over block_k input channels each; each tap's x
                # tile is the rectangle it shifts the output tile onto. Rows past c_in
                # read as zero and cancel the next tap's weight columns.
                # The copy guards columns per vector, not per element, so a tile whose
                # window crosses either end of x loads element by element instead.
                window_start = bx * block_n - pad_left
                interior = (window_start >= 0) & (
                    window_start + block_n + (kernel_l - 1) * dilation_l <= l_in
                )
                for k_iter in T.Pipelined(kernel_l // taps * channel_blocks, num_stages=num_stages):
                    kw = k_iter // channel_blocks * taps
                    c0 = k_iter % channel_blocks * block_k
                    _load_tap(
                        x,
                        weight_flat,
                        weight_shared,
                        data_shared,
                        0,
                        kw,
                        c0,
                        by,
                        bz,
                        window_start,
                        interior,
                    )
                    if taps == 2:
                        _load_tap(
                            x,
                            weight_flat,
                            weight_shared,
                            data_shared,
                            block_k,
                            kw + 1,
                            c0,
                            by,
                            bz,
                            window_start,
                            interior,
                        )
                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (ol < out_l),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv1d_unit_stride_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_unit_stride_body(x, weight_flat, out, bias)

            return _conv1d_unit_stride_bias_main

        @T.prim_func
        def _conv1d_unit_stride_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight_flat: T.Tensor((c_out, k_total), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_unit_stride_body(x, weight_flat, out, None)

        return _conv1d_unit_stride_main

    return _conv1d_unit_stride_func


def _conv1d_group_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
    groups: int = 1,
    c_in_g: int = 0,
    c_out_g: int = 0,
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    k_total = c_in_g * kernel_l

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_group_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_group_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out_g, block_m),
                n * groups,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                batch_id = bz // groups
                group_id = bz % groups
                oc_base = group_id * c_out_g + by * block_m

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, k in T.Parallel(block_m, block_k):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        k_idx = k_iter * block_k + k
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        weight_shared[i, k] = T.if_then_else(
                            (oc_g < c_out_g) & (k_idx < k_total),
                            weight[oc, ci_g, kw],
                            T.cast(0.0, dtype),
                        )

                    for k, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + k
                        ol = bx * block_n + j
                        # k runs over (kernel_l, c_in_g), matching the general Conv1d
                        # kernel; the weight is staged by gather because that order is not
                        # the one it is stored in.
                        kw = k_idx // c_in_g
                        ci_g = k_idx % c_in_g
                        il = ol * stride_l + kw * dilation_l - pad_left
                        data_shared[k, j] = T.if_then_else(
                            (k_idx < k_total) & (ol < out_l) & (il >= 0) & (il < l_in),
                            x[batch_id, group_id * c_in_g + ci_g, il],
                            T.cast(0.0, dtype),
                        )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    ol = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (ol < out_l),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (ol < out_l),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                if c_out_g % block_m == 0:
                    # The tile ends on this group's last channel, so the copy cannot spill
                    # into the next group's rows.
                    T.copy(out_shared, out[batch_id, oc_base, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        ol = bx * block_n + j
                        if oc_g < c_out_g and ol < out_l:
                            out[batch_id, oc, ol] = out_shared[i, j]

        if has_bias:

            @T.prim_func
            def _conv1d_group_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_group_body(x, weight, out, bias)

            return _conv1d_group_bias_main

        @T.prim_func
        def _conv1d_group_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in_g, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_group_body(x, weight, out, None)

        return _conv1d_group_main

    return _conv1d_group_func


def _conv1d_direct_kernel(
    n: int,
    c_in: int,
    l_in: int,
    c_out: int,
    kernel_l: int,
    stride_l: int,
    pad_left: int,
    pad_right: int,
    dilation_l: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv1d_direct_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv1d_direct_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_l, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(CONV_SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for kw in T.serial(kernel_l):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        ol = bx * block_n + j
                        il = ol * stride_l + kw * dilation_l - pad_left
                        valid = (oc < c_out) & (ol < out_l) & (il >= 0) & (il < l_in)
                        out_local[i, j] += T.if_then_else(
                            valid,
                            T.cast(x[bz, oc, il], accum_dtype)
                            * T.cast(weight[oc, 0, kw], accum_dtype),
                            T.cast(0.0, accum_dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    ol = bx * block_n + j
                    if oc < c_out and ol < out_l:
                        if has_bias:
                            out[bz, oc, ol] = T.cast(
                                out_local[i, j] + T.cast(bias[oc], accum_dtype),
                                dtype,
                            )
                        else:
                            out[bz, oc, ol] = T.cast(out_local[i, j], dtype)

        if has_bias:

            @T.prim_func
            def _conv1d_direct_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, 1, kernel_l), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv1d_direct_body(x, weight, out, bias)

            return _conv1d_direct_bias_main

        @T.prim_func
        def _conv1d_direct_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, 1, kernel_l), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_l), dtype),  # type: ignore
        ):
            _conv1d_direct_body(x, weight, out, None)

        return _conv1d_direct_main

    return _conv1d_direct_func


__all__ = [
    "_conv1d_direct_kernel",
    "_conv1d_group_kernel",
    "_conv1d_kernel",
    "_conv1d_pointwise_kernel",
    "_conv1d_unit_stride_kernel",
]
