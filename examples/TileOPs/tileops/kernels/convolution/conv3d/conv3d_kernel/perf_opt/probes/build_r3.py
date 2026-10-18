#!/usr/bin/env python3
"""Build Conv3dBiasFwdOp_round2_r3_ndhwc.py from the current best kernel.

Adds ONE new compile-time path (R3) at the same nesting level as R1:

  R3 = padded-N multi-row spatial tiling (R2 structure) + channel-last (NDHWC)
       input, so every Cube GM->L1 im2col copy is one contiguous run and the
       GEMM consumes B as [N, K] with b_transpose=True.

Host side: the xdec slot carries the NDHWC padded input for stride-1 shapes
(polyphase shapes keep the decimated NCDHW tensor there), and x_pad gains one
extra all-zero depth plane so the contiguous span read can overrun the last
plane safely.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.dirname(HERE)
src = open(os.path.join(PERF, "Conv3dBiasFwdOp.py")).read()

# ---------------------------------------------------------------- host prep
old = """    x_pad = torch.zeros((n, c_in_pad, d_pad, h_pad, w_pad), dtype=dtype, device=dev)"""
new = """    # One extra all-zero depth plane (`_host_prep` -> plane_tot stride): the R3
    # Cube copy reads a contiguous n_tile span whose tail can run kw-1 elements
    # past the last plane of the last sample, where the exactly-sized allocation
    # ends on a page boundary.
    x_pad = torch.zeros((n, c_in_pad, d_pad + 1, h_pad, w_pad), dtype=dtype,
                        device=dev)"""
assert old in src
src = src.replace(old, new)

old = """        flat = x_pad.reshape(n, c_in_pad, d_pad, h_pad, w_pad)"""
new = """        flat = x_pad.reshape(n, c_in_pad, d_pad + 1, h_pad, w_pad)"""
assert old in src
src = src.replace(old, new)

old = """    else:
        xdec = x_pad.clone()"""
new = """    else:
        # Stride-1 shapes do not use the polyphase tensor, so the xdec slot
        # carries the channel-last (NDHWC) padded input that R3 reads.  Building
        # it here costs one permuted copy instead of the previous full clone.
        xdec = torch.zeros((n, d_pad + 1, h_pad, w_pad, c_in_pad), dtype=dtype,
                           device=dev)
        xdec[:, pd:pd + d_in, ph:ph + h_in, pw:pw + w_in, :x.shape[1]] = (
            x.permute(0, 2, 3, 4, 1).to(dtype)
        )
        xdec = xdec.reshape(n, (d_pad + 1) * h_pad * w_pad, c_in_pad)"""
assert old in src
src = src.replace(old, new)

# ------------------------------------------------------------- plane_tot var
old = """        n_kc = (c_in_pad + block_k - 1) // block_k
        k_full = KV * c_in_pad"""
new = """        n_kc = (c_in_pad + block_k - 1) // block_k
        k_full = KV * c_in_pad
        # `_host_prep` allocates one extra zero depth plane, so every path
        # declares the padded per-channel stride.
        plane_tot = (d_pad + 1) * h_pad * w_pad"""
assert old in src
src = src.replace(old, new)

src = src.replace("T.Tensor((n, c_in_pad, d_pad * h_pad * w_pad), dtype)",
                  "T.Tensor((n, c_in_pad, plane_tot), dtype)")

# ---------------------------------------------------------------- R3 path
anchor = "        if use_r1:\n"
assert src.count(anchor) == 1

block = '''        # ---- R3: padded-N tiling + channel-last input ---------------------------
        # unet-encoder (c_in_pad=32) is dominated by the Cube im2col copy: one
        # MTE2 instruction per plane-strided channel row (msprof: 91.8k MTE2
        # instructions, GM->L1 at 5.6% of rated bandwidth, aic_mte2_ratio 0.99).
        # Two structural changes, both compile-time:
        #   1. N axis = r3_rows output rows in *padded-width* coordinates
        #      (n_tile = r3_rows*w_pad): one copy span covers r3_rows rows at
        #      their native w_pad pitch, so copies/output-row drop by r3_rows.
        #   2. input staged channel-last (NDHWC, via the xdec slot): the KW taps
        #      shift by c_in_pad elements in a flat stream, so each tap copy is
        #      ONE contiguous run and the GEMM reads B as [N, K] with
        #      b_transpose=True (probe-verified on this toolchain, unlike the
        #      Developer-mode failure recorded in CG-2026-0026).
        # Row groups overlap at the tail (oh0 = min(hg*R, out_h-R)); the
        # duplicated rows are recomputed with identical values.
        _L0C_CAP = 120 * 1024
        r3_rows = max(1, min(512 // w_pad, out_h,
                             _L0C_CAP // (block_m * 4) // w_pad))
        n_tile3 = r3_rows * w_pad
        _l1_r3 = (block_m * c_in_pad + n_tile3 * c_in_pad) * 2
        use_r3 = (
            _stride1
            and dd == 1
            and dh == 1
            and dw == 1
            and n_kc == 1
            and gy <= num_kernels
            and out_w <= block_n
            and r3_rows >= 2
            and c_in_pad >= 16
            and c_in_pad % 16 == 0
            and _l1_r3 <= _L1_BUDGET
            and block_m * n_tile3 * 4 <= _L0C_CAP
        )
        hgroups = (out_h + r3_rows - 1) // r3_rows
        num_logical3 = n * out_d * hgroups
        share3 = num_kernels // gy
        chunk3 = (num_logical3 + share3 - 1) // share3
        ncols3 = _ceil16(out_w)

        if use_r3:

            @T.macro
            def _cube_r3(out, w_flat, x_nd, ws, kernel_id, num_local_tasks):
                with T.Scope("Cube"):
                    l1_a = T.alloc_L1([block_m, c_in_pad], dtype)
                    l1_b = T.alloc_L1([n_tile3, c_in_pad], dtype)
                    l0_c = T.alloc_L0C([block_m, n_tile3], accum_dtype)
                    by = kernel_id % gy
                    sub = kernel_id // gy
                    tm = T.min(block_m, c_out - by * block_m)
                    tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                    for task_id in T.serial(num_local_tasks):
                        idx = sub * chunk3 + task_id
                        bz = idx // (out_d * hgroups)
                        rem = idx % (out_d * hgroups)
                        od = rem // hgroups
                        hg = rem % hgroups
                        oh0 = T.min(hg * r3_rows, out_h - r3_rows)
                        for di in T.serial(kd):
                            for hi in T.serial(kh):
                                for wi in T.serial(kw):
                                    with T.rs("PIPE_MTE2"):
                                        T.copy(
                                            w_flat[
                                                (di * kh + hi) * kw + wi,
                                                by * block_m:by * block_m + tm,
                                                0:c_in_pad,
                                            ],
                                            l1_a[0:tm, 0:c_in_pad],
                                        )
                                        off = (
                                            ((od + di * dd) * h_pad + oh0 + hi * dh)
                                            * w_pad
                                            + wi * dw
                                        )
                                        T.copy(
                                            x_nd[bz, off:off + n_tile3, 0:c_in_pad],
                                            l1_b,
                                        )
                                    with T.rs("PIPE_C"):
                                        T.gemm(
                                            l1_a,
                                            l1_b,
                                            l0_c,
                                            initC=((di * kh + hi) * kw + wi == 0),
                                            b_transpose=True,
                                            size=[tmc, c_in_pad, n_tile3],
                                        )
                        if has_bias:
                            for r in T.serial(r3_rows):
                                slot = (task_id * r3_rows + r) % SLOTS
                                with T.rs("PIPE_FIX"):
                                    T.sync_block_wait(2 * slot + 1)
                                    T.copy(
                                        l0_c[0:tm, r * w_pad:r * w_pad + out_w],
                                        ws[kernel_id, slot, 0:tm, 0:out_w],
                                    )
                                    T.sync_block_set(2 * slot)
                        else:
                            for r in T.serial(r3_rows):
                                with T.rs("PIPE_FIX"):
                                    obase = (od * out_h + oh0 + r) * out_w
                                    T.copy(
                                        l0_c[0:tm, r * w_pad:r * w_pad + out_w],
                                        out[
                                            bz,
                                            by * block_m:by * block_m + tm,
                                            obase:obase + out_w,
                                        ],
                                    )

            @T.macro
            def _bias_epilogue_r3(out, bias, ws, kernel_id, num_local_tasks):
                with T.Scope("Vector"):
                    ub_f32 = T.alloc_ub([block_m, ncols3], accum_dtype)
                    ub_f16 = T.alloc_ub([block_m, ncols3], dtype)
                    bias_16 = T.alloc_ub([block_m, 1], dtype)
                    bias_col = T.alloc_ub([block_m, 1], accum_dtype)
                    bias_brc = T.alloc_ub([block_m, ncols3], accum_dtype)
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_set(1)
                        T.sync_block_set(3)
                    by = kernel_id % gy
                    sub = kernel_id // gy
                    tm = T.min(block_m, c_out - by * block_m)
                    for task_id in T.serial(num_local_tasks):
                        idx = sub * chunk3 + task_id
                        bz = idx // (out_d * hgroups)
                        rem = idx % (out_d * hgroups)
                        od = rem // hgroups
                        hg = rem % hgroups
                        oh0 = T.min(hg * r3_rows, out_h - r3_rows)
                        for r in T.serial(r3_rows):
                            slot = (task_id * r3_rows + r) % SLOTS
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(2 * slot)
                                T.copy(
                                    ws[kernel_id, slot, 0:tm, 0:out_w],
                                    ub_f32[0:tm, 0:out_w],
                                )
                                T.copy(
                                    bias[by * block_m:by * block_m + tm],
                                    bias_16[0:tm, 0:1],
                                )
                            T.vcast(bias_16, bias_col, round_mode="rint")
                            T.vbrc(bias_col, bias_brc)
                            T.vadd(bias_brc, ub_f32, ub_f32)
                            T.vcast(ub_f32, ub_f16, round_mode="rint")
                            with T.rs("PIPE_MTE3"):
                                obase = (od * out_h + oh0 + r) * out_w
                                T.copy(
                                    ub_f16[0:tm, 0:out_w],
                                    out[
                                        bz,
                                        by * block_m:by * block_m + tm,
                                        obase:obase + out_w,
                                    ],
                                )
                                T.sync_block_set(2 * slot + 1)

            if has_bias:

                @T.prim_func
                def _conv3d_bias_main_r3(
                    x_pad: T.Tensor((n, c_in_pad, plane_tot), dtype),
                    w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
                    out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
                    bias: T.Tensor((c_out,), dtype),
                    xdec: T.Tensor((n, plane_tot, c_in_pad), dtype),
                    ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                        sub = kernel_id // gy
                        nlt = T.min(chunk3, T.max(num_logical3 - sub * chunk3, 0))
                        _cube_r3(out, w_flat, xdec, ws, kernel_id, nlt)
                        _bias_epilogue_r3(out, bias, ws, kernel_id, nlt)

                return _conv3d_bias_main_r3

            @T.prim_func
            def _conv3d_main_r3(
                x_pad: T.Tensor((n, c_in_pad, plane_tot), dtype),
                w_flat: T.Tensor((KV, c_out, c_in_pad), dtype),
                out: T.Tensor((n, c_out, out_d * out_h * out_w), dtype),
                xdec: T.Tensor((n, plane_tot, c_in_pad), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    sub = kernel_id // gy
                    nlt = T.min(chunk3, T.max(num_logical3 - sub * chunk3, 0))
                    _cube_r3(out, w_flat, xdec, None, kernel_id, nlt)

            return _conv3d_main_r3

'''

src = src.replace(anchor, block + anchor)

# docstring note
src = src.replace("""Run: python Conv3dBiasFwdOp.py --level {L0,all}""",
                  """R2/R3 (Stage-4 round 2, unet path): see the ``use_r3`` block -- padded-N
multi-row tiling with a channel-last staged input (one contiguous MTE2 run per
tap instead of ``c_in_pad`` plane-strided segments).

Run: python Conv3dBiasFwdOp.py --level {L0,all}""")

out = os.path.join(PERF, "Conv3dBiasFwdOp_round2_r3_ndhwc.py")
open(out, "w").write(src)
import ast
ast.parse(src)
print("written", out, len(src.splitlines()), "lines")
