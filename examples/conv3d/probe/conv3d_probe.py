# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Feasibility probe: 3-D implicit-GEMM Conv3d on npuir (Cube path), not a migration.

Question: can the canonical 3-D implicit-GEMM convolution

    out[C_out, M] = sum_{tap=(kd,kh,kw)} W_tap[C_out, C_in] @ im2col_tap[C_in, M]

with im2col over the 3-D kernel volume gathered by strided, slice-form ``T.copy``
from a 5-D NCDHW input, plus an in-kernel weight re-layout and ``T.gemm``, be
expressed, compiled and run on the npuir Cube path (target="npuir", Expert mode:
T.Scope("Cube")/T.Scope("Vector") + T.alloc_L1 / T.alloc_L0C / T.gemm / T.copy)?
And is the bias path expressible with the verified MixCV Cube->Vector handshake?

This is the Conv3d analogue of the Conv1d precedent at
``examples/conv1d/probe/conv1d_geomm_probe.py`` (verified, max_abs_err 5.7e-05).
The extra dimensions vs Conv1d are the whole point of the probe:

  * 5-D GM tensors  x[N, C_in, D, H, W], weight[C_out, C_in, kd, kh, kw];
  * im2col over kd*kh*kw = 27 taps (vs kW=3), each tap a strided slice whose
    "row" stride is DP*HP and whose in-row shift is kw (3-D address arithmetic);
  * in-kernel weight re-layout [C_out, C_in, KV] -> [KV, C_out, C_in] via two
    T.transpose axis-swaps (KV = kd*kh*kw), matching the Conv1d R3 prelude.

Probe workload (small, representative of the r3d-stem / unet-encoder workloads:
stride=1, dilation=1, padding=1, kd=kh=kw=3):
    N=1, C_in=16, C_out=32, D=H=4, W=16, padding=1
    -> out_d=out_h=4, out_w=16, out_dhw=256, KV=27.
    x is staged zero-padded to [1, 16, 6, 6, 18] so every im2col tap slice is an
    in-bounds read (the padding-1 workloads' OOB handling, as a prelude).

Tiling (chosen so no tail masking is needed -> isolates the 3-D gather):
    BLOCK_M = C_out  = 32,  BLOCK_N = out_w = 16,  BLOCK_K = C_in = 16
    Grid = out_d*out_h = 16 blocks, one output (od,oh) row per block;
    each block runs 27 accumulating T.gemm over the kd*kh*kw taps.

Why per-tap strided slices (not a flat im2col): a block of consecutive
flattened output positions does NOT map to a shifted input window once the
kernel is 3-D, so the Conv1d flatten-and-shift trick does not generalise; the
block must stay inside one output row, and each tap's input row (id, ih) is
read as the slice x_pad[0, :, od+kd, oh+kh, kw : kw+BLOCK_N] (row stride DP*HP,
contiguous in W, shifted by kw). Slice-form T.copy is the documented, verified
workaround for strided GM->L1 regions (CG-2026-0004; strided T.load_nd2nz
silently misreads; docs/Tilelang.language/内存操作/T.copy.md §2.4 ex.4).

Two kernels are probed:
  A. no-bias  : pure Cube, L0C -> GM (fp32) fixpipe. Isolates the 3-D gather.
  B. with-bias: MixCV. Cube relays the fp32 L0C tile through a GM workspace to
     Vector, which adds bias (vcast -> vbrc -> vadd in fp32), casts to fp16 and
     stores. Mirrors the verified Conv1d bias handshake (flag set by producer
     Cube, waited by consumer Vector).

Golden = torch.nn.functional.conv3d (pure-torch CPU), compared in fp32.
Independent of the NPU algorithm.

Run: python conv3d_probe.py
"""

import os
import time
import traceback

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T

# ---- workload (stride=1, dilation=1, padding=1, kd=kh=kw=3) ----
N_BATCH = 1
C_IN = 16
C_OUT = 32
KD = KH = KW = 3
KV = KD * KH * KW  # 27
PAD = 1

D = 4
H = 4
W = 16
OUT_D = D  # (D + 2*PAD - (KD-1) - 1) // 1 + 1 = D
OUT_H = H
OUT_W = W
OUT_DHW = OUT_D * OUT_H * OUT_W  # 256

# zero-padded staging: x_pad[n, c, id+PAD, ih+PAD, iw+PAD] = x[...]
DP = D + 2 * PAD  # 6
HP = H + 2 * PAD  # 6
WP = W + 2 * PAD  # 18

DTYPE = "float16"
ACCUM = "float32"

# ---- tiling ----
BLOCK_M = C_OUT  # 32
BLOCK_N = OUT_W  # 16
BLOCK_K = C_IN  # 16
GX = OUT_DHW // BLOCK_N  # 16 blocks (one output row each)

# Toggle the in-kernel weight re-layout (Vector prelude + cross-scope flag).
# The 3-D im2col + gemm hard part is exercised either way; the prelude is the
# Conv1d R3 pattern re-validated on the 3-D weight volume.
IN_KERNEL_LAYOUT = True


@tilelang.jit(out_idx=[3], target="npuir")
def build_conv3d_gemm():
    """Pure-Cube 3-D implicit GEMM; L0C -> GM (fp32)."""

    @T.prim_func
    def main(
        x_pad: T.Tensor((N_BATCH, C_IN, DP, HP, WP), DTYPE),
        weight: T.Tensor((C_OUT, C_IN, KV), DTYPE),
        w_flat: T.Tensor((KV, C_OUT, C_IN), DTYPE),
        out: T.Tensor((C_OUT, OUT_DHW), ACCUM),
    ):
        with T.Kernel(GX, is_npu=True) as (cid, _):
            row = cid  # one output (od, oh) row per block
            od = row // OUT_H
            oh = row % OUT_H

            if IN_KERNEL_LAYOUT:
                with T.Scope("Vector"):
                    wsrc = T.alloc_ub([C_OUT, C_IN, KV], DTYPE)
                    wt = T.alloc_ub([C_OUT, KV, C_IN], DTYPE)
                    wdst = T.alloc_ub([KV, C_OUT, C_IN], DTYPE)
                    # weight [C_out, C_in, KV] -> w_flat [KV, C_out, C_in]
                    T.copy(weight, wsrc)
                    T.transpose(wsrc, wt, permutation=[0, 2, 1])
                    T.transpose(wt, wdst, permutation=[1, 0, 2])
                    for tap in T.serial(KV):
                        T.copy(
                            wdst[tap, 0:C_OUT, 0:C_IN],
                            w_flat[tap, 0:C_OUT, 0:C_IN],
                        )
                    with T.rs("PIPE_MTE3"):
                        T.sync_block_set(4)  # weight re-layout ready (Vector -> Cube)

            with T.Scope("Cube"):
                l1_a = T.alloc_L1([BLOCK_M, BLOCK_K], DTYPE)
                l1_b = T.alloc_L1([BLOCK_K, BLOCK_N], DTYPE)
                l0_c = T.alloc_L0C([BLOCK_M, BLOCK_N], ACCUM)

                if IN_KERNEL_LAYOUT:
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(4)  # wait weight re-layout

                for tap in T.serial(KV):
                    kd = tap // (KH * KW)
                    kh = (tap // KW) % KH
                    kw = tap % KW
                    id_ = od + kd  # padded-space index (i - PAD + PAD)
                    ih = oh + kh
                    with T.rs("PIPE_MTE2"):
                        T.copy(
                            w_flat[tap, 0:BLOCK_M, 0:BLOCK_K],
                            l1_a[0:BLOCK_M, 0:BLOCK_K],
                        )
                        # im2col tap slice: 5-D x_pad -> 2-D [C_in, BLOCK_N]
                        # region, row stride DP*HP, contiguous in W, shifted kw.
                        T.copy(
                            x_pad[0, 0:C_IN, id_, ih, kw : kw + BLOCK_N],
                            l1_b[0:BLOCK_K, 0:BLOCK_N],
                        )
                    with T.rs("PIPE_C"):
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=(tap == 0),
                            size=[BLOCK_M, BLOCK_K, BLOCK_N],
                        )

                with T.rs("PIPE_FIX"):
                    T.copy(
                        l0_c,
                        out[0:BLOCK_M, row * BLOCK_N : row * BLOCK_N + BLOCK_N],
                    )

    return main


@tilelang.jit(out_idx=[3], target="npuir")
def build_conv3d_gemm_bias():
    """MixCV 3-D implicit GEMM with bias: Cube -> ws -> Vector (+bias, cast)."""

    @T.prim_func
    def main(
        x_pad: T.Tensor((N_BATCH, C_IN, DP, HP, WP), DTYPE),
        weight: T.Tensor((C_OUT, C_IN, KV), DTYPE),
        bias: T.Tensor((C_OUT,), DTYPE),
        out: T.Tensor((C_OUT, OUT_DHW), DTYPE),
        w_flat: T.Tensor((KV, C_OUT, C_IN), DTYPE),
        ws: T.Tensor((GX, BLOCK_M, BLOCK_N), ACCUM),
    ):
        with T.Kernel(GX, is_npu=True) as (cid, _):
            row = cid
            od = row // OUT_H
            oh = row % OUT_H

            if IN_KERNEL_LAYOUT:
                with T.Scope("Vector"):
                    wsrc = T.alloc_ub([C_OUT, C_IN, KV], DTYPE)
                    wt = T.alloc_ub([C_OUT, KV, C_IN], DTYPE)
                    wdst = T.alloc_ub([KV, C_OUT, C_IN], DTYPE)
                    T.copy(weight, wsrc)
                    T.transpose(wsrc, wt, permutation=[0, 2, 1])
                    T.transpose(wt, wdst, permutation=[1, 0, 2])
                    for tap in T.serial(KV):
                        T.copy(
                            wdst[tap, 0:C_OUT, 0:C_IN],
                            w_flat[tap, 0:C_OUT, 0:C_IN],
                        )
                    with T.rs("PIPE_MTE3"):
                        T.sync_block_set(4)  # weight re-layout ready (Vector -> Cube)

            with T.Scope("Cube"):
                l1_a = T.alloc_L1([BLOCK_M, BLOCK_K], DTYPE)
                l1_b = T.alloc_L1([BLOCK_K, BLOCK_N], DTYPE)
                l0_c = T.alloc_L0C([BLOCK_M, BLOCK_N], ACCUM)

                if IN_KERNEL_LAYOUT:
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(4)

                for tap in T.serial(KV):
                    kd = tap // (KH * KW)
                    kh = (tap // KW) % KH
                    kw = tap % KW
                    id_ = od + kd
                    ih = oh + kh
                    with T.rs("PIPE_MTE2"):
                        T.copy(
                            w_flat[tap, 0:BLOCK_M, 0:BLOCK_K],
                            l1_a[0:BLOCK_M, 0:BLOCK_K],
                        )
                        T.copy(
                            x_pad[0, 0:C_IN, id_, ih, kw : kw + BLOCK_N],
                            l1_b[0:BLOCK_K, 0:BLOCK_N],
                        )
                    with T.rs("PIPE_C"):
                        T.gemm(
                            l1_a,
                            l1_b,
                            l0_c,
                            initC=(tap == 0),
                            size=[BLOCK_M, BLOCK_K, BLOCK_N],
                        )

                # relay the fp32 accumulator tile to Vector through GM
                with T.rs("PIPE_FIX"):
                    T.copy(l0_c, ws[row, 0:BLOCK_M, 0:BLOCK_N])
                    T.sync_block_set(5)  # ws ready (Cube -> Vector)

            with T.Scope("Vector"):
                ub_f32 = T.alloc_ub([BLOCK_M, BLOCK_N], ACCUM)
                ub_f16 = T.alloc_ub([BLOCK_M, BLOCK_N], DTYPE)
                bias_16 = T.alloc_ub([BLOCK_M, 1], DTYPE)
                bias_col = T.alloc_ub([BLOCK_M, 1], ACCUM)
                bias_brc = T.alloc_ub([BLOCK_M, BLOCK_N], ACCUM)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(5)  # ws ready
                    T.copy(ws[row, 0:BLOCK_M, 0:BLOCK_N], ub_f32[0:BLOCK_M, 0:BLOCK_N])
                    T.copy(bias[0:BLOCK_M], bias_16[0:BLOCK_M, 0:1])
                T.vcast(bias_16, bias_col, round_mode="rint")
                T.vbrc(bias_col, bias_brc)
                T.vadd(bias_brc, ub_f32, ub_f32)  # fp32-domain bias add
                T.vcast(ub_f32, ub_f16, round_mode="rint")
                with T.rs("PIPE_MTE3"):
                    T.copy(
                        ub_f16[0:BLOCK_M, 0:BLOCK_N],
                        out[0:BLOCK_M, row * BLOCK_N : row * BLOCK_N + BLOCK_N],
                    )

    return main


def _report(tag, real, golden):
    diff = (real - golden).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = golden.abs().clamp_min(1e-3)
    max_rel = float((diff / denom).max().item())
    print(f"[{tag}] max_abs_err = {max_abs:.6e}")
    print(f"[{tag}] mean_abs_err= {mean_abs:.6e}")
    print(f"[{tag}] max_rel_err = {max_rel:.6e}")
    print(f"[{tag}] golden absmax = {float(golden.abs().max().item()):.4f}")
    return max_abs


def main():
    print("tilelang", tilelang.__version__, "target=npuir mode=Expert")
    torch.manual_seed(0)

    x = torch.randn(N_BATCH, C_IN, D, H, W, dtype=torch.float16).contiguous()
    weight = torch.randn(C_OUT, C_IN, KD, KH, KW, dtype=torch.float16).contiguous()
    bias = torch.randn(C_OUT, dtype=torch.float16).contiguous()

    # independent pure-torch CPU golden (padding=1, stride=1, dilation=1)
    golden = torch.nn.functional.conv3d(
        x.float(), weight.float(), padding=PAD
    )[0].reshape(C_OUT, OUT_DHW)
    golden_bias = torch.nn.functional.conv3d(
        x.float(), weight.float(), bias.float(), padding=PAD
    )[0].reshape(C_OUT, OUT_DHW)

    # zero-padded 5-D staging; the pre-pad moves every im2col tap in-bounds.
    x_pad = torch.nn.functional.pad(x.float(), (PAD, PAD, PAD, PAD, PAD, PAD)).to(
        torch.float16
    ).contiguous().to("npu")
    # in-kernel layout consumes weight viewed as [C_out, C_in, KV]; the view is a
    # zero-copy reshape of the contiguous 5-D weight.
    weight_kv = weight.reshape(C_OUT, C_IN, KV).contiguous().to("npu")

    # ---------- A. no-bias (pure Cube) ----------
    print("building no-bias kernel ...")
    t0 = time.perf_counter()
    k_a = build_conv3d_gemm()
    print(f"  compiled OK in {time.perf_counter() - t0:.1f}s")
    w_flat = torch.empty((KV, C_OUT, C_IN), dtype=torch.float16, device="npu")
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out_a = k_a(x_pad, weight_kv, w_flat)
    torch.npu.synchronize()
    print(f"  ran OK in {(time.perf_counter() - t0) * 1e3:.1f} ms; "
          f"out shape={tuple(out_a.shape)} dtype={out_a.dtype}")
    max_a = _report("A-no-bias", out_a.cpu().float(), golden)

    # ---------- B. with-bias (MixCV) ----------
    print("building bias kernel ...")
    t0 = time.perf_counter()
    k_b = build_conv3d_gemm_bias()
    print(f"  compiled OK in {time.perf_counter() - t0:.1f}s")
    w_flat_b = torch.empty((KV, C_OUT, C_IN), dtype=torch.float16, device="npu")
    ws = torch.empty((GX, BLOCK_M, BLOCK_N), dtype=torch.float32, device="npu")
    bias_npu = bias.to("npu")
    torch.npu.synchronize()
    t0 = time.perf_counter()
    out_b = k_b(x_pad, weight_kv, bias_npu, w_flat_b, ws)
    torch.npu.synchronize()
    print(f"  ran OK in {(time.perf_counter() - t0) * 1e3:.1f} ms; "
          f"out shape={tuple(out_b.shape)} dtype={out_b.dtype}")
    max_b = _report("B-bias", out_b.cpu().float(), golden_bias)

    ok = max_a < 5e-2 and max_b < 5e-2
    print(f"[PROBE] no-bias max_abs_err={max_a:.3e} bias max_abs_err={max_b:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("[PROBE_FAIL] exception during build/run:")
        traceback.print_exc()
        raise SystemExit(2)
