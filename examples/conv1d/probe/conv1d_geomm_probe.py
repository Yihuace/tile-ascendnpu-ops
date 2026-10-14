# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Feasibility probe: GEMM-based Conv1d on npuir (Cube path), not a migration.

Question: can the canonical implicit-GEMM conv

    output[C_out, L_out] = W[C_out, C_in*kW] @ im2col[C_in*kW, L_out]

be expressed (with a conv-specific, strided/overlapping im2col stage), compiled
and run on the npuir Cube path (target="npuir", Expert mode: T.alloc_L1 /
T.alloc_L0C / T.gemm / T.copy)?

Probe workload = smallest real manifest workload ``whisper-conv1``:
    N=1, C_in=80, L_in=3000, C_out=1280, kW=3, stride=1, pad=0, dilation=1
    -> L_out = 2998, fp16.

Dim adjustments (stated per probe contract):
  * L_out=2998 is not a multiple of the fractal N (16) nor of BLOCK_N=64, so
    the GEMM N axis is padded to N_COLS=ceil16(2998)=3008=64*47 (exact
    tiling). Only the real 2998 columns are compared.
  * To make every padded output column a legal convolution (no OOB read of
    x[l+kw], l<3008, kw<3 -> max index 3009), the input is right-padded with
    12 zero columns to L_PAD=3012.
  * K = C_in*kW = 240 (multiple of 16). M = C_out = 1280.

im2col staging (the conv-specific access): for tap kw, rows
[kw*C_in : (kw+1)*C_in) of the B tile are x[:, l0+kw : l0+kw+BLOCK_N] -- three
overlapping, strided windows of x (row stride = L_PAD). Per TRAP-load-nd2nz-
strided, T.load_nd2nz silently flat-misreads strided GM regions, so the
slice-form T.copy (the documented, verified workaround; equals T.load_nd2nz in
Expert GM->L1) is used.

Golden = torch.nn.functional.conv1d (CPU), compared in fp32.

Run: python conv1d_geomm_probe.py
"""

import os
import time
import traceback

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T

# ---- workload (whisper-conv1) ----
N_BATCH = 1
C_IN = 80
L_IN = 3000
C_OUT = 1280
KW = 3
STRIDE = 1
PAD = 0
L_OUT = 2998
K_TOTAL = C_IN * KW  # 240

# ---- layout / tiling ----
N_COLS = ((L_OUT + 15) // 16) * 16  # 3008
L_PAD = L_IN + 12  # 3012 (>= (N_COLS-1) + KW-1 + 1 = 3009)
BLOCK_M = 128
BLOCK_N = 64
DTYPE = "float16"
ACCUM = "float32"

GX = N_COLS // BLOCK_N  # 47
GY = C_OUT // BLOCK_M  # 10


@tilelang.jit(out_idx=[-1], target="npuir")
def build_conv_gemm():
    @T.prim_func
    def main(
        Wf: T.Tensor((C_OUT, K_TOTAL), DTYPE),
        xp: T.Tensor((C_IN, L_PAD), DTYPE),
        out: T.Tensor((C_OUT, N_COLS), ACCUM),
    ):
        with T.Kernel(GY * GX, is_npu=True) as (cid, _):
            by = cid // GX  # M tile
            bx = cid % GX  # N tile
            with T.Scope("Cube"):
                l1_a = T.alloc_L1([BLOCK_M, K_TOTAL], DTYPE)
                l1_b = T.alloc_L1([K_TOTAL, BLOCK_N], DTYPE)
                l0_c = T.alloc_L0C([BLOCK_M, BLOCK_N], ACCUM)

                # A tile: contiguous rows of the flattened weight.
                T.copy(
                    Wf[by * BLOCK_M : by * BLOCK_M + BLOCK_M, 0:K_TOTAL],
                    l1_a,
                )

                # B tile (im2col): three overlapping, strided tap rectangles.
                for kw in range(KW):
                    T.copy(
                        xp[0:C_IN, bx * BLOCK_N + kw : bx * BLOCK_N + kw + BLOCK_N],
                        l1_b[kw * C_IN : (kw + 1) * C_IN, 0:BLOCK_N],
                    )

                T.gemm(l1_a, l1_b, l0_c, initC=True, size=[BLOCK_M, K_TOTAL, BLOCK_N])

                T.copy(
                    l0_c,
                    out[
                        by * BLOCK_M : by * BLOCK_M + BLOCK_M,
                        bx * BLOCK_N : bx * BLOCK_N + BLOCK_N,
                    ],
                )

    return main


def main():
    print("tilelang", tilelang.__version__, "target=npuir mode=Expert")
    torch.manual_seed(0)

    x = torch.randn(N_BATCH, C_IN, L_IN, dtype=torch.float16).contiguous()
    weight = torch.randn(C_OUT, C_IN, KW, dtype=torch.float16).contiguous()

    # golden (CPU, fp32 accumulation of exact fp16 inputs)
    golden = torch.nn.functional.conv1d(x.float(), weight.float())[0]  # [C_out, L_out]

    # W_flat[m, kw*C_in + ci] = weight[m, ci, kw]
    wf = weight.permute(0, 2, 1).reshape(C_OUT, K_TOTAL).contiguous()
    # right-pad input
    xp = torch.zeros(C_IN, L_PAD, dtype=torch.float16)
    xp[:, :L_IN] = x[0]

    print("building kernel ...")
    t0 = time.perf_counter()
    kernel = build_conv_gemm()
    t_build = time.perf_counter() - t0
    print(f"  compiled OK in {t_build:.1f}s")

    wf_npu = wf.to("npu")
    xp_npu = xp.contiguous().to("npu")
    torch.npu.synchronize()

    t0 = time.perf_counter()
    out = kernel(wf_npu, xp_npu)
    torch.npu.synchronize()
    t_run = (time.perf_counter() - t0) * 1e3
    print(f"  ran OK in {t_run:.1f} ms; out shape={tuple(out.shape)} dtype={out.dtype}")

    real = out.cpu().float()[:, :L_OUT]
    diff = (real - golden).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = golden.abs().clamp_min(1e-3)
    max_rel = float((diff / denom).max().item())

    print(f"  max_abs_err = {max_abs:.6e}")
    print(f"  mean_abs_err= {mean_abs:.6e}")
    print(f"  max_rel_err = {max_rel:.6e}")
    print(f"  golden absmax = {float(golden.abs().max().item()):.4f}")

    ok = max_abs < 5e-2
    print(f"[PROBE] {'' if ok else 'PARTIAL/FAIL: '}max_abs_err={max_abs:.3e}")
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
