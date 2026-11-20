# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""1x1 (pointwise) Conv1d forward kernel for npuir (Cube Expert / MixCV).

Migrated from the GPU TileLang factory ``_conv1d_pointwise_kernel``
(``TileOPs-main/src/tileops/kernels/convolution/conv1d.py``) to
``target="npuir"``, ``ascend_mode="Expert"`` (``T.Scope("Cube")`` +
``T.alloc_L1`` / ``T.alloc_L0C`` / ``T.gemm``), per DESIGN.md v1 and the
Stage-2 REVIEW.md.

Math (source semantics, kept verbatim, DESIGN §1.3):

    out[n, c_o, l] = bias[c_o] + sum_{c_i} W[c_o, c_i] * x[n, c_i, l]
    W = weight[:, :, 0]   (kW=1),   L_out = L_in

Structure (DESIGN §0.6 / §3.3 / §6):
  - 1-D persistent ``T.Kernel(num_kernels)`` with ``num_kernels =
    NPUUtils.get().get_aicore_num()`` (verified = 20 on Ascend 910B3);
    logical task = (bz, by, bx) round-robin over ``num_local_tasks``; ghost
    cores clamp ``cid`` to the last legal task (gqa precedent).
  - Implicit GEMM: A = W[C_out, C_in] (M x K), B = x[C_in, L] (K x N),
    K blocked fp32 accumulation via ``T.gemm(..., initC=(k==0))``; slice-form
    ``T.copy`` performs GM->L1 auto ND2NZ (avoids the strided
    ``T.load_nd2nz`` misread, CG-2026-0004).
  - Tail handling (R2): ``T.copy`` moves only the valid region (dynamic slice
    extents); the gemm ``size`` uses the fractal clamp
    ``T.max(16, ceil16(tail))``; the output is written with a clipped slice.
  - ``has_bias=False``: pure Cube (single Scope), L0C -> GM fixpipe
    (fp32 -> fp16/bf16 quantize).
  - ``has_bias=True``: MixCV.  Cube relays the fp32 L0C tile through the GM
    workspace ``ws[kernel_id, slot]`` (fp32, no quantize); Vector loads it,
    adds ``bias`` in fp32 (``vcast`` -> ``vbrc`` -> ``vadd``), casts back to
    ``dtype`` (``vcast``) and stores to ``out``.  The Cube<->Vector relay uses
    a depth-2 double-buffer with a 4-flag bidirectional handshake:
      flag 2*slot   = ws ready   (producer Cube  set  / consumer Vector wait)
      flag 2*slot+1 = slot free  (consumer Vector set  / producer Cube  wait)
    The "slot free" prologue is executed by the CONSUMER (Vector) side, which
    mirrors the verified PL-1.12 pattern in ``_ssd_chunk_scan_fwd_kernel.py``
    (a same-core producer self set->wait does not retire: it deadlocks).

Run: python _conv1d_pointwise_kernel.py --level {L0,all}
"""

import argparse
import os

# Expert mode is required for the persistent kernel + Cube gemm (+ Vector
# epilogue) structure; set before importing tilelang.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
import tilelang
import tilelang.language as T
from tilelang.utils import NPUUtils

# Stage-3 verified programming mode (must match TILELANG_ASCEND_MODE above).
ASCEND_MODE = "Expert"

# Tiling (DESIGN §5.2, inherited from the GPU Conv1dPointwiseKernel
# default_config).
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
SLOTS = 2

_ACCUM = "float32"


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_conv1d_pointwise(x, weight, bias=None):
    """Pointwise Conv1d reference (kW=1, stride=1, pad=0, dilation=1).

    Based on the source semantics only (DESIGN §0.1 / §8.1): ``weight`` is
    ``[C_out, C_in, 1]``; computed in fp32 then rounded to the input dtype.
    Independent of the NPU Cube/Vector algorithm.
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    out_f32 = torch.nn.functional.conv1d(
        x_cpu.to(torch.float32),
        w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=1,
        padding=0,
        dilation=1,
    )
    return out_f32.to(x_cpu.dtype)


# Alias so the module exposes a literal ``golden`` as well.
golden = golden_conv1d_pointwise


# ---------- Kernel ----------
def _conv1d_pointwise_kernel(n, c_in, l_in, c_out, has_bias, dtype="float16"):
    """NPU factory mirroring the GPU ``_conv1d_pointwise_kernel`` builder.

    Returns an inner ``@tilelang.jit(target="npuir")`` factory taking the NPU
    tiling parameters (the GPU-only ``threads`` / ``enable_rasterization``
    parameters are dropped).
    """
    accum_dtype = _ACCUM
    out_l = l_in  # kW=1, stride=1, pad=0, dilation=1
    num_kernels = NPUUtils.get().get_aicore_num()

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv1d_pointwise_func(block_m, block_n, block_k, num_stages):
        gx = (out_l + block_n - 1) // block_n
        gy = (c_out + block_m - 1) // block_m
        num_logical = gy * gx * n

        if has_bias:

            @T.prim_func
            def _pointwise_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, c_in), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                ws: T.Tensor((num_kernels, SLOTS, block_m, block_n), accum_dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                    num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                    with T.Scope("Cube"):
                        l1_a = T.alloc_L1([block_m, block_k], dtype)
                        l1_b = T.alloc_L1([block_k, block_n], dtype)
                        l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)
                        for task_id in T.serial(num_local_tasks):
                            cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                            bz = cid // (gy * gx)
                            by = (cid // gx) % gy
                            bx = cid % gx
                            tm = T.min(block_m, c_out - by * block_m)
                            tn = T.min(block_n, out_l - bx * block_n)
                            tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                            tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                            for k in T.serial(T.ceildiv(c_in, block_k)):
                                tk = T.min(block_k, c_in - k * block_k)
                                tkc = T.max(16, T.ceildiv(tk, 16) * 16)
                                with T.rs("PIPE_MTE2"):
                                    T.copy(
                                        weight[
                                            by * block_m : by * block_m + tm,
                                            k * block_k : k * block_k + tk,
                                        ],
                                        l1_a[0:tm, 0:tk],
                                    )
                                    T.copy(
                                        x[
                                            bz,
                                            k * block_k : k * block_k + tk,
                                            bx * block_n : bx * block_n + tn,
                                        ],
                                        l1_b[0:tk, 0:tn],
                                    )
                                with T.rs("PIPE_C"):
                                    T.gemm(
                                        l1_a,
                                        l1_b,
                                        l0_c,
                                        initC=(k == 0),
                                        b_transpose=False,
                                        size=[tmc, tkc, tnc],
                                    )
                            slot = task_id % SLOTS
                            with T.rs("PIPE_FIX"):
                                T.sync_block_wait(2 * slot + 1)  # slot free (WAR)
                                T.copy(
                                    l0_c[0:tm, 0:tn],
                                    ws[kernel_id, slot, 0:tm, 0:tn],
                                )
                                T.sync_block_set(2 * slot)  # ws ready
                    with T.Scope("Vector"):
                        ub_f32 = T.alloc_ub([block_m, block_n], accum_dtype)
                        ub_f16 = T.alloc_ub([block_m, block_n], dtype)
                        bias_16 = T.alloc_ub([block_m, 1], dtype)
                        bias_col = T.alloc_ub([block_m, 1], accum_dtype)
                        bias_brc = T.alloc_ub([block_m, block_n], accum_dtype)
                        # Depth-2 prologue: both slots start free.  Executed by
                        # the CONSUMER (Vector) so the producer (Cube) first
                        # waits across cores retire (PL-1.12 / ssd precedent).
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_set(1)
                            T.sync_block_set(3)
                        for task_id in T.serial(num_local_tasks):
                            cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                            bz = cid // (gy * gx)
                            by = (cid // gx) % gy
                            bx = cid % gx
                            tm = T.min(block_m, c_out - by * block_m)
                            tn = T.min(block_n, out_l - bx * block_n)
                            slot = task_id % SLOTS
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(2 * slot)  # ws ready
                                T.copy(
                                    ws[kernel_id, slot, 0:tm, 0:tn],
                                    ub_f32[0:tm, 0:tn],
                                )
                                T.copy(
                                    bias[by * block_m : by * block_m + tm],
                                    bias_16[0:tm, 0:1],
                                )
                            T.vcast(bias_16, bias_col, round_mode="rint")
                            T.vbrc(bias_col, bias_brc)
                            T.vadd(bias_brc, ub_f32, ub_f32)  # fp32-domain bias add
                            T.vcast(ub_f32, ub_f16, round_mode="rint")
                            with T.rs("PIPE_MTE3"):
                                T.copy(
                                    ub_f16[0:tm, 0:tn],
                                    out[
                                        bz,
                                        by * block_m : by * block_m + tm,
                                        bx * block_n : bx * block_n + tn,
                                    ],
                                )
                                T.sync_block_set(2 * slot + 1)  # slot free

            return _pointwise_bias_main

        @T.prim_func
        def _pointwise_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, c_in), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
                num_local_tasks = T.ceildiv(num_logical - kernel_id, num_kernels)
                with T.Scope("Cube"):
                    l1_a = T.alloc_L1([block_m, block_k], dtype)
                    l1_b = T.alloc_L1([block_k, block_n], dtype)
                    l0_c = T.alloc_L0C([block_m, block_n], accum_dtype)
                    for task_id in T.serial(num_local_tasks):
                        cid = T.min(task_id * num_kernels + kernel_id, num_logical - 1)
                        bz = cid // (gy * gx)
                        by = (cid // gx) % gy
                        bx = cid % gx
                        tm = T.min(block_m, c_out - by * block_m)
                        tn = T.min(block_n, out_l - bx * block_n)
                        tmc = T.max(16, T.ceildiv(tm, 16) * 16)
                        tnc = T.max(16, T.ceildiv(tn, 16) * 16)
                        for k in T.serial(T.ceildiv(c_in, block_k)):
                            tk = T.min(block_k, c_in - k * block_k)
                            tkc = T.max(16, T.ceildiv(tk, 16) * 16)
                            with T.rs("PIPE_MTE2"):
                                T.copy(
                                    weight[
                                        by * block_m : by * block_m + tm,
                                        k * block_k : k * block_k + tk,
                                    ],
                                    l1_a[0:tm, 0:tk],
                                )
                                T.copy(
                                    x[
                                        bz,
                                        k * block_k : k * block_k + tk,
                                        bx * block_n : bx * block_n + tn,
                                    ],
                                    l1_b[0:tk, 0:tn],
                                )
                            with T.rs("PIPE_C"):
                                T.gemm(
                                    l1_a,
                                    l1_b,
                                    l0_c,
                                    initC=(k == 0),
                                    b_transpose=False,
                                    size=[tmc, tkc, tnc],
                                )
                        with T.rs("PIPE_FIX"):
                            T.copy(
                                l0_c[0:tm, 0:tn],
                                out[
                                    bz,
                                    by * block_m : by * block_m + tm,
                                    bx * block_n : bx * block_n + tn,
                                ],
                            )

        return _pointwise_main

    return _conv1d_pointwise_func


# ---------- Precision comparison ----------
_TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (1e-4, 1e-4),
}


def _dtype_str(dtype):
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[dtype]


def _run_case(n, c_in, l_in, c_out, has_bias, dtype, tag, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=dtype)
    weight = torch.randn(c_out, c_in, 1, dtype=dtype)
    bias = torch.randn(c_out, dtype=dtype) if has_bias else None

    fn = _conv1d_pointwise_kernel(n, c_in, l_in, c_out, has_bias, _dtype_str(dtype))
    kernel = fn(BLOCK_M, BLOCK_N, BLOCK_K, 1)

    x_npu = x.to("npu")
    w_npu = weight[:, :, 0].contiguous().to("npu")
    if has_bias:
        ws = torch.empty(
            (NPUUtils.get().get_aicore_num(), SLOTS, BLOCK_M, BLOCK_N),
            dtype=torch.float32,
            device="npu",
        )
        out = kernel(x_npu, w_npu, bias.to("npu"), ws)
    else:
        out = kernel(x_npu, w_npu)
    torch.npu.synchronize()

    ref = golden_conv1d_pointwise(x, weight, bias)
    atol, rtol = _TOL[dtype]
    assert tuple(out.shape) == (n, c_out, l_in), f"shape {tuple(out.shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype}"
    max_abs = float((out.cpu().float() - ref.float()).abs().max().item())
    try:
        torch.testing.assert_close(out.cpu(), ref, atol=atol, rtol=rtol)
        ok = True
    except AssertionError as exc:  # noqa: BLE001
        print(f"  [{tag}] FAIL: {str(exc).splitlines()[0]}")
        ok = False
    print(
        f"  [{tag}] n={n} c_in={c_in} l_in={l_in} c_out={c_out} "
        f"has_bias={has_bias} {_dtype_str(dtype)}: "
        f"{'PASS' if ok else 'FAIL'} (max_abs={max_abs:.3e})"
    )
    return ok, max_abs


# ---------- hierarchical testing ----------
def run_L0():
    """L0 gate (blocking): representative workloads, both paths."""
    print("== L0 (blocking) ==")
    ok = True
    # whisper-1x1 (DESIGN §8.3)
    ok &= _run_case(1, 80, 3000, 1280, False, torch.float16, "L0-whisper-nobias", seed=0)[0]
    ok &= _run_case(1, 80, 3000, 1280, True, torch.float16, "L0-whisper-bias", seed=1)[0]
    # encodec-deep-1x1 (DESIGN §8.3)
    ok &= _run_case(1, 128, 600, 256, False, torch.float16, "L0-encodec-nobias", seed=2)[0]
    ok &= _run_case(1, 128, 600, 256, True, torch.float16, "L0-encodec-bias", seed=3)[0]
    # non-divisible C_out / L / C_in (DESIGN §8.3 must-test 1)
    ok &= _run_case(1, 50, 1000, 100, False, torch.float16, "L0-nondiv-nobias", seed=4)[0]
    ok &= _run_case(1, 50, 1000, 100, True, torch.float16, "L0-nondiv-bias", seed=5)[0]
    # batch > 1 (DESIGN §8.3 must-test 3)
    ok &= _run_case(3, 96, 300, 192, True, torch.float16, "L0-batch3-bias", seed=6)[0]
    assert ok, "L0 FAILED (blocking)"
    print("[L0] PASS (7 cases)")


def run_L1():
    """L1 gate (blocking): dtypes f16/bf16, both paths, more shapes."""
    print("== L1 (blocking) ==")
    ok = True
    cases = [
        (1, 80, 3000, 1280, "whisper"),
        (1, 128, 600, 256, "encodec"),
        (1, 50, 1000, 100, "nondiv"),
        (2, 64, 512, 128, "batch2"),
        (1, 32, 256, 192, "small"),
    ]
    seed = 100
    for n, c_in, l_in, c_out, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"L1-{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                ok &= _run_case(n, c_in, l_in, c_out, has_bias, dtype, tag, seed=seed)[0]
                seed += 1
    assert ok, "L1 FAILED (blocking)"
    print("[L1] PASS (20 cases)")


def run_L2():
    """L2 (warn-only): larger / odd shapes."""
    print("== L2 (warn-only) ==")
    cases = [
        (4, 128, 1024, 512, "L2-batch4"),
        (1, 256, 2048, 640, "L2-deep"),
        (1, 16, 2048, 64, "L2-c16"),
        (2, 80, 777, 333, "L2-odd"),
    ]
    n_pass = 0
    for n, c_in, l_in, c_out, name in cases:
        for dtype in (torch.float16, torch.bfloat16):
            for has_bias in (False, True):
                tag = f"{name}-{'bias' if has_bias else 'nobias'}-{_dtype_str(dtype)}"
                try:
                    ok, _ = _run_case(n, c_in, l_in, c_out, has_bias, dtype, tag, seed=200)
                    n_pass += 1 if ok else 0
                except Exception as e:  # noqa: BLE001 - warn-only layer
                    print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases) * 4
    if n_pass == total:
        print(f"[L2] PASS ({total} cases)")
    else:
        print(f"[L2] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


def run_boundary():
    """Boundary (warn-only): tiny / degenerate valid shapes."""
    print("== Boundary (warn-only) ==")
    cases = [
        (1, 16, 16, 16),  # minimum fractal-aligned tile
        (1, 20, 40, 48),  # non-16-multiple channel dims
        (1, 48, 128, 64),  # exact single tile
        (1, 17, 33, 33),  # all non-divisible / non-aligned
    ]
    n_pass = 0
    for n, c_in, l_in, c_out in cases:
        for has_bias in (False, True):
            tag = f"B-{n}x{c_in}x{l_in}x{c_out}-{'bias' if has_bias else 'nobias'}"
            try:
                ok, _ = _run_case(n, c_in, l_in, c_out, has_bias, torch.float16, tag, seed=300)
                n_pass += 1 if ok else 0
            except Exception as e:  # noqa: BLE001 - warn-only layer
                print(f"  [{tag}] WARN (exception): {e}")
    total = len(cases) * 2
    if n_pass == total:
        print(f"[Boundary] PASS ({total} cases)")
    else:
        print(f"[Boundary] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
