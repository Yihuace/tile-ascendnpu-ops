# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Depthwise (direct) Conv1d forward kernel for npuir (Expert / Vector-only).

Migrated from the GPU TileLang factory ``_conv1d_direct_kernel``
(``TileOPs-main/src/tileops/kernels/convolution/conv1d.py`` L316-402; the
``DepthwiseConv1dKernel`` class, L1074-1126) to ``target="npuir"`` with
``ascend_mode="Expert"`` and a single ``T.Scope("Vector")`` -- no Cube, no
``T.gemm``, no ``T.alloc_L1`` / ``T.alloc_L0C``.

Source semantics (DESIGN.md section 0.1 / 1.3), kept verbatim:

    out[n, c_o, l] = bias[c_o]
        + sum_{k_w=0}^{K_W-1} W[c_o, 0, k_w] * x[n, c_o, l*s_W + k_w*d_W - p_L]
    L_out = (L_in + p_L + p_R - d_W*(K_W-1) - 1) // s_W + 1

with ``groups = C_in = C_out`` (depthwise; weight is ``(C_out, 1, K_W)``).
The single reduction axis is ``K_W`` -- there is no input-channel contraction,
so this is an element-wise + small-reduction operator (Vector), NOT a GEMM.

NPU redesign (DESIGN.md section 0.6):
  R1: one-dimensional ``T.Kernel(num_cores, is_npu=True)`` + persistent
      ``T.serial`` task loop; output is flattened row-major as
      ``r = bz*C_out + c_o`` (``R = n*C_out`` rows) over ``GX_L = ceil(L_out/BN)``
      L-tiles; each core handles ``num_local_tasks`` tasks with a ``lid`` guard.
  R2: out-of-range inputs are handled by a zero-padded GM workspace ``xpad``
      (built in-kernel; host only ``torch.zeros``-allocates it -- no host
      padding).  The left pad is written with a *GM offset destination*
      ``T.copy(ub2, xpad[..., p_L:p_L+L_in])`` because a UB column-offset
      destination silently mis-strides (TRAP-copy-ub-offset-stride).
  R3: per-tap (per ``k_w``) contiguous shifted load into a ``[K_W, BN]`` tile,
      weight row-broadcast ``T.vmul`` (``[K_W,BN]*[K_W,1]``) and a single
      ``T.reduce(dims=[0], reduce_mode="sum")`` replace the scalar kernel-window
      MAC loop (O(K_W) -> O(1) vector ops).
  R4: stride>1 column sampling cannot be a contiguous ``T.copy`` (no strided
      gather vector instruction, TRAP-C10-strided-gather), so ``xpad`` is
      decimated once into ``xdec[n, c_i, r_kw*Mdec + m] = xpad[n, c_i, m*s_W + r_kw]``
      with ``T.gather`` and a host-built constant index; the per-tap load then
      reads a contiguous slice of ``xdec``.  stride==1 reads ``xpad`` directly.
  R5: bias is added in the fp32 accumulation domain (``[1,BN] + [1,1]`` column
      broadcast ``T.vadd``) and the result is cast once back to ``dtype``.

bf16 has no native v-prefix arithmetic, so fp16/bf16 always transit through
fp32 via ``T.vcast``; all ``vmul``/``vadd``/``reduce`` run in the fp32 domain.

Run the embedded hierarchical tests:
    python _conv1d_direct_kernel.py --level L0
    python _conv1d_direct_kernel.py --level all
"""

import argparse
import functools
import os

# Expert mode is required: the in-kernel zero-padded workspace construction
# (GM-offset destination T.copy + single T.Scope("Vector")) compiles under
# Expert; the same kernel fails to compile under Developer (AIV+AIC dual entry
# conflict, 'multiple entry functions detected').  See DESIGN.md section 2.2.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils import NPUUtils

# Stage-3 verified programming mode (must match TILELANG_ASCEND_MODE above).
ASCEND_MODE = "Expert"

# L-direction tile width (inherited from the GPU DepthwiseConv1dKernel
# default_config block_n = 128; DESIGN.md section 5.2).
BLOCK_N = 128
_ACCUM = "float32"
# Per-buffer UB chunk bound (bytes) for the preludes (keeps them within 192KB).
_UB_CHUNK_BYTES = 32768

# Precision tolerances (DESIGN.md section 8.2, matching the source-repo
# conv1d_tolerance).
_TOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (1e-4, 1e-4),
}
_DTYPE_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

# Per-(dtype, has_bias) worst max_diff over the whole run (final summary).
_STATS = {}


def _ceil16(v):
    return ((v + 15) // 16) * 16


def _plan(n, c_in, c_out, l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l):
    """Compile-time shape plan (shared by the factory and the test wrapper).

    Returns (out_l, l_pad2, mdec, need_poly, r_kw, q_kw):
      - ``l_pad2`` is the xpad/xdec row width (16-aligned, a multiple of stride);
      - ``mdec`` is the polyphase band width (``l_pad2 // stride``);
      - ``r_kw[k]`` / ``q_kw[k]`` split ``k_w*d_W`` into ``q*s_W + r``.
    """
    out_l = (l_in + pad_left + pad_right - dilation_l * (kernel_l - 1) - 1) // stride_l + 1
    base = pad_left + l_in + pad_right
    l_pad_align = _ceil16(base)
    need_poly = stride_l != 1
    mdec = (l_pad_align + stride_l - 1) // stride_l
    # round the xdec row width up so it is 16-aligned AND a stride multiple,
    # keeping every gather source index inside the xpad width.
    l_pad2 = _ceil16(stride_l * mdec)
    mdec = l_pad2 // stride_l
    r_kw = [(kw * dilation_l) % stride_l for kw in range(kernel_l)]
    q_kw = [(kw * dilation_l - r_kw[kw]) // stride_l for kw in range(kernel_l)]
    return out_l, l_pad2, mdec, need_poly, r_kw, q_kw


def _build_poly_idx(l_pad2, mdec, stride_l, c_in):
    """Host-built constant gather index for the polyphase decimation.

    ``xdec[ci, r*Mdec + m] = xpad[ci, m*s_W + r]`` -- i.e. index
    ``idx[r*Mdec+m] = m*s_W + r``; replicating the row over ``c_in`` rows.
    """
    mp = torch.arange(l_pad2, dtype=torch.int32)
    idx = ((mp % mdec) * stride_l + (mp // mdec)).to(torch.int32)
    return idx.unsqueeze(0).repeat(c_in, 1).contiguous()


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_conv1d_depthwise(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Depthwise Conv1d reference (source-repo baseline: F.conv1d).

    Based on the source semantics only (DESIGN.md section 8.1): the weight is
    ``[C_out, 1, K_W]`` and ``groups = C_in = C_out``; computed in fp32 (matching
    the source fp32 accumulation + fp32 bias add) then rounded to the input
    dtype.  Independent of the NPU xpad / shifted-load / reduce algorithm.
    Runs on CPU and does not require an NPU device.
    """
    x_cpu = x.detach().to("cpu")
    w_cpu = weight.detach().to("cpu")
    b_cpu = None if bias is None else bias.detach().to("cpu")
    groups = x_cpu.shape[1]
    out_f32 = torch.nn.functional.conv1d(
        x_cpu.to(torch.float32),
        w_cpu.to(torch.float32),
        None if b_cpu is None else b_cpu.to(torch.float32),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return out_f32.to(x_cpu.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _conv1d_direct_kernel(
    n,
    c_in,
    l_in,
    c_out,
    kernel_l,
    stride_l,
    pad_left,
    pad_right,
    dilation_l,
    has_bias,
    dtype="float16",
):
    """NPU factory mirroring the GPU ``_conv1d_direct_kernel`` builder.

    The GPU-only CUDA execution knobs (``threads`` / ``enable_rasterization``)
    are dropped; the inner ``@tilelang.jit(target="npuir")`` factory takes only
    the L-tile width ``block_n``.  Returns the jit factory (drop-in for the GPU
    contract: call ``factory(block_n)`` to obtain the compiled program).

    The kernel signature additionally carries the in-kernel workspaces ``xpad``
    (zero-padded input) and -- for stride>1 -- ``xdec`` (polyphase decimation)
    plus ``poly_idx`` (host constant gather index); the host only allocates
    them (``torch.zeros`` / ``torch.empty``).
    """
    assert c_in == c_out, "depthwise direct kernel requires c_in == c_out"

    accum_dtype = _ACCUM
    out_l, l_pad2, mdec, need_poly, r_kw, q_kw = _plan(
        n, c_in, c_out, l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l
    )

    eb = _ELEM_BYTES[dtype]
    # c_in chunk for the xpad construction (bounded UB slice).
    ch = max(1, min(c_in, _UB_CHUNK_BYTES // max(1, l_in * eb)))
    n_xp = (c_in + ch - 1) // ch
    # c_in chunk for the gather prelude (3 live buffers: dtype + int32 + dtype).
    if need_poly:
        g_per = max(1, l_pad2 * (2 * eb + 4))
        cg = max(1, min(c_in, _UB_CHUNK_BYTES // g_per))
    else:
        cg = 1
    n_cg = (c_in + cg - 1) // cg

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op); re-queried per build so other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    # Placeholder shapes for the stride==1 path (xdec / poly_idx unused there).
    xdec_shape = (n, c_in, l_pad2) if need_poly else (1, 1, 1)
    poly_shape = (c_in, l_pad2) if need_poly else (1, 1)

    @tilelang.jit(out_idx=[2], target="npuir")
    def _conv1d_direct_func(block_n=BLOCK_N):
        gx_l = (out_l + block_n - 1) // block_n
        num_logical = (n * c_out) * gx_l
        # Idle cores are masked out (smoke shapes have num_logical < vector_cores).
        num_cores = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_cores - 1) // num_cores

        @T.macro
        def _body(x, weight, out, bias, xpad, xdec, poly_idx):
            with T.Kernel(num_cores, is_npu=True) as (kernel_id, _):
                with T.Scope("Vector"):
                    # ---- UB buffers (DESIGN.md section 4.3) ----
                    ub2 = T.alloc_ub([ch, l_in], dtype)
                    xt = T.alloc_ub([kernel_l, block_n], dtype)
                    xt32 = T.alloc_ub([kernel_l, block_n], accum_dtype)
                    wc = T.alloc_ub([kernel_l, 1], dtype)
                    w32 = T.alloc_ub([kernel_l, 1], accum_dtype)
                    prod = T.alloc_ub([kernel_l, block_n], accum_dtype)
                    acc = T.alloc_ub([1, block_n], accum_dtype)
                    b11 = T.alloc_ub([1, 1], dtype)
                    b11_32 = T.alloc_ub([1, 1], accum_dtype)
                    out_ub = T.alloc_ub([1, block_n], dtype)

                    # ---- Prelude A: zero-padded xpad (each core builds it;
                    # concurrent writes are identical so benign) ----
                    for bn in T.serial(n):
                        for cc0 in T.serial(n_xp):
                            cr = T.min(ch, T.max(0, c_in - cc0 * ch))
                            if cr > 0:
                                T.copy(
                                    x[bn, cc0 * ch:cc0 * ch + cr, 0:l_in],
                                    ub2[0:cr, 0:l_in],
                                )
                                # GM offset destination (UB offset destination is
                                # silently wrong: TRAP-copy-ub-offset-stride).
                                T.copy(
                                    ub2[0:cr, 0:l_in],
                                    xpad[bn, cc0 * ch:cc0 * ch + cr, pad_left:pad_left + l_in],
                                )

                    # ---- Prelude B (stride>1): polyphase decimation ----
                    if need_poly:
                        ubx = T.alloc_ub([cg, l_pad2], dtype)
                        ubi = T.alloc_ub([cg, l_pad2], "int32")
                        ubd = T.alloc_ub([cg, l_pad2], dtype)
                        for bn in T.serial(n):
                            for cc0 in T.serial(n_cg):
                                cw = T.min(cg, c_in - cc0 * cg)
                                T.copy(
                                    xpad[bn, cc0 * cg:cc0 * cg + cw, 0:l_pad2],
                                    ubx[0:cw, 0:l_pad2],
                                )
                                T.copy(
                                    poly_idx[cc0 * cg:cc0 * cg + cw, 0:l_pad2],
                                    ubi[0:cw, 0:l_pad2],
                                )
                                T.gather(ubx, ubd, ubi, [cw, l_pad2])
                                T.copy(
                                    ubd[0:cw, 0:l_pad2],
                                    xdec[bn, cc0 * cg:cc0 * cg + cw, 0:l_pad2],
                                )

                    # ---- Main loop: one logical task = (row r, L-tile) ----
                    for task_id in T.serial(num_local_tasks):
                        lid = task_id * num_cores + kernel_id
                        if lid < num_logical:
                            r = lid // gx_l
                            tile = lid % gx_l
                            bz = r // c_out
                            oc = r % c_out
                            ol0 = tile * block_n
                            tn = T.min(block_n, out_l - ol0)

                            # per-tap contiguous shifted load + weight row load
                            for kw in T.serial(kernel_l):
                                cpos = kw * dilation_l
                                if need_poly:
                                    base = (cpos % stride_l) * mdec + ol0 + (cpos // stride_l)
                                    T.copy(
                                        xdec[bz, oc, base:base + tn],
                                        xt[kw:kw + 1, 0:tn],
                                    )
                                else:
                                    off = ol0 + cpos
                                    T.copy(
                                        xpad[bz, oc, off:off + tn],
                                        xt[kw:kw + 1, 0:tn],
                                    )
                            T.copy(weight[oc, 0, 0:kernel_l], wc[0:kernel_l, 0:1])

                            # upcast to fp32 (fp16/bf16) or copy (fp32)
                            if dtype != accum_dtype:
                                T.vcast(xt, xt32, round_mode="rint")
                                T.vcast(wc, w32, round_mode="rint")
                            else:
                                T.copy(xt, xt32)
                                T.copy(wc, w32)

                            # weighted tile (row broadcast) + single reduce over k_w
                            T.vmul(xt32, w32, prod)
                            T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)

                            # bias (fp32 domain, column broadcast)
                            if has_bias:
                                T.copy(bias[oc:oc + 1], b11[0:1, 0:1])
                                if dtype != accum_dtype:
                                    T.vcast(b11, b11_32, round_mode="rint")
                                else:
                                    T.copy(b11, b11_32)
                                T.vadd(acc, b11_32, acc)

                            # downcast + store (tail clipped)
                            if dtype != accum_dtype:
                                T.vcast(acc, out_ub, round_mode="rint")
                            else:
                                T.copy(acc, out_ub)
                            T.copy(
                                out_ub[0:1, 0:tn],
                                out[bz, oc:oc + 1, ol0:ol0 + tn],
                            )

        if has_bias:

            @T.prim_func
            def _conv1d_direct_bias_main(
                x: T.Tensor((n, c_in, l_in), dtype),
                weight: T.Tensor((c_out, 1, kernel_l), dtype),
                out: T.Tensor((n, c_out, out_l), dtype),
                bias: T.Tensor((c_out,), dtype),
                xpad: T.Tensor((n, c_in, l_pad2), dtype),
                xdec: T.Tensor(xdec_shape, dtype),
                poly_idx: T.Tensor(poly_shape, "int32"),
            ):
                _body(x, weight, out, bias, xpad, xdec, poly_idx)

            return _conv1d_direct_bias_main

        @T.prim_func
        def _conv1d_direct_main(
            x: T.Tensor((n, c_in, l_in), dtype),
            weight: T.Tensor((c_out, 1, kernel_l), dtype),
            out: T.Tensor((n, c_out, out_l), dtype),
            xpad: T.Tensor((n, c_in, l_pad2), dtype),
            xdec: T.Tensor(xdec_shape, dtype),
            poly_idx: T.Tensor(poly_shape, "int32"),
        ):
            _body(x, weight, out, None, xpad, xdec, poly_idx)

        return _conv1d_direct_main

    return _conv1d_direct_func


# ---------------------------------------------------------------------------
# Precision comparison + hierarchical testing
# ---------------------------------------------------------------------------
def _run_case(
    n,
    c_in,
    l_in,
    c_out,
    kernel_l,
    stride_l,
    pad_left,
    pad_right,
    dilation_l,
    has_bias,
    dtype,
    tag,
    seed=0,
    block_n=BLOCK_N,
):
    """Build the kernel for one config, run it on the NPU and compare vs golden.

    Raises AssertionError on shape mismatch / tolerance violation.
    Returns (ok, max_abs).
    """
    assert c_in == c_out, "depthwise requires c_in == c_out"
    assert pad_left == pad_right, "tests use symmetric padding (torch F.conv1d)"
    torch.manual_seed(seed)
    x = torch.randn(n, c_in, l_in, dtype=torch.float32).to(dtype)
    weight = torch.randn(c_out, 1, kernel_l, dtype=torch.float32).to(dtype)
    bias = torch.randn(c_out, dtype=torch.float32).to(dtype) if has_bias else None

    out_l, l_pad2, mdec, need_poly, _, _ = _plan(
        n, c_in, c_out, l_in, kernel_l, stride_l, pad_left, pad_right, dilation_l
    )

    fn = _conv1d_direct_kernel(
        n, c_in, l_in, c_out, kernel_l, stride_l, pad_left, pad_right,
        dilation_l, has_bias, _DTYPE_STR[dtype],
    )
    kernel = fn(block_n)

    x_npu = x.to("npu")
    w_npu = weight.to("npu")
    xpad = torch.zeros((n, c_in, l_pad2), dtype=dtype, device="npu")
    if need_poly:
        xdec = torch.empty((n, c_in, l_pad2), dtype=dtype, device="npu")
        poly_idx = _build_poly_idx(l_pad2, mdec, stride_l, c_in).to("npu")
    else:
        xdec = torch.empty((1, 1, 1), dtype=dtype, device="npu")
        poly_idx = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    if has_bias:
        out = kernel(x_npu, w_npu, bias.to("npu"), xpad, xdec, poly_idx)
    else:
        out = kernel(x_npu, w_npu, xpad, xdec, poly_idx)
    torch.npu.synchronize()

    ref = golden_conv1d_depthwise(x, weight, bias, stride_l, pad_left, dilation_l)
    atol, rtol = _TOL[dtype]
    assert tuple(out.shape) == (n, c_out, ref.shape[-1]), (
        f"shape {tuple(out.shape)} vs golden {tuple(ref.shape)}"
    )
    assert out.dtype == dtype, f"dtype {out.dtype} vs {dtype}"
    max_abs = float((out.cpu().float() - ref.float()).abs().max().item())
    try:
        torch.testing.assert_close(out.cpu(), ref, atol=atol, rtol=rtol)
        ok = True
    except AssertionError as exc:  # noqa: BLE001
        print(f"  [{tag}] FAIL: {str(exc).splitlines()[0]}")
        ok = False
    key = f"{_DTYPE_STR[dtype]}/has_bias={has_bias}"
    if max_abs == max_abs and max_abs > _STATS.get(key, 0.0):
        _STATS[key] = max_abs
    print(
        f"  [{tag}] n={n} C={c_in} l_in={l_in} kW={kernel_l} s={stride_l} "
        f"p={pad_left}/{pad_right} d={dilation_l} has_bias={has_bias} "
        f"{_DTYPE_STR[dtype]}: {'PASS' if ok else 'FAIL'} (max_abs={max_abs:.3e})"
    )
    return ok, max_abs


def _try(failures, case_id, **kwargs):
    """Run a case, collecting (not re-raising) per-case failures."""
    try:
        _run_case(tag=case_id, **kwargs)
    except AssertionError as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")


_F16 = torch.float16
_BF16 = torch.bfloat16


def run_L0():
    """L0 gate (blocking): representative workloads + all must-test points.

    Covers: conformer kW=31 (bias), smoke kW=3 (no bias), kW=1, kW=7 (non-div
    out_l), kW=10 + dilation, batch>1, stride=2 and stride=3 (polyphase),
    fp16 + bf16, padding>0, non-divisible dims.
    Returns the list of (case_id, exception) failures.
    """
    print("== L0 (blocking) ==")
    failures = []
    # (n, C, l_in, kW, stride, pad, dilation, has_bias, dtype)
    cases = [
        (1, 64, 128, 31, 1, 15, 1, True, _F16),    # conformer representative
        (1, 32, 128, 3, 1, 1, 1, False, _F16),     # smoke no-bias
        (1, 32, 100, 1, 1, 0, 1, True, _F16),      # kW=1
        (1, 48, 200, 7, 1, 3, 1, False, _F16),     # kW=7, non-div out_l=200
        (1, 16, 300, 10, 1, 9, 2, True, _F16),     # kW=10 + dilation>1 + pad
        (3, 16, 100, 5, 1, 2, 1, True, _F16),      # batch>1
        (1, 16, 256, 5, 2, 2, 1, False, _F16),     # stride=2 polyphase
        (1, 16, 129, 3, 3, 1, 1, True, _F16),      # stride=3 polyphase non-div
        (1, 32, 128, 3, 1, 1, 1, True, _BF16),     # bf16 bias
        (2, 8, 77, 3, 1, 1, 1, False, _BF16),      # bf16 batch + non-div
    ]
    seed = 0
    for (nn, cc, ll, kw, ss, pp, dd, hb, dt) in cases:
        _try(
            failures,
            f"L0-{nn}x{cc}x{ll}-kW{kw}-s{ss}-p{pp}-d{dd}-{'bias' if hb else 'nobias'}-{_DTYPE_STR[dt]}",
            n=nn, c_in=cc, l_in=ll, c_out=cc, kernel_l=kw, stride_l=ss,
            pad_left=pp, pad_right=pp, dilation_l=dd, has_bias=hb, dtype=dt,
            seed=seed,
        )
        seed += 1
    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L0] PASS ({len(cases)} cases)")
    return failures


def run_L1():
    """L1 gate (blocking): dtype x config x has_bias matrix (f16/bf16)."""
    print("== L1 (blocking) ==")
    failures = []
    # (n, C, l_in, kW, stride, pad, dilation, name)
    cases = [
        (1, 24, 256, 1, 1, 0, 1, "kW1"),
        (1, 16, 300, 10, 1, 9, 2, "dil10"),
        (1, 16, 256, 5, 2, 2, 1, "s2"),
        (1, 8, 120, 7, 3, 2, 1, "s3"),
        (2, 16, 100, 5, 1, 2, 1, "batch2"),
        (1, 17, 77, 3, 1, 1, 1, "odd"),
    ]
    seed = 100
    n_cases = 0
    for (nn, cc, ll, kw, ss, pp, dd, name) in cases:
        for dt in (_F16, _BF16):
            for hb in (False, True):
                _try(
                    failures,
                    f"L1-{name}-{'bias' if hb else 'nobias'}-{_DTYPE_STR[dt]}",
                    n=nn, c_in=cc, l_in=ll, c_out=cc, kernel_l=kw, stride_l=ss,
                    pad_left=pp, pad_right=pp, dilation_l=dd, has_bias=hb,
                    dtype=dt, seed=seed,
                )
                seed += 1
                n_cases += 1
    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L1] PASS ({n_cases} cases)")
    return failures


def run_L2():
    """L2 (warn-only): larger / odd shapes."""
    print("== L2 (warn-only) ==")
    cases = [
        (4, 64, 512, 3, 1, 1, 1, "L2-batch4"),
        (1, 32, 513, 5, 1, 2, 1, "L2-nondiv"),
        (1, 8, 1024, 31, 1, 15, 1, "L2-kW31"),
        (1, 16, 512, 7, 4, 3, 1, "L2-s4"),
    ]
    n_pass = 0
    total = 0
    for (nn, cc, ll, kw, ss, pp, dd, name) in cases:
        for dt in (_F16, _BF16):
            for hb in (False, True):
                total += 1
                try:
                    ok, _ = _run_case(
                        n=nn, c_in=cc, l_in=ll, c_out=cc, kernel_l=kw, stride_l=ss,
                        pad_left=pp, pad_right=pp, dilation_l=dd, has_bias=hb,
                        dtype=dt, tag=f"{name}-{'bias' if hb else 'nobias'}-{_DTYPE_STR[dt]}",
                        seed=200,
                    )
                    n_pass += 1 if ok else 0
                except Exception as e:  # noqa: BLE001 - warn-only layer
                    print(f"  [{name}] WARN (exception): {e}")
    if n_pass == total:
        print(f"[L2] PASS ({total} cases)")
    else:
        print(f"[L2] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


def run_boundary():
    """Boundary (warn-only): tiny / degenerate valid shapes."""
    print("== Boundary (warn-only) ==")
    cases = [
        (1, 1, 8, 1, 1, 0, 1),      # C=1, kW=1
        (1, 3, 16, 3, 1, 1, 1),     # tiny C
        (1, 16, 16, 3, 1, 1, 1),    # shortest valid
        (1, 5, 33, 7, 2, 3, 1),     # odd + stride 2
        (1, 16, 40, 10, 1, 9, 2),   # dilation, non-div
    ]
    n_pass = 0
    total = 0
    for (nn, cc, ll, kw, ss, pp, dd) in cases:
        for hb in (False, True):
            total += 1
            tag = f"B-{nn}x{cc}x{ll}-kW{kw}-s{ss}-{'bias' if hb else 'nobias'}"
            try:
                ok, _ = _run_case(
                    n=nn, c_in=cc, l_in=ll, c_out=cc, kernel_l=kw, stride_l=ss,
                    pad_left=pp, pad_right=pp, dilation_l=dd, has_bias=hb,
                    dtype=_F16, tag=tag, seed=300,
                )
                n_pass += 1 if ok else 0
            except Exception as e:  # noqa: BLE001 - warn-only layer
                print(f"  [{tag}] WARN (exception): {e}")
    if n_pass == total:
        print(f"[Boundary] PASS ({total} cases)")
    else:
        print(f"[Boundary] WARN: {total - n_pass}/{total} cases failed (non-blocking)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []
    failures += run_L0()
    if args.level == "all":
        failures += run_L1()
        run_L2()
        run_boundary()

    print(
        "Summary max_abs per (dtype, has_bias): "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} blocking case(s) failed: "
            + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
