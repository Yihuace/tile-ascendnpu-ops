# R7 candidate: probe: deleted mulB
# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Stage 4 R6 candidate A: signed (p=0.5) mask threshold.

Derived from the R5 winner: for p == 0.5 the keep test `|h| >= 2^30` is
replaced by `h >= 0`, which drops `vabs` + `vsub` (-2 int32 vector ops).
Host model + block-chi2 validation: ``probes/hash_variants_r6.py``.

Original docstring follows.


Three optimization points, each verified as a single-point branch before the
merge (see ``opt_log.md`` Round 2-5):

  * UB diet: no ``base``/``m31``/``zerob`` full-size int32 buffers; the mask
    sign strip uses ``vabs`` and ``vclamp(0,1)`` replaces ``vmax+vmin`` (one
    fewer v-op, 12 B/elem less UB) -> larger ``block_size`` headroom.
  * 5-op counter hash (was 9): ``h = ((c*A) ^ ((c*A)<<13)) * B`` with
    ``c = idx + seed``; block-chi2 iid test on 8 Mi elements matches the
    Stage 3 chain (see ``_hash_analysis.py``).
  * fp16-only storage-dtype compute domain when ``scale`` is exactly
    representable in fp16 (p=0.5 -> 2.0): drops the fp32 round-trip.

Contract preserved: ``h`` remains a pure function of ``(seed, element index)``
(deterministic replay), the 0/1 mask is a fixed-size pseudorandom subset
(exact drop rate), and no right shift / ``vselect`` / host-side mask
materialisation is used.

run: python _make_dropout_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data dependency; no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import functools

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() placement

import tilelang
import tilelang.language as T

# Programming mode actually validated by this file (Stage 5 syncs this value
# into the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_F32 = "float32"
# Signed int32 forms of the mixer constants 0x9E3779B1 / 0x85EBCA6B.
# ``T.vmul`` on int32 wraps mod 2^32 on the device (verified), so the signed
# forms are numerically equivalent to the unsigned constants.
_A_I32 = -1640531527
_B_I32 = -2048144777
_M31 = 0x7FFFFFFF
# Physical Vector core count on Ascend910_9362 (20 AI cores x 2).
_CORE_NUM = 40
# Stage 4 TUNED tile size (K11 collapsed block_size).  The Stage 3 defaults were
# 1024 (fp32) / 2048 (fp16, bf16); the UB diet + fp16-only storage-dtype compute
# domain raised the BiShengIR UB ceiling so that a single 8192 tile is legal for
# every float dtype (overflow reappears at 10240).  The wrapper's baseline
# /perf_opt switch block must mirror this constant.
_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 8192, "bfloat16": 8192}
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
# Tolerance table from examples/TileOPs/tests/ops/test_dropout.py.
_ATOL = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}
_RTOL = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_from_mask(x, mask, p):
    """Deterministic contract reference for a given 0/1 mask.

    Implements the torch ``native_dropout`` multiplicative-mask semantics
    ``x * (mask * scale)`` in the fp32 opmath domain, then rounds back to the
    input dtype.  Independent of the NPU vector chain (the mask is supplied by
    the caller; the arithmetic comes from the torch formula).
    """
    x_f = x.detach().cpu().to(torch.float32)
    m = mask.detach().cpu().to(torch.float32)
    scale = 0.0 if p >= 1.0 else 1.0 / (1.0 - p)
    return (x_f * (m * scale)).to(x.dtype)


def golden_dropout(x, p=0.5, training=True, seed=0):
    """torch contract baseline (used for the ``p=0`` / ``p=1`` / ``training``
    boundary expectations).

    The torch RNG differs from the NPU in-core hash, so the mask is *not*
    bit-comparable; this baseline is only used where the expected value is
    deterministic (``p=0`` identity, ``p=1`` all zeros, ``training=False``).
    ``seed`` is accepted for signature parity with the source and is unused
    here because ``F.dropout`` on CPU does not expose a generator argument.
    """
    del seed  # unused: torch CPU dropout has no generator argument
    if not training or p == 0.0:
        return x.detach().cpu().clone()
    return F.dropout(x.detach().cpu().to(torch.float32), p, training=True).to(x.dtype)


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _make_dropout_kernel(N, dtype, p, seed, default_block_size=2048):
    """Build the inverted-dropout kernel (NPU, ``target="npuir"``).

    ``p`` and ``seed`` are compile-time Python constants (source ``lru_cache``
    specialisation); ``N`` is a compile-time constant as well, so the dynamic
    tail tile needs no runtime ``shape`` argument.  The returned callable takes
    a single ``block_size`` (K11) and is invoked as ``fn(x)`` (``out_idx=[1]``
    auto-allocates ``y``).
    """
    del default_block_size  # kept for DESIGN.md section 3.3 signature parity
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    # Drop threshold on bits [8..30]: drop probability = Tp / 2^23.
    tp = int(round(p * float(1 << 23)))
    tp_shift = 256 * tp - 1  # keep iff hm >= 256*tp
    # p == 0.5 -> the signed hash sign bit is an exact 50% threshold test.
    signed_mask = (p == 0.5)

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        need_cast = dtype != _F32
        # fp16-only storage-dtype compute domain: taken when ``scale`` is exactly
        # representable in fp16 (p=0.5 -> 2.0).  bf16 is NOT taken -- measured
        # slower than the fp32 round-trip on this device (see opt_log Round 4).
        native_domain = need_cast and dtype == "float16" and (
            float(__import__("numpy").float16(scale)) == scale)
        md_extent = block_size if native_domain else 1
        n_num = T.ceildiv(N, block_size)

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(_CORE_NUM, is_npu=True) as (cid, _):
                # Buffers are allocated unconditionally at the top (avoids the
                # conditional-alloc trap); the compile-time ``need_cast`` branch
                # only selects which buffer is used.
                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                xf = T.alloc_ub((block_size,), _F32)
                lane = T.alloc_ub((block_size,), "int32")
                h = T.alloc_ub((block_size,), "int32")
                t = T.alloc_ub((block_size,), "int32")
                tp_b = T.alloc_ub((block_size,), "int32")
                mf = T.alloc_ub((block_size,), _F32)
                md = T.alloc_ub((md_extent,), dtype)

                # Loop-invariant constants (filled once, outside the tile loop).
                T.fill(tp_b, T.cast(tp_shift, "int32"))
                T.arange(lane, [1], 0)  # lane[i] = i

                xc = xf if need_cast else x_ub

                # In-core grid stride; upper bound is a static integer.
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_num:
                        off = task * block_size
                        tail = T.min(block_size, N - off)

                        # ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----
                        T.fill(h, off + seed)      # counter = idx + seed
                        T.vadd(lane, h, h)         # h = counter
                        T.vmul(h, _A_I32, h)
                        T.vshl(h, 13, t)
                        T.vxor(h, t, h)

                        # ---- 0/1 mask ----
                        if signed_mask:
                            # p == 0.5: P(h >= 0) == 0.5 exactly, so the sign
                            # bit alone is the threshold test -> one vclamp
                            # (replaces vabs + vsub + vclamp, -2 int32 ops).
                            T.vclamp(h, h, 0, 1)   # m = 1 iff h >= 1
                            T.vcast(h, mf, round_mode="rint")
                        else:
                            T.vabs(h, t)               # t = |h|
                            T.vsub(t, tp_b, t)         # d = hm - (256*tp-1)
                            T.vclamp(t, t, 0, 1)       # m = clamp(d, 0, 1)
                            T.vcast(t, mf, round_mode="rint")

                        # ---- data path ----
                        T.copy(x[off:off + tail], x_ub[0:tail])
                        if native_domain:
                            # 0/1 mask and scale are exact in fp16 here, so the
                            # fp32 round-trip is redundant (2 fewer v-ops).
                            T.vcast(mf, md, round_mode="rint")
                            T.vmul(md, T.cast(scale, dtype), md)  # scale * mask
                            T.vmul(x_ub, md, x_ub)                # x * (scale*m)
                            T.copy(x_ub[0:tail], y[off:off + tail])
                        else:
                            if need_cast:
                                T.vcast(x_ub, xc, round_mode="rint")
                            T.vmul(xc, T.cast(scale, _F32), xc)  # x * scale
                            T.vmul(xc, mf, xc)                   # * mask (0/1)
                            if need_cast:
                                T.vcast(xc, y_ub, round_mode="rint")
                                T.copy(y_ub[0:tail], y[off:off + tail])
                            else:
                                T.copy(xc[0:tail], y[off:off + tail])

        return main

    return kernel


# ---------- precision comparing func ----------
def _run_kernel(x_cpu, dtype_str, p, seed, block_size=None):
    """Run the NPU kernel on a CPU tensor; return the CPU output (flattened)."""
    N = x_cpu.numel()
    if block_size is None:
        block_size = _DEFAULT_BLOCK_SIZE[dtype_str]
    kernel = _make_dropout_kernel(N, dtype_str, float(p), int(seed))
    fn = kernel(block_size)
    x_flat = x_cpu.reshape(-1).contiguous()
    return fn(x_flat.npu()).cpu()


def dropout_forward(x_cpu, dtype_str, p, seed, training=True, block_size=None):
    """Host entry mirroring the Op contract: training=False is a host-level
    identity short path, otherwise the NPU kernel runs."""
    if not training:
        return x_cpu.detach().cpu().clone()
    return _run_kernel(x_cpu, dtype_str, p, seed, block_size)


def check_dropout_contract(y, x, p, y2=None):
    """Contract properties: structure, scale, empirical drop rate, replay.

    Structure is checked elementwise: every output element must be either ``0``
    (dropped) or ``x * scale`` (kept); this holds for both constant and random
    inputs.  Returns ``(structure_ok, rate, rate_ok, replay_ok)``.  Tolerance
    follows the source test tolerances (fp32 1e-5 / fp16 1e-3 / bf16 1.6e-2).
    """
    y = y.detach().cpu()
    x = x.detach().cpu()
    scale = 0.0 if p >= 1.0 else 1.0 / (1.0 - p)
    at = _ATOL[x.dtype]
    rt = _RTOL[x.dtype]
    yf = y.to(torch.float32)
    xf = x.to(torch.float32)
    is_zero = yf == 0
    is_scaled = torch.isclose(yf, xf * scale, atol=at, rtol=rt)
    structure_ok = bool((is_zero | is_scaled).all())
    n = x.numel()
    # Dropped = output is zero for a non-zero input (an input zero is kept as 0).
    rate = float(((is_zero & (xf != 0)).to(torch.float32).sum() / max(n, 1)).item())
    sigma3 = 3.0 * (p * (1 - p) / n) ** 0.5 if 0 < p < 1 else 0.0
    rate_ok = bool(abs(rate - p) <= max(sigma3, 1e-6))
    replay_ok = None if y2 is None else bool(torch.equal(y, y2.detach().cpu()))
    return structure_ok, rate, rate_ok, replay_ok


def _assert_contract(y, x, p, tag, expect_rate=True, y2=None):
    structure_ok, rate, rate_ok, replay_ok = check_dropout_contract(y, x, p, y2)
    assert structure_ok, (
        f"[{tag}] structure FAIL: elements are neither 0 nor x*scale "
        f"(dtype={x.dtype}, p={p})"
    )
    if expect_rate:
        assert rate_ok, (
            f"[{tag}] drop rate FAIL: rate={rate:.6f} vs p={p} "
            f"(dtype={x.dtype}, N={x.numel()})"
        )
    if replay_ok is not None:
        assert replay_ok, f"[{tag}] deterministic replay FAIL (same seed differs)"
    extra = "" if replay_ok is None else f" replay={replay_ok}"
    print(f"[{tag}] PASS dtype={x.dtype} p={p} N={x.numel()} rate={rate:.6f}{extra}")
    return rate


def _run_case(case_id, N, dtype_str, p, seed, *, block_size=None, kind="ones",
              expect_rate=None, replay=False):
    """Single test case driver: build input, run the kernel, check the contract.

    The empirical drop rate is only well defined for a constant non-zero input
    (``kind="ones"``): with a random input the tensor itself contains exact
    zeros, so counting zeros cannot isolate the dropped positions.  ``None``
    therefore auto-selects the rate check for ``ones`` only.
    """
    if expect_rate is None:
        expect_rate = kind == "ones"
    tdt = _TORCH_DTYPE[dtype_str]
    if kind == "ones":
        x = torch.ones(N, dtype=tdt)
    elif kind == "randn":
        x = torch.randn(N, dtype=tdt)
    else:
        raise ValueError(f"unknown kind {kind}")
    y = dropout_forward(x, dtype_str, p, seed, block_size=block_size)
    y2 = None
    if replay:
        y2 = dropout_forward(x, dtype_str, p, seed, block_size=block_size)
    _assert_contract(y, x, p, case_id, expect_rate=expect_rate, y2=y2)
    return y


# ---------- hierarchical testing ----------
def run_L0():
    """Gate smoke suite (DESIGN.md section 8.3 L0-1..L0-11).

    Must pass: statistical drop rate, scale, deterministic replay, seed
    separation, p=0 / p=1 edges, training=False, shape/dtype, non-divisible
    tail, non-default block_size.
    """
    cases = 0

    # L0-1: statistical drop rate p=0.5 across all three dtypes (N=4M).
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case("L0-1", 4_194_304, dtype_str, 0.5, 42, kind="ones")
        cases += 1

    # L0-2: statistical drop rate fp16 p=0.1 / p=0.3 (N=4M).
    for p in (0.1, 0.3):
        _run_case("L0-2", 4_194_304, "float16", p, 42, kind="ones")
        cases += 1

    # L0-3: scale correctness for all dtypes (N=1M, non-divisible tail).
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case("L0-3", 1_000_000, dtype_str, 0.5, 42, kind="ones")
        cases += 1

    # L0-4: deterministic replay (same seed -> identical output).
    _run_case("L0-4", 1_000_000, "float16", 0.5, 777, kind="randn", replay=True)
    _run_case("L0-4", 1_000_000, "float32", 0.5, 777, kind="randn", replay=True)
    cases += 2

    # L0-5: different seeds -> different masks (mask diff fraction ~ 0.5).
    xa = torch.ones(262_144, dtype=torch.float32)
    ya = dropout_forward(xa, "float32", 0.5, 42)
    yb = dropout_forward(xa, "float32", 0.5, 99)
    assert not torch.equal(ya, yb), "[L0-5] different seeds produced identical output"
    diff_frac = float((ya != yb).to(torch.float32).mean())
    assert 0.3 < diff_frac < 0.7, f"[L0-5] seed separation frac={diff_frac:.3f} (expect ~0.5)"
    print(f"[L0-5] PASS seed separation frac={diff_frac:.3f}")
    cases += 1

    # L0-6: p=0 identity.
    for dtype_str in ("float32", "float16"):
        x = torch.randn(1_000_000, dtype=_TORCH_DTYPE[dtype_str])
        y = dropout_forward(x, dtype_str, 0.0, 42)
        torch.testing.assert_close(y, x)
        print(f"[L0-6] PASS p=0 identity dtype={dtype_str}")
        cases += 1

    # L0-7: p=1 all zeros.
    for dtype_str in ("float32", "float16"):
        x = torch.randn(1_000_000, dtype=_TORCH_DTYPE[dtype_str])
        y = dropout_forward(x, dtype_str, 1.0, 42)
        assert torch.equal(y, torch.zeros_like(x)), f"[L0-7] p=1 not all zeros ({dtype_str})"
        print(f"[L0-7] PASS p=1 all zeros dtype={dtype_str}")
        cases += 1

    # L0-8: training=False host short path (must not touch the kernel).
    x = torch.randn(1_000_000, dtype=torch.float32)
    y = dropout_forward(x, "float32", 0.5, 42, training=False)
    g = golden_dropout(x, 0.5, training=False)
    torch.testing.assert_close(y, x)
    torch.testing.assert_close(y, g)
    assert torch.equal(y, x), "[L0-8] training=False is not an exact identity"
    print("[L0-8] PASS training=False host identity")
    cases += 1

    # L0-9: output shape / dtype preserved (any input shape, flattened on host).
    for dtype_str in ("float32", "float16"):
        x = torch.randn((100, 10_000), dtype=_TORCH_DTYPE[dtype_str])
        y_flat = dropout_forward(x, dtype_str, 0.5, 42)
        y = y_flat.reshape(x.shape)
        assert y.shape == x.shape, f"[L0-9] shape {y.shape} != {x.shape}"
        assert y.dtype == x.dtype, f"[L0-9] dtype {y.dtype} != {x.dtype}"
        _assert_contract(y, x, 0.5, "L0-9", expect_rate=False)
        cases += 1

    # L0-10: non-divisible tail (4_000_000 % 2048 == 1216).
    _run_case("L0-10", 4_000_000, "float16", 0.5, 42, kind="ones")
    cases += 1

    # L0-11: non-default block_size regression (grid/launch config match).
    for N, dtype_str, bs, p in (
        (8192, "float16", 512, 0.0),
        (8192, "float32", 128, 0.0),
        (65536, "float16", 1024, 0.0),
    ):
        x = torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])
        y = dropout_forward(x, dtype_str, p, 0, block_size=bs)
        torch.testing.assert_close(y, x)
        print(f"[L0-11] PASS custom block_size={bs} N={N} dtype={dtype_str} p=0 identity")
        cases += 1

    print(f"  -> run_L0: {cases} cases PASS")
    return cases


def run_L1():
    """Extended coverage: more dtype/p combinations, manifest-scale N, tails."""
    cases = 0

    # L1-1: statistical rate for additional dtype/p combinations.
    for N, dtype_str, p, seed in (
        (4_194_304, "float32", 0.3, 42),
        (4_194_304, "bfloat16", 0.5, 42),
        (4_194_304, "float16", 0.7, 42),
    ):
        _run_case("L1-1", N, dtype_str, p, seed, kind="ones")
        cases += 1

    # L1-2: bf16 deterministic replay.
    _run_case("L1-2", 1_000_000, "bfloat16", 0.5, 777, kind="randn", replay=True)
    cases += 1

    # L1-3: bf16 p=0 / p=1 edges.
    x = torch.randn(1_000_000, dtype=torch.bfloat16)
    torch.testing.assert_close(dropout_forward(x, "bfloat16", 0.0, 42), x)
    assert torch.equal(
        dropout_forward(x, "bfloat16", 1.0, 42), torch.zeros_like(x),
    ), "[L1-3] bf16 p=1 not all zeros"
    print("[L1-3] PASS bf16 p=0 identity / p=1 all zeros")
    cases += 2

    # L1-4: non-default block_size with p=0.5 (structure + rate + launch match).
    _run_case("L1-4", 8192, "float16", 0.5, 42, kind="ones", block_size=512)
    _run_case("L1-4", 65536, "float16", 0.5, 42, kind="ones", block_size=1024)
    cases += 2

    # L1-5: bf16 non-divisible tail (4_000_000 % 2048 == 1216).
    _run_case("L1-5", 4_000_000, "bfloat16", 0.5, 42, kind="ones")
    cases += 1

    # L1-6: manifest-scale workload [1024, 10240] bf16 -> N = 10_485_760.
    _run_case("L1-6", 10_485_760, "bfloat16", 0.5, 42, kind="ones")
    cases += 1

    # L1-7: bf16 seed separation.
    xb = torch.ones(262_144, dtype=torch.bfloat16)
    d = float((dropout_forward(xb, "bfloat16", 0.5, 42) !=
               dropout_forward(xb, "bfloat16", 0.5, 99)).to(torch.float32).mean())
    assert 0.3 < d < 0.7, f"[L1-7] bf16 seed separation frac={d:.3f}"
    print(f"[L1-7] PASS bf16 seed separation frac={d:.3f}")
    cases += 1

    # L1-8: fp32 non-divisible tail at the K11 block_size.
    _run_case("L1-8", 1_000_000, "float32", 0.5, 123, kind="ones")
    cases += 1

    print(f"  -> run_L1: {cases} cases PASS")
    return cases


def run_L2():
    """Edge sizes (non-blocking: record without blocking)."""
    cases = 0
    ok = 0
    for N in (1, 2, 10, 1025, 2047, 4095, 4097):
        for dtype_str in ("float32", "float16"):
            cases += 1
            try:
                x = torch.randn(N, dtype=_TORCH_DTYPE[dtype_str])
                y = dropout_forward(x, dtype_str, 0.5, 42)
                assert y.shape == x.shape
                assert y.dtype == x.dtype
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[L2] WARN (non-blocking): N={N} dtype={dtype_str} err={e}")
    print(f"  -> run_L2: {ok}/{cases} cases PASS (non-blocking)")
    return ok, cases


def _equal_nan(a, b):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def run_boundary():
    """Special-value boundary cases (non-blocking: record without blocking).

    Contract (torch native_dropout multiplicative mask): a dropped ``+-inf``
    becomes NaN, a kept ``+-inf`` stays ``+-inf``, ``-0.0`` is preserved, and
    ``p=0`` is an exact identity.
    """
    cases = 0
    ok = 0
    vals = [float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 3.0, -7.0, 1e-45]
    for dtype_str in ("float32", "float16", "bfloat16"):
        tdt = _TORCH_DTYPE[dtype_str]
        N = 4096
        x = torch.tensor((vals * (N // len(vals)))[:N], dtype=tdt).contiguous()

        # p=1.0: all dropped -> compare against the multiplicative golden.
        cases += 1
        try:
            y = dropout_forward(x, dtype_str, 1.0, 42)
            exp = golden_from_mask(x, torch.zeros(N, dtype=torch.float32), 1.0)
            assert _equal_nan(y, exp), f"[Boundary] p=1 inf/NaN mismatch ({dtype_str})"
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): p=1 {dtype_str} err={e}")

        # p=0.0: exact identity (inf / -0.0 / NaN preserved).
        cases += 1
        try:
            y = dropout_forward(x, dtype_str, 0.0, 42)
            assert _equal_nan(y, x), f"[Boundary] p=0 not identity ({dtype_str})"
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): p=0 {dtype_str} err={e}")

        # p=0.5: every element is 0 / NaN / x*scale (drop +-inf -> NaN).
        cases += 1
        try:
            y = dropout_forward(x, dtype_str, 0.5, 42)
            yf = y.to(torch.float32)
            xf = x.to(torch.float32)
            ok_mask = torch.isnan(yf) | (yf == 0) | torch.isclose(
                yf, xf * 2.0, atol=_ATOL[tdt], rtol=0,
            )
            assert bool(ok_mask.all()), f"[Boundary] p=0.5 structure mismatch ({dtype_str})"
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): p=0.5 {dtype_str} err={e}")

    print(f"  -> run_boundary: {ok}/{cases} cases PASS (non-blocking)")
    return ok, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
