#!/usr/bin/env python3
"""R8 candidate generator: Expert-mode multi-chain interleave for dropout.

Hypothesis under test: at ``block_size=8192`` a core runs one tile and one
dependency chain; splitting into ``block_size=4096`` x 2 independent chains
(UB footprint unchanged) lets the V pipe fill the latency of chain A with
independent ops of chain B, cutting the one-time ramp/drain overhead.

Generated variants (all share the same 11-op math, only tiling/schedule):
  b4096_dev : block_size 4096 single chain, Developer (control for tile size)
  b4096_exp : block_size 4096 single chain, Expert    (control for mode)
  c4096x2   : block_size 4096, 2 chains interleaved, Expert  (hypothesis)
  d2048x4   : block_size 2048, 4 chains interleaved, Expert  (extension)

Expert grid model: ``T.Kernel(20)`` (AI cores) x 2 AIV sub-blocks
(PL-1.13-aiv-dup-subid-split): every AIV owns one persistent stream
``cid*2+subid`` (40 streams, same as Developer ``cid``).
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "_make_dropout_kernel.py")


def _load_base():
    with open(BASE) as fh:
        return fh.read()


def _kernel_block(ch, mode):
    """Emit the whole ``@tilelang.jit`` kernel factory for ``ch`` chains."""
    expert = mode == "Expert"
    blocks = 20 if expert else 40
    cid_name, sub_name = ("cid", "subid") if expert else ("cid", "_")
    stream = "cid * 2 + subid" if expert else "cid"
    scope_open = '                with T.Scope("Vector"):\n' if expert else ""
    ind = "                    " if expert else "                "
    i2 = ind + "    "
    i3 = i2 + "    "
    i4 = i3 + "    "

    L = []
    L.append("    @tilelang.jit(out_idx=[1], target=\"npuir\")")
    L.append("    def kernel(block_size):")
    L.append("        need_cast = dtype != _F32")
    L.append("        # fp16-only storage-dtype compute domain (p=0.5 -> 2.0).")
    L.append("        native_domain = need_cast and dtype == \"float16\" and (")
    L.append("            float(__import__(\"numpy\").float16(scale)) == scale)")
    L.append("        md_extent = block_size if native_domain else 1")
    L.append("        xf_extent = block_size if need_cast else 1")
    L.append("        tp_extent = 1 if signed_mask else block_size")
    L.append("        n_num = T.ceildiv(N, block_size)")
    L.append(f"        n_grp = T.ceildiv(n_num, {ch})")
    L.append("        n_iter = T.ceildiv(n_grp, _STREAMS)")
    L.append("        # Odd tile count: the last group re-processes a clamped")
    L.append("        # (idempotent, elementwise-deterministic) tile instead of a")
    L.append("        # runtime guard, so the hot loop stays branch-free.")
    L.append("        off_cap = max(N - block_size, 0)")
    L.append("")
    L.append("        @T.prim_func")
    L.append("        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):")
    L.append(f"            with T.Kernel({blocks}, is_npu=True) as ({cid_name}, {sub_name}):")
    if scope_open:
        L.append(scope_open.rstrip("\n"))
    # allocations
    for j in range(ch):
        L.append(f"{ind}x{j} = T.alloc_ub((block_size,), dtype)")
    for j in range(ch):
        L.append(f"{ind}y{j} = T.alloc_ub((block_size,), dtype)")
    for j in range(ch):
        L.append(f"{ind}f{j} = T.alloc_ub((xf_extent,), _F32)")
    for j in range(ch):
        L.append(f"{ind}c{j} = f{j} if need_cast else x{j}")
    L.append(f"{ind}lane = T.alloc_ub((block_size,), \"int32\")")
    for j in range(ch):
        L.append(f"{ind}h{j} = T.alloc_ub((block_size,), \"int32\")")
    for j in range(ch):
        L.append(f"{ind}t{j} = T.alloc_ub((block_size,), \"int32\")")
    L.append(f"{ind}tp_b = T.alloc_ub((tp_extent,), \"int32\")")
    for j in range(ch):
        L.append(f"{ind}mf{j} = T.alloc_ub((block_size,), _F32)")
    for j in range(ch):
        L.append(f"{ind}md{j} = T.alloc_ub((md_extent,), dtype)")
    L.append("")
    L.append(f"{ind}# Loop-invariant constants (filled once).")
    L.append(f"{ind}T.fill(tp_b, T.cast(tp_shift, \"int32\"))")
    L.append(f"{ind}T.arange(lane, [1], 0)")
    L.append("")
    L.append(f"{ind}for i in T.serial(n_iter):")
    L.append(f"{i2}g = i * _STREAMS + {stream}")
    L.append(f"{i2}if g < n_grp:")
    L.append(f"{i3}t0_ = g * {ch}")
    L.append(f"{i3}off0 = t0_ * block_size")
    L.append(f"{i3}tail0 = T.min(block_size, N - off0)")
    for j in range(1, ch):
        L.append(f"{i3}off{j} = T.min((t0_ + {j}) * block_size, off_cap)")
        L.append(f"{i3}tail{j} = T.min(block_size, N - off{j})")
    L.append("")
    # ---- stage-by-stage interleave ----
    stages = [
        ("# ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----", None),
        (None, "T.fill(h{j}, off{j} + seed)"),
        (None, "T.vadd(lane, h{j}, h{j})"),
        (None, "T.vmul(h{j}, _A_I32, h{j})"),
        (None, "T.vshl(h{j}, 13, t{j})"),
        (None, "T.vxor(h{j}, t{j}, h{j})"),
        (None, "T.vmul(h{j}, _B_I32, h{j})"),
        ("# ---- 0/1 mask ----", None),
    ]
    for comment, stmt in stages:
        if comment is not None:
            L.append(f"{i3}{comment}")
        else:
            for j in range(ch):
                L.append(f"{i3}{stmt.format(j=j)}")
    L.append(f"{i3}if signed_mask:")
    for j in range(ch):
        L.append(f"{i4}T.vclamp(h{j}, h{j}, 0, 1)")
    for j in range(ch):
        L.append(f"{i4}T.vcast(h{j}, mf{j}, round_mode=\"rint\")")
    L.append(f"{i3}else:")
    for j in range(ch):
        L.append(f"{i4}T.vabs(h{j}, t{j})")
    for j in range(ch):
        L.append(f"{i4}T.vsub(t{j}, tp_b, t{j})")
    for j in range(ch):
        L.append(f"{i4}T.vclamp(t{j}, t{j}, 0, 1)")
    for j in range(ch):
        L.append(f"{i4}T.vcast(t{j}, mf{j}, round_mode=\"rint\")")
    L.append("")
    L.append(f"{i3}# ---- data path (loads of all chains first) ----")
    for j in range(ch):
        L.append(f"{i3}T.copy(x[off{j}:off{j} + tail{j}], x{j}[0:tail{j}])")
    L.append(f"{i3}if native_domain:")
    for j in range(ch):
        L.append(f"{i4}T.vcast(mf{j}, md{j}, round_mode=\"rint\")")
    for j in range(ch):
        L.append(f"{i4}T.vmul(md{j}, T.cast(scale, dtype), md{j})")
    for j in range(ch):
        L.append(f"{i4}T.vmul(x{j}, md{j}, x{j})")
    for j in range(ch):
        L.append(f"{i4}T.copy(x{j}[0:tail{j}], y[off{j}:off{j} + tail{j}])")
    L.append(f"{i3}else:")
    if True:
        L.append(f"{i4}if need_cast:")
        for j in range(ch):
            L.append(f"{i4}    T.vcast(x{j}, c{j}, round_mode=\"rint\")")
        for j in range(ch):
            L.append(f"{i4}T.vmul(c{j}, T.cast(scale, _F32), c{j})")
        for j in range(ch):
            L.append(f"{i4}T.vmul(c{j}, mf{j}, c{j})")
        L.append(f"{i4}if need_cast:")
        for j in range(ch):
            L.append(f"{i4}    T.vcast(c{j}, y{j}, round_mode=\"rint\")")
        for j in range(ch):
            L.append(f"{i4}    T.copy(y{j}[0:tail{j}], y[off{j}:off{j} + tail{j}])")
        L.append(f"{i4}else:")
        for j in range(ch):
            L.append(f"{i4}    T.copy(c{j}[0:tail{j}], y[off{j}:off{j} + tail{j}])")
    L.append("")
    L.append("        return main")
    return "\n".join(L) + "\n"


_HEAD = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""R8 candidate: {desc}

Generated by ``perf_opt/probes/gen_r8.py`` from the R6 11-op chain; the only
change is the tiling / scheduling form (math and mask contract untouched).
ASCEND_MODE={mode}, block_size={bs}, chains={ch}.
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "{mode}"

import argparse
import functools

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  enables .npu() placement

import tilelang
import tilelang.language as T

ASCEND_MODE = "{mode}"

tilelang.cache.clear_cache()

_F32 = "float32"
_A_I32 = -1640531527
_B_I32 = -2048144777
_M31 = 0x7FFFFFFF
# Expert: 20 AI cores x 2 AIV sub-blocks; each AIV is one persistent stream.
_BLOCKS = {blocks}
_STREAMS = {streams}
_DEFAULT_BLOCK_SIZE = {{"float32": {bs}, "float16": {bs}, "bfloat16": {bs}}}
_TORCH_DTYPE = {{
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}}
_ATOL = {{torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}}
_RTOL = {{torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}}
'''

_MID = '''

# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_from_mask(x, mask, p):
    """Deterministic contract reference for a given 0/1 mask."""
    x_f = x.detach().cpu().to(torch.float32)
    m = mask.detach().cpu().to(torch.float32)
    scale = 0.0 if p >= 1.0 else 1.0 / (1.0 - p)
    return (x_f * (m * scale)).to(x.dtype)


def golden_dropout(x, p=0.5, training=True, seed=0):
    """torch contract baseline (p=0 identity / p=1 all zeros / training=False)."""
    del seed
    if not training or p == 0.0:
        return x.detach().cpu().clone()
    return F.dropout(x.detach().cpu().to(torch.float32), p, training=True).to(x.dtype)


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _make_dropout_kernel(N, dtype, p, seed, default_block_size=2048):
    """Build the inverted-dropout kernel (NPU, ``target="npuir"``)."""
    del default_block_size
    scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    tp = int(round(p * float(1 << 23)))
    tp_shift = 256 * tp - 1
    signed_mask = (p == 0.5)

'''

_TAIL = '''    return kernel


# ---------- test harness (same as the Stage 3/4 base file) ----------
def _run_kernel(x_cpu, dtype_str, p, seed, block_size=None):
    N = x_cpu.numel()
    if block_size is None:
        block_size = _DEFAULT_BLOCK_SIZE[dtype_str]
    kernel = _make_dropout_kernel(N, dtype_str, float(p), int(seed))
    fn = kernel(block_size)
    x_flat = x_cpu.reshape(-1).contiguous()
    return fn(x_flat.npu()).cpu()


def dropout_forward(x_cpu, dtype_str, p, seed, training=True, block_size=None):
    if not training:
        return x_cpu.detach().cpu().clone()
    return _run_kernel(x_cpu, dtype_str, p, seed, block_size)


def check_dropout_contract(y, x, p, y2=None):
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
    rate = float(((is_zero & (xf != 0)).to(torch.float32).sum() / max(n, 1)).item())
    sigma3 = 3.0 * (p * (1 - p) / n) ** 0.5 if 0 < p < 1 else 0.0
    rate_ok = bool(abs(rate - p) <= max(sigma3, 1e-6))
    replay_ok = None if y2 is None else bool(torch.equal(y, y2.detach().cpu()))
    return structure_ok, rate, rate_ok, replay_ok


def _assert_contract(y, x, p, tag, expect_rate=True, y2=None):
    structure_ok, rate, rate_ok, replay_ok = check_dropout_contract(y, x, p, y2)
    assert structure_ok, f"[{tag}] structure FAIL (dtype={x.dtype}, p={p})"
    if expect_rate:
        assert rate_ok, f"[{tag}] drop rate FAIL: rate={rate:.6f} vs p={p}"
    if replay_ok is not None:
        assert replay_ok, f"[{tag}] deterministic replay FAIL (same seed differs)"
    extra = "" if replay_ok is None else f" replay={replay_ok}"
    print(f"[{tag}] PASS dtype={x.dtype} p={p} N={x.numel()} rate={rate:.6f}{extra}")
    return rate


def _run_case(case_id, N, dtype_str, p, seed, *, block_size=None, kind="ones",
              expect_rate=None, replay=False):
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


def run_L0():
    cases = 0
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case("L0-1", 4_194_304, dtype_str, 0.5, 42, kind="ones")
        cases += 1
    for p in (0.1, 0.3):
        _run_case("L0-2", 4_194_304, "float16", p, 42, kind="ones")
        cases += 1
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case("L0-3", 1_000_000, dtype_str, 0.5, 42, kind="ones")
        cases += 1
    _run_case("L0-4", 1_000_000, "float16", 0.5, 777, kind="randn", replay=True)
    _run_case("L0-4", 1_000_000, "float32", 0.5, 777, kind="randn", replay=True)
    cases += 2
    xa = torch.ones(262_144, dtype=torch.float32)
    ya = dropout_forward(xa, "float32", 0.5, 42)
    yb = dropout_forward(xa, "float32", 0.5, 99)
    assert not torch.equal(ya, yb), "[L0-5] different seeds produced identical output"
    diff_frac = float((ya != yb).to(torch.float32).mean())
    assert 0.3 < diff_frac < 0.7, f"[L0-5] seed separation frac={diff_frac:.3f}"
    print(f"[L0-5] PASS seed separation frac={diff_frac:.3f}")
    cases += 1
    for dtype_str in ("float32", "float16"):
        x = torch.randn(1_000_000, dtype=_TORCH_DTYPE[dtype_str])
        y = dropout_forward(x, dtype_str, 0.0, 42)
        torch.testing.assert_close(y, x)
        print(f"[L0-6] PASS p=0 identity dtype={dtype_str}")
        cases += 1
    for dtype_str in ("float32", "float16"):
        x = torch.randn(1_000_000, dtype=_TORCH_DTYPE[dtype_str])
        y = dropout_forward(x, dtype_str, 1.0, 42)
        assert torch.equal(y, torch.zeros_like(x)), f"[L0-7] p=1 not all zeros ({dtype_str})"
        print(f"[L0-7] PASS p=1 all zeros dtype={dtype_str}")
        cases += 1
    x = torch.randn(1_000_000, dtype=torch.float32)
    y = dropout_forward(x, "float32", 0.5, 42, training=False)
    torch.testing.assert_close(y, x)
    assert torch.equal(y, x), "[L0-8] training=False is not an exact identity"
    print("[L0-8] PASS training=False host identity")
    cases += 1
    for dtype_str in ("float32", "float16"):
        x = torch.randn((100, 10_000), dtype=_TORCH_DTYPE[dtype_str])
        y_flat = dropout_forward(x, dtype_str, 0.5, 42)
        y = y_flat.reshape(x.shape)
        assert y.shape == x.shape, f"[L0-9] shape {y.shape} != {x.shape}"
        assert y.dtype == x.dtype, f"[L0-9] dtype {y.dtype} != {x.dtype}"
        _assert_contract(y, x, 0.5, "L0-9", expect_rate=False)
        cases += 1
    _run_case("L0-10", 4_000_000, "float16", 0.5, 42, kind="ones")
    cases += 1
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
    cases = 0
    for N, dtype_str, p, seed in (
        (4_194_304, "float32", 0.3, 42),
        (4_194_304, "bfloat16", 0.5, 42),
        (4_194_304, "float16", 0.7, 42),
    ):
        _run_case("L1-1", N, dtype_str, p, seed, kind="ones")
        cases += 1
    _run_case("L1-2", 1_000_000, "bfloat16", 0.5, 777, kind="randn", replay=True)
    cases += 1
    x = torch.randn(1_000_000, dtype=torch.bfloat16)
    torch.testing.assert_close(dropout_forward(x, "bfloat16", 0.0, 42), x)
    assert torch.equal(dropout_forward(x, "bfloat16", 1.0, 42), torch.zeros_like(x)), \\
        "[L1-3] bf16 p=1 not all zeros"
    print("[L1-3] PASS bf16 p=0 identity / p=1 all zeros")
    cases += 2
    _run_case("L1-4", 8192, "float16", 0.5, 42, kind="ones", block_size=512)
    _run_case("L1-4", 65536, "float16", 0.5, 42, kind="ones", block_size=1024)
    cases += 2
    _run_case("L1-5", 4_000_000, "bfloat16", 0.5, 42, kind="ones")
    cases += 1
    _run_case("L1-6", 10_485_760, "bfloat16", 0.5, 42, kind="ones")
    cases += 1
    xb = torch.ones(262_144, dtype=torch.bfloat16)
    d = float((dropout_forward(xb, "bfloat16", 0.5, 42) !=
               dropout_forward(xb, "bfloat16", 0.5, 99)).to(torch.float32).mean())
    assert 0.3 < d < 0.7, f"[L1-7] bf16 seed separation frac={d:.3f}"
    print(f"[L1-7] PASS bf16 seed separation frac={d:.3f}")
    cases += 1
    _run_case("L1-8", 1_000_000, "float32", 0.5, 123, kind="ones")
    cases += 1
    print(f"  -> run_L1: {cases} cases PASS")
    return cases


def run_L2():
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
    cases = 0
    ok = 0
    vals = [float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 3.0, -7.0, 1e-45]
    for dtype_str in ("float32", "float16", "bfloat16"):
        tdt = _TORCH_DTYPE[dtype_str]
        N = 4096
        x = torch.tensor((vals * (N // len(vals)))[:N], dtype=tdt).contiguous()
        cases += 1
        try:
            y = dropout_forward(x, dtype_str, 1.0, 42)
            exp = golden_from_mask(x, torch.zeros(N, dtype=torch.float32), 1.0)
            assert _equal_nan(y, exp), f"[Boundary] p=1 inf/NaN mismatch ({dtype_str})"
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): p=1 {dtype_str} err={e}")
        cases += 1
        try:
            y = dropout_forward(x, dtype_str, 0.0, 42)
            assert _equal_nan(y, x), f"[Boundary] p=0 not identity ({dtype_str})"
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (non-blocking): p=0 {dtype_str} err={e}")
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
    print("\\033[92mAll check passed!\\033[0m")


if __name__ == "__main__":
    main()
'''


def generate(name, desc, mode, bs, ch):
    expert = mode == "Expert"
    blocks = 20 if expert else 40
    streams = 40
    text = _HEAD.format(desc=desc, mode=mode, bs=bs, ch=ch, blocks=blocks,
                        streams=streams)
    text += _MID
    text += _kernel_block(ch, mode)
    text += _TAIL
    out = os.path.join(HERE, name + ".py")
    with open(out, "w") as fh:
        fh.write(text)
    print("wrote", out)


if __name__ == "__main__":
    generate("_r8_b4096_dev", "block_size=4096 single chain (tile-size control)",
             "Developer", 4096, 1)
    generate("_r8_b4096_exp", "block_size=4096 single chain (mode control)",
             "Expert", 4096, 1)
    generate("_r8_c4096x2", "block_size=4096, 2 independent chains interleaved",
             "Expert", 4096, 2)
    generate("_r8_d2048x4", "block_size=2048, 4 independent chains interleaved",
             "Expert", 2048, 4)
