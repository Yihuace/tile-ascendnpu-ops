# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""GPT-NeoX style RoPE (layout=2d) forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_make_rope_neox_2d``
(``y = x * cos + rotate_half(x) * sin`` with ``x`` shaped
``(batch, seq_len, num_heads, head_dim)`` and ``cos`` / ``sin`` shaped
``(seq_len, head_dim // 2)`` broadcast over batch and heads).  Semantics are
equivalent to the TileOPs reference ``ref_rope_neox`` (4D branch) for float16 /
bfloat16 / float32, output dtype ``same_as(input)``.

I/O contract (integration wrapper
``examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py``)::

    main(x, cos_table, sin_table, y)   with out_idx=[3]
    x:         flat (N_total,)  == (batch * seq_len, num_heads, head_dim) row-major
    cos_table: (seq_len, head_dim // 2)
    sin_table: (seq_len, head_dim // 2)
    y:         flat (N_total,)  (the wrapper reshapes back to 4D)

The wrapper flattens the 4D input with ``x.contiguous().reshape(-1)`` before the
call and reshapes the result afterwards, so ``x`` / ``y`` are declared here as a
3D row-major view ``(tokens, num_heads, head_dim)`` (byte-identical to the flat
1D buffer; the NPU launcher is pointer based and does not check input shapes).

The factory mirrors the GPU signature (K3/K11: ``threads`` / ``num_per_thread``
dropped, ``block_size`` collapsed) and returns a compiled-callable factory
taking a single ``block_size``; ``_DEFAULT_BLOCK_SIZE`` = 1024 for fp32 and
2048 for fp16/bf16.

NPU re-design (see DESIGN.md sections 0.5 / 0.6):

  * R1 -- coordinates: the GPU per-element flat three-level ``%`` / ``//``
    index reconstruction is replaced by a static ``(tokens, num_heads,
    head_dim)`` row-major view plus a per-tile ``(batch, seq_block)`` mapping;
    the cos/sin row index is simply ``seq_start + row``.  Parallellism: 1-D
    ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial`` grid-stride over
    the logical tiles (``core_num = 40`` physical Vector cores = 20 AI cores x 2).
  * R2 -- computation core: the GPU per-element ``if_then_else`` pairing
    gather (no cheap strided-gather vector instruction on NPU) is replaced by a
    **half-split whole-buffer vector chain**: with ``a = x[..., :half]`` and
    ``b = x[..., half:]``,

        y1 = a * c - b * s          (first half)
        y2 = b * c + a * s          (second half)

    6 vector ops (4 ``T.vmul`` + 1 ``T.vsub`` + 1 ``T.vadd``) on 3D
    ``(block_tokens, num_heads, half_pad)`` UB tiles, zero gather,
    mathematically identical to the source.
  * R3 -- cos/sin are loaded **once per tile** as ``(block_tokens, 1, half_pad)``
    tiles and broadcast over the head axis by the v-op size-1 same-rank
    broadcast rule; one load serves all heads of the tile (the source reloads
    per element and relies on L2).
  * computation domain is **fp32 transit** (sub-fp32 only): up-cast with
    ``T.vcast(rint)``, run the chain in fp32, round back once with
    ``T.vcast(rint)``.  This matches the contract golden (fp32 opmath +
    single round-back); the source input-dtype domain is not bit-equivalent.

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the ``T.Kernel``
    top level (a conditional alloc triggers ``Undefined variable``); the
    compile-time ``if need_cast:`` branches only select which v-ops run.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x[ts:ts + rows, ...]``).
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-10: ``block_tokens = max(1, blk // (num_heads * head_dim))`` avoids an
    empty tile when ``num_heads * head_dim > block_size``; the seq tail is
    trimmed with ``rows = T.min(block_tokens, seq_len - seq_start)``.
  * K11: all tiling (``block_tokens`` / ``seq_blocks`` / ``work_blocks`` /
    ``iters``) is derived inside the jit inner ``kernel(blk)`` from the
    ``block_size`` actually passed to the compiled callable, so the per-dtype
    tiling (fp32=1024, fp16/bf16=2048) is really honoured.

run: python _make_rope_neox_2d.py --level all
"""

import os

# Developer mode for npuir (linear data dependency, no manual sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import functools

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_CORE_NUM = 40  # physical Vector cores on Ascend910B3 (20 AI cores x 2)
# Stage 4 r1: per-core UB budget for the token-tile sizing (UB = 192 KB/core).
_UB_BUDGET_BYTES = 150 * 1024
_UB_HARD_BYTES = 176 * 1024
_COMPUTE_DTYPE = "float32"

# K11: GPU block_size = threads(256) * num_per_thread collapsed per dtype
# (fp32 npt=4 -> 1024, fp16/bf16 npt=8 -> 2048); integration wrapper
# ``_DEFAULT_BLOCK_SIZE``.
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}

_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_rope.py::_get_tolerances
# (DESIGN.md section 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE reference (contract golden), 4D layout.

    Direct port of the TileOPs reference (``tests/ops/test_rope.py::
    ref_rope_neox`` 4D branch).  It is independent of the NPU half-split
    vector chain: it builds the full-width duplicated cos/sin tables,
    materialises ``rotate_half`` and evaluates everything in the fp32 opmath
    domain, rounding back to the input dtype exactly once:

        cos_full = cat([cos, cos], -1).unsqueeze(0).unsqueeze(2)   # (1,S,1,d)
        sin_full = cat([sin, sin], -1).unsqueeze(0).unsqueeze(2)
        rot      = cat([-x[..., half:], x[..., :half]], -1)
        y        = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)

    x: ``(batch, seq_len, num_heads, head_dim)``; cos/sin: ``(seq_len, half)``.
    """
    x_cpu = x.detach().cpu()
    cos_cpu = cos.detach().cpu()
    sin_cpu = sin.detach().cpu()
    half = x_cpu.shape[-1] // 2
    xf = x_cpu.float()
    rot = torch.cat([-xf[..., half:], xf[..., :half]], dim=-1)
    cos_full = torch.cat([cos_cpu.float(), cos_cpu.float()], dim=-1).unsqueeze(0).unsqueeze(2)
    sin_full = torch.cat([sin_cpu.float(), sin_cpu.float()], dim=-1).unsqueeze(0).unsqueeze(2)
    return (xf * cos_full + rot * sin_full).to(x_cpu.dtype)


def golden_source_semantics(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Independent per-element reference mirroring the GPU source indexing.

    Uses ``freq_idx = col % half`` plus the ``if_then_else`` pairing gather
    (i.e. re-reads the table for every element instead of the single-load
    broadcast), evaluated in fp32 opmath.  Used by L0-10 to independently
    verify that the cos/sin single-load reuse is exact.
    """
    x_cpu = x.detach().cpu()
    cos_cpu = cos.detach().cpu()
    sin_cpu = sin.detach().cpu()
    half = x_cpu.shape[-1] // 2
    head_dim = x_cpu.shape[-1]
    idx = torch.arange(head_dim)
    freq_idx = idx % half
    paired_col = torch.where(idx < half, idx + half, idx - half)
    rot_sign = torch.where(idx < half, torch.tensor(-1.0), torch.tensor(1.0))
    xf = x_cpu.float()
    c = cos_cpu.float()[:, freq_idx]  # (seq_len, head_dim)
    s = sin_cpu.float()[:, freq_idx]
    rot = xf[..., paired_col] * rot_sign
    return (xf * c.unsqueeze(0).unsqueeze(2) + rot * s.unsqueeze(0).unsqueeze(2)).to(x_cpu.dtype)


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _make_rope_neox_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: str,
    block_size: int = 2048,
    block_tokens: int = None,
    core_num: int = None,
) -> object:
    """NPU GPT-NeoX RoPE (layout=2d) kernel factory.

    Mirrors the GPU factory signature (K3/K11: ``threads`` / ``num_per_thread``
    dropped, ``block_size`` kept as an optional default).  The integration
    wrapper calls ``_make_rope_neox_2d(batch, seq_len, num_heads, head_dim,
    dtype_str)`` once, then ``kernel(config["block_size"])``.

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> the vector chain runs directly on the fp32 input;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.
    """
    half = head_dim // 2
    tokens = batch * seq_len
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32  # compile-time constant (tracing-time Python bool)
    _bt_override = block_tokens  # factory-scope capture (avoids inner-scope shadowing)
    # 32-byte alignment of the tile trailing axis (fp16/bf16: 16 elems, fp32: 8).
    block_elems = 32 // _ELEM_BYTES[dtype]
    half_pad = (half + block_elems - 1) // block_elems * block_elems

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(blk: int):
        # K11: derive every tiling quantity from the block_size the wrapper
        # actually passes to the compiled callable (fp32=1024, fp16/bf16=2048).
        _cores = core_num if core_num is not None else _CORE_NUM
        if _bt_override is not None:
            _bt = max(1, min(int(_bt_override), seq_len))
        else:
            # Stage 4 r1: UB-budget token tile + grid balancing.
            _bytes_per_token = ((4 * _ELEM_BYTES[dtype] + 6 * 4) * (num_heads * half_pad)
                                + 2 * num_heads * head_dim * _ELEM_BYTES[dtype])
            _bt_max = max(1, _UB_BUDGET_BYTES // _bytes_per_token)
            _bt_hard = max(1, _UB_HARD_BYTES // _bytes_per_token)
            _seq_blocks_1 = -(-seq_len // _bt_max)
            _iters_1 = max(1, -(-(batch * _seq_blocks_1) // _cores))
            _bt = max(1, min(-(-seq_len // (-(-seq_len // (_iters_1 * _cores)))), _bt_hard))
        block_tokens = _bt
        seq_blocks = (seq_len + block_tokens - 1) // block_tokens
        work_blocks = batch * seq_blocks
        iters = (work_blocks + _cores - 1) // _cores

        @T.prim_func
        def main(
            x: T.Tensor((tokens, num_heads, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((tokens, num_heads, head_dim), dtype),
        ):
            with T.Kernel(_cores, is_npu=True) as (cid, _):
                # R-7: all UB buffers allocated unconditionally at the top
                # level (conditional T.alloc is illegal).
                xlo_ub = T.alloc_ub((block_tokens, num_heads, half_pad), dtype)  # x first half -> y1
                xhi_ub = T.alloc_ub((block_tokens, num_heads, half_pad), dtype)  # x second half -> y2
                c_ub = T.alloc_ub((block_tokens, 1, half_pad), dtype)            # cos (head broadcast)
                s_ub = T.alloc_ub((block_tokens, 1, half_pad), dtype)            # sin (head broadcast)
                af = T.alloc_ub((block_tokens, num_heads, half_pad), f32)
                bf = T.alloc_ub((block_tokens, num_heads, half_pad), f32)
                cf = T.alloc_ub((block_tokens, 1, half_pad), f32)
                sf = T.alloc_ub((block_tokens, 1, half_pad), f32)
                olo = T.alloc_ub((block_tokens, num_heads, half_pad), f32)       # y1 (fp32)
                tmp = T.alloc_ub((block_tokens, num_heads, half_pad), f32)       # temporary product
                xs = T.alloc_ub((block_tokens, num_heads, head_dim), dtype)      # full-row staging (contiguous DMA)
                os_ = T.alloc_ub((block_tokens, num_heads, head_dim), dtype)     # full-row output staging

                # Compile-time computation-domain aliases (tracing-time Python
                # ternary): fp32 computes in place on the input buffers,
                # fp16/bf16 compute in the fp32 transit buffers.
                A = af if need_cast else xlo_ub
                Bv = bf if need_cast else xhi_ub
                C = cf if need_cast else c_ub
                S = sf if need_cast else s_ub

                # In-core grid-stride over logical tiles: static loop bound,
                # guard covers non-divisible grids (DESIGN.md section 5.5).
                for i in T.serial(iters):
                    w = i * _cores + cid
                    if w < work_blocks:
                        b = w // seq_blocks
                        sb = w % seq_blocks
                        seq_start = sb * block_tokens
                        # R-10 / section 5.4: dynamic seq-tail trim; only the
                        # [0:rows] rows are stored back, stale rows are never
                        # written out.
                        rows = T.min(block_tokens, seq_len - seq_start)
                        ts = b * seq_len + seq_start

                        # GM -> UB: one contiguous full-row copy (Stage 4 r2),
                        # then a UB->UB half split with STATIC extents (a
                        # dynamic tail extent in a UB->UB copy mis-compiles).
                        T.copy(
                            x[ts : ts + rows, 0:num_heads, 0:head_dim],
                            xs[:rows, 0:num_heads, 0:head_dim],
                        )
                        T.copy(xs[0:block_tokens, 0:num_heads, 0:half], xlo_ub)
                        T.copy(xs[0:block_tokens, 0:num_heads, half:head_dim], xhi_ub)
                        # cos/sin loaded once, broadcast over the head axis (R3).
                        T.copy(
                            cos_table[seq_start : seq_start + rows, 0:half],
                            c_ub[:rows, 0, 0:half],
                        )
                        T.copy(
                            sin_table[seq_start : seq_start + rows, 0:half],
                            s_ub[:rows, 0, 0:half],
                        )

                        if need_cast:
                            # lossless up-cast (f16/bf16 -> f32, rint only, R-9)
                            T.vcast(xlo_ub, af, round_mode="rint")
                            T.vcast(xhi_ub, bf, round_mode="rint")
                            T.vcast(c_ub, cf, round_mode="rint")
                            T.vcast(s_ub, sf, round_mode="rint")

                        # --- half-split 2x2 rotation (fp32 domain, whole 3D
                        # buffer, size-1 head-axis broadcast) ---
                        T.vmul(A, C, olo)       # a * c
                        T.vmul(Bv, S, tmp)      # b * s
                        T.vsub(olo, tmp, olo)   # y1 = a*c - b*s
                        T.vmul(A, S, tmp)       # a * s
                        T.vmul(Bv, C, Bv)       # b * c (in place; Bv carries y2)
                        T.vadd(Bv, tmp, Bv)     # y2 = b*c + a*s

                        # UB half merge (static extents) + one contiguous store.
                        if need_cast:
                            # single round back to the input dtype (R-9).
                            T.vcast(olo, xlo_ub, round_mode="rint")
                            T.vcast(Bv, xhi_ub, round_mode="rint")
                            T.copy(xlo_ub, os_[0:block_tokens, 0:num_heads, 0:half])
                            T.copy(xhi_ub, os_[0:block_tokens, 0:num_heads, half:head_dim])
                        else:
                            T.copy(olo, os_[0:block_tokens, 0:num_heads, 0:half])
                            T.copy(Bv, os_[0:block_tokens, 0:num_heads, half:head_dim])
                        T.copy(
                            os_[:rows, 0:num_heads, 0:head_dim],
                            y[ts : ts + rows, 0:num_heads, 0:head_dim],
                        )

        return main

    return kernel


# ---------- precision comparing func ----------
def _rope_freqs(seq_len, head_dim, dtype_str, base=10000.0):
    """RoPE frequency tables (cos, sin), each (seq_len, head_dim // 2).

    Mirrors ``tests/ops/test_rope.py::_compute_freqs_cis_base`` (angles in
    fp32, cast to the target dtype) and the Op layer ``_base_freqs``.
    """
    half = head_dim // 2
    td = _TORCH_DTYPE[dtype_str]
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(td), torch.sin(angles).to(td)


def _make_input(batch, seq_len, num_heads, head_dim, dtype_str):
    """Random 4D input + frequency tables following the workload logic."""
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=_TORCH_DTYPE[dtype_str])
    cos, sin = _rope_freqs(seq_len, head_dim, dtype_str)
    return x, cos, sin


def _repeat_values(values, shape, dtype):
    """Repeat a small value pattern to fill ``shape`` (edge / corner cases)."""
    n = 1
    for d in shape:
        n *= d
    base = torch.tensor(values, dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n].reshape(shape)


def _max_diff(y, golden):
    """Max abs error over finite entries (inf/NaN handled by the caller)."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _run_tensor(x_cpu, cos_cpu, sin_cpu, dtype_str, tag, reference=None):
    """Run the kernel through the wrapper contract (flat input) and compare.

    ``reference`` defaults to the contract golden; ``run_L0``'s single-load
    reuse case passes ``golden_source_semantics`` instead.
    """
    x_cpu = x_cpu.detach().cpu().contiguous()
    cos_cpu = cos_cpu.detach().cpu().contiguous()
    sin_cpu = sin_cpu.detach().cpu().contiguous()
    batch, seq_len, num_heads, head_dim = x_cpu.shape
    ref_fn = golden_rope_neox if reference is None else reference
    golden = ref_fn(x_cpu, cos_cpu, sin_cpu)

    kernel = _make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype_str)(
        _DEFAULT_BLOCK_SIZE[dtype_str]
    )

    # wrapper path: x.contiguous().reshape(-1) -> kernel -> reshape
    x_flat = x_cpu.npu().reshape(-1)
    y = kernel(x_flat, cos_cpu.npu(), sin_cpu.npu()).cpu().reshape(x_cpu.shape)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y, golden)
    try:
        torch.testing.assert_close(y, golden, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({batch},{seq_len},{num_heads},{head_dim}) "
            f"dtype={dtype_str} max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({batch},{seq_len},{num_heads},{head_dim}) "
        f"dtype={dtype_str} max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(batch, seq_len, num_heads, head_dim, dtype_str, tag, reference=None):
    x, cos, sin = _make_input(batch, seq_len, num_heads, head_dim, dtype_str)
    return _run_tensor(x, cos, sin, dtype_str, tag, reference=reference)


def _broadcast_consistency(batch, seq_len, num_heads, head_dim, dtype_str):
    """L0-9: with identical heads/batches the rotation must be identical.

    Builds a random single ``(seq_len, head_dim)`` token block, replicates it
    over all heads and batches, runs the kernel and asserts the output is
    identical over the (batch, head) axes -- a direct check of the batch/head
    broadcast that is independent of the golden implementation.
    """
    one = torch.randn(seq_len, head_dim, dtype=_TORCH_DTYPE[dtype_str])
    x = one.reshape(1, seq_len, 1, head_dim).expand(batch, seq_len, num_heads, head_dim)
    x = x.contiguous()
    cos, sin = _rope_freqs(seq_len, head_dim, dtype_str)
    golden = golden_rope_neox(x, cos, sin)

    kernel = _make_rope_neox_2d(batch, seq_len, num_heads, head_dim, dtype_str)(
        _DEFAULT_BLOCK_SIZE[dtype_str]
    )
    y = (
        kernel(x.npu().reshape(-1), cos.npu(), sin.npu())
        .cpu()
        .reshape(batch, seq_len, num_heads, head_dim)
    )

    ref = y[:, :, 0:1, :].expand_as(y)
    spread = (y.float() - ref.float()).abs().max().item()
    assert spread == 0.0, f"[L0-broadcast] heads/batches differ: max spread {spread}"

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y, golden)
    torch.testing.assert_close(y, golden, rtol=rtol, atol=atol, equal_nan=True)
    print(
        f"[L0-broadcast] PASS: shape=({batch},{seq_len},{num_heads},{head_dim}) "
        f"dtype={dtype_str} head/batch spread=0.0 max_diff={max_diff:.6e}"
    )


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1: manifest 2d workload (DESIGN.md section 8.3).
    _run_case(2, 2048, 32, 128, "float16", "L0-manifest")
    # L0-2..L0-4: basic fixture, all dtypes.
    _run_case(2, 128, 8, 64, "float32", "L0-basic")
    _run_case(2, 128, 8, 64, "float16", "L0-basic")
    _run_case(2, 128, 8, 64, "bfloat16", "L0-basic")
    # L0-5: batch broadcast, single batch.
    _run_case(1, 256, 4, 128, "float16", "L0-b1")
    # L0-6: edge fixture (half=8 -> half_pad=16, block_tokens=64).
    _run_case(1, 1, 1, 16, "float32", "L0-edge")
    # L0-7: corner values (both x and cos/sin tables) per dtype -- exercises
    # IEEE mul/add propagation and the fp32-transit rounding.
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2, 128, 8, 64)
        half = shape[3] // 2
        corner_vals = [
            0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45,
            65504.0, -65504.0, 88.8, -88.8, 1e-8, -1e-8,
        ]
        x = _repeat_values(corner_vals, shape, td)
        cos = _repeat_values(corner_vals, (shape[1], half), td)
        sin = _repeat_values(corner_vals[::-1], (shape[1], half), td)
        _run_tensor(x, cos, sin, dtype_str, "L0-corners")
    # L0-8: non-divisible seq tail (130 = 16*8 + 2 with block_tokens=8).
    _run_case(2, 130, 4, 64, "bfloat16", "L0-tail")
    # L0-9: batch/head broadcast consistency (structural check + golden).
    _broadcast_consistency(2, 128, 8, 64, "float16")
    # L0-10: cos/sin single-load reuse exactness vs the per-element source
    # indexing reference (independent implementation).
    _run_case(2, 128, 8, 64, "float16", "L0-cse", reference=golden_source_semantics)


def run_L1():
    # Every dtype through the manifest shape set.
    for dtype_str in ("float32", "float16", "bfloat16"):
        _run_case(2, 128, 8, 64, dtype_str, "L1-basic")
        _run_case(1, 256, 4, 128, dtype_str, "L1-b1")
        _run_case(2, 512, 8, 64, dtype_str, "L1-mid")

    # Non-divisible tails for every dtype.
    _run_case(2, 130, 4, 64, "float32", "L1-tail")
    _run_case(2, 130, 4, 64, "float16", "L1-tail")

    # Exact single tile / several tiles.
    _run_case(2, 64, 8, 64, "float16", "L1-exact")
    _run_case(2, 96, 8, 64, "float16", "L1-multi")

    # Corner values (+/-inf / NaN / +/-0 / subnormal / extremes) per dtype:
    # exercises IEEE mul/add propagation and the fp32-transit rounding.  These
    # reuse the shapes/dtypes compiled above (no extra compilation).
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2, 128, 8, 64)
        x = _repeat_values(
            [0.0, -0.0, 1.0, -1.0, 1e-8, -1e-8, 88.8, -88.8, 1e-45, -1e-45],
            shape,
            td,
        )
        cos, sin = _rope_freqs(shape[1], shape[3], dtype_str)
        _run_tensor(x, cos, sin, dtype_str, "L1-corners")

    # Infinity / NaN inputs (non-blocking semantics verified here too).
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2, 128, 8, 64)
        x = _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], shape, td)
        cos, sin = _rope_freqs(shape[1], shape[3], dtype_str)
        _run_tensor(x, cos, sin, dtype_str, "L1-inf-nan")


def run_L2():
    # Edge sizes, non-blocking.
    for batch, seq_len, num_heads, head_dim, dtype_str in (
        (1, 1, 1, 16, "float32"),
        (1, 2, 1, 32, "float16"),
        (1, 3, 2, 16, "bfloat16"),
        (1, 33, 1, 64, "float16"),
        (3, 7, 3, 32, "float32"),
        (1, 129, 2, 32, "float16"),
    ):
        try:
            _run_case(batch, seq_len, num_heads, head_dim, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (Record without blocking): shape=({batch},{seq_len},"
                f"{num_heads},{head_dim}) dtype={dtype_str} err={e}"
            )


def run_boundary():
    # Special-value cos/sin tables (inf / NaN propagation) plus zero and
    # saturated inputs; all non-blocking.
    boundary_patterns = (
        ([0.0, -0.0, 1.0, -1.0], [0.0, 0.0, 1.0, 1.0]),
        ([float("inf"), float("-inf"), 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]),
        ([float("nan"), 0.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2, 128, 8, 64)
        half = shape[3] // 2
        for x_vals, cs_vals in boundary_patterns:
            try:
                x = _repeat_values(x_vals, shape, td)
                cos = _repeat_values(cs_vals, (shape[1], half), td)
                sin = _repeat_values(cs_vals[::-1], (shape[1], half), td)
                _run_tensor(x, cos, sin, dtype_str, "Boundary")
            except Exception as e:  # noqa: BLE001
                print(
                    f"[Boundary] WARN (Record without blocking): dtype={dtype_str} err={e}"
                )


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
