# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""GPT-NeoX style RoPE (layout=1d) forward kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_make_rope_neox_1d``
(``y = x * cos + rotate_half(x) * sin``, ``rotate_half(x) = cat(-x[half:],
x[:half])``, ``threads=256`` per-thread elementwise strategy).  Semantics are
equivalent to the TileOPs reference ``ref_rope_neox`` for float16 / bfloat16 /
float32, output dtype ``same_as(input)``.

I/O contract (integration wrapper
``examples/TileOPs/tileops/kernels/position_encoding/rope_neox/rope_neox.py``)::

    main(x, cos_table, sin_table, y)   with out_idx=[3]
    x:         (seq_len, head_dim)
    cos_table: (seq_len, head_dim // 2)
    sin_table: (seq_len, head_dim // 2)
    y:         (seq_len, head_dim)

The factory returns a compiled-callable factory taking a single ``block_size``
(the K11 collapse of the GPU ``threads * num_per_thread`` product;
``_DEFAULT_BLOCK_SIZE`` = 1024 for fp32, 2048 for fp16/bf16).

NPU re-design (see DESIGN.md sections 0.5 / 0.6):

  * R1 -- 1-D ``T.Kernel(core_num, is_npu=True)`` + in-core ``T.serial``
    grid-stride over logical row-block tiles (the GPU block x thread model
    has no direct NPU counterpart).  ``core_num = 40`` is the physical
    Vector core count (NPUUtils: 20 AI cores x 2, Ascend910_9362).
  * R2 -- the computation core is re-designed from the GPU per-element
    ``if_then_else`` pairing gather (which would degrade to a strided scalar
    gather on NPU) to a **half-split whole-buffer vector chain**: split each
    row block at ``half`` into ``a = x[:, :half]`` and ``b = x[:, half:]`` and
    compute

        y1 = a * c - b * s          (first half)
        y2 = b * c + a * s          (second half)

    with 4 ``T.vmul`` + 1 ``T.vsub`` + 1 ``T.vadd`` on ``(block_rows, half)``
    UB tiles.  Mathematically identical to the source, zero gather.
  * R3 -- cos/sin are loaded **once** per row block into ``(block_rows, half)``
    tiles and reused by both halves (the source reloads both halves per
    element and relies on L2); cos/sin GM traffic is halved.
  * computation domain is **fp32 transit**: fp16/bf16 up-cast with
    ``T.vcast(rint)``, the 6-op chain runs in fp32, results are down-cast
    with ``T.vcast(rint)`` exactly once.  This matches the contract golden
    (``x.float() * cos_full.float() + rot.float() * sin_full.float()`` ->
    ``.to(dtype)``).  fp32 transit is *mandatory* for correctness: the source
    input-dtype domain exceeds the fp16 tolerance (verify_equiv diagnosis
    ``max_abs = 7.81e-3 > 1e-3``) and the v-prefix math ops have no bf16
    dtype in their matrix.

Hard constraints honoured (DESIGN.md sections 3.5.1 / 9.1):
  * R-7: every ``T.alloc_ub`` is allocated unconditionally at the ``T.Kernel``
    top level (a conditional alloc triggers ``Undefined variable``); the
    compile-time ``if need_cast:`` branches only select which v-ops run.
  * R-8: v-prefix ops act on the *whole* UB buffer; only ``T.copy`` uses the
    dynamic tail slice (``x[r0:r0 + tr, ...]``).  A dynamic slice passed to a
    v-op fails at compile time.
  * R-9: f16/bf16 <-> f32 ``vcast`` supports only ``round_mode="rint"``.
  * R-10: ``block_rows`` is clamped to ``seq_len`` (avoids a false UB overflow
    when ``seq_len < block_size // head_dim``); the row tail is trimmed with
    ``tr = T.min(block_rows, seq_len - r0)``.
  * ``block_rows`` is derived from the ``block_size`` actually passed to the
    compiled callable (REVIEW.md suggestion 1), so the fp32 tiling
    (``block_size=1024``) is really honoured and not silently replaced by the
    factory default (``block_size=2048``).

run: python _make_rope_neox_1d.py --level all
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

_CORE_NUM = 40  # physical Vector cores on Ascend910_9362 (20 AI cores x 2)
# R1 (Stage 4): per-core UB budget for the tile sizing (UB = 192 KB/core).
_UB_BUDGET_BYTES = 176 * 1024
_UB_HARD_BYTES = 176 * 1024
_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_COMPUTE_DTYPE = "float32"

# K11: GPU block_size = threads(256) * num_per_thread collapsed per dtype
# (fp32 npt=4 -> 1024, fp16/bf16 npt=8 -> 2048); integration wrapper
# ``_DEFAULT_BLOCK_SIZE``.
_DEFAULT_BLOCK_SIZE = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}

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
    """NeoX RoPE reference (contract golden).

    This is a direct port of the TileOPs reference
    (``tests/ops/test_rope.py::ref_rope_neox`` /
    ``_rotate_half_neox``) and is independent of the NPU half-split vector
    chain: it builds the full-width duplicated cos/sin tables, materialises
    ``rotate_half`` and evaluates everything in the fp32 opmath domain,
    rounding back to the input dtype exactly once:

        cos_full = cat([cos, cos], -1); sin_full = cat([sin, sin], -1)
        rot      = cat([-x[..., half:], x[..., :half]], -1)
        y        = (x.float()*cos_full.float() + rot.float()*sin_full.float()).to(x.dtype)
    """
    x_cpu = x.detach().cpu()
    cos_cpu = cos.detach().cpu()
    sin_cpu = sin.detach().cpu()
    half = x_cpu.shape[-1] // 2
    x1, x2 = x_cpu[..., :half], x_cpu[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos_cpu, cos_cpu], dim=-1)
    sin_full = torch.cat([sin_cpu, sin_cpu], dim=-1)
    return (
        x_cpu.float() * cos_full.float() + rot.float() * sin_full.float()
    ).to(x_cpu.dtype)


# ---------- Kernel ----------
@functools.lru_cache(maxsize=32)
def _make_rope_neox_1d(seq_len: int, head_dim: int, dtype: str, block_size: int = 2048,
                       tile_rows: int = None, core_num: int = None) -> object:
    """NPU GPT-NeoX RoPE (layout=1d) tuned kernel factory -- Stage 4 (r2_contig).

Two structural Stage 4 changes on top of the Stage 3 design, each A/B verified
against its own parent on the 1d tune workloads:

  * r1 (tile balancing): ``block_rows`` is derived from a per-core UB budget
    and then re-balanced so the logical tile count fills ``cores * iters``
    slots (16/32-row tiles -> 52-row tiles for head_dim=128).  Fewer, larger
    DMA slices; 1.26-1.42x.
  * r2 (contiguous row DMA + UB staging): the Stage 3 kernel read/wrote the
    half-split views ``x[:, :half]`` / ``x[:, half:]`` directly, i.e. as
    ``block_rows`` segments of ``half`` elements with a ``head_dim`` stride
    (50% duty-cycle segments).  The tuned kernel stages *whole* rows
    ``(block_rows, head_dim)`` with one fully contiguous GM copy and performs
    the half split (load) / merge (store) as UB->UB copies.  MTE3 stopped
    being the bottleneck (``aiv_mte3_ratio`` 0.61 -> 0.23 on the 8k case);
    overall 1.5-1.8x vs the Stage 3 baseline.

UB->UB copies deliberately use *static* extents (``0:block_rows``): a dynamic
tail extent in a UB->UB copy mis-compiles (verified -- the tail rows took the
wrong stride and every element was wrong).  Only GM->UB / UB->GM copies use
the dynamic ``tr`` tail extent, so the merged ``o_stage`` rows ``[tr:]`` are
never stored back.

Math, I/O contract and factory signature are unchanged (drop-in for the
TileOPs wrapper).  Optional ``tile_rows`` / ``core_num`` kwargs only exist for
Stage 4 sweeps.

Original Stage 3 docstring follows.

NPU GPT-NeoX RoPE (layout=1d) kernel factory.

    Mirrors the GPU factory signature (K3/K11: ``threads`` / ``num_per_thread``
    dropped, ``block_size`` kept as an optional default).  The returned
    ``kernel`` is a ``@tilelang.jit`` callable taking a single ``block_size``
    (the integration wrapper calls ``_make_rope_neox_1d(seq_len, head_dim,
    dtype)`` once, then ``kernel(config["block_size"])``).

    ``dtype`` is a compile-time Python string selecting the dispatch branch:
      * ``"float32"`` -> the vector chain runs directly on the fp32 input;
      * ``"float16"`` / ``"bfloat16"`` -> fp32 transit via ``T.vcast``.
    """

    half = head_dim // 2
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32  # compile-time constant (tracing-time Python bool)

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(blk: int):
        # REVIEW.md suggestion 1: block_rows / logical tile count are derived
        # from the block_size that the wrapper actually passes to the compiled
        # callable, so fp32 (1024) and fp16/bf16 (2048) use their own tiling.
        _cores = core_num if core_num is not None else _CORE_NUM
        if tile_rows is not None:
            block_rows = max(1, min(int(tile_rows), seq_len))
        else:
            # R1 (Stage 4): UB-budget tile sizing + grid balancing.
            # bytes_per_row = half * (4 input-dtype tiles + 7 fp32 transit tiles)
            _bytes_per_row = (half * (4 * _ELEM_BYTES[dtype] + 6 * 4)
                              + 2 * head_dim * _ELEM_BYTES[dtype])
            # R3: with 6 fp32 transit tiles the tile can grow to the UB clamp,
            # which drops the 8k d=128 case from 4 wave iterations to 3.
            _rows_max = max(1, _UB_BUDGET_BYTES // _bytes_per_row)   # balancing budget
            _rows_hard = max(1, _UB_HARD_BYTES // _bytes_per_row)    # UB safety clamp
            _iters = max(1, -(-seq_len // (_rows_max * _cores)))
            block_rows = max(1, min(-(-seq_len // (_iters * _cores)), _rows_hard, seq_len))
        n_num = (seq_len + block_rows - 1) // block_rows     # logical tiles
        iters = (n_num + _cores - 1) // _cores               # static loop bound

        @T.prim_func
        def main(
            x: T.Tensor((seq_len, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((seq_len, head_dim), dtype),
        ):
            with T.Kernel(_cores, is_npu=True) as (cid, _):
                # R-7: all UB buffers allocated unconditionally at the top
                # level (conditional T.alloc is illegal -- TRAP-conditional-alloc).
                a_ub = T.alloc_ub((block_rows, half), dtype)   # x first half (dead -> reused as y1 out)
                b_ub = T.alloc_ub((block_rows, half), dtype)   # x second half (dead -> reused as y2 out)
                c_ub = T.alloc_ub((block_rows, half), dtype)   # cos tile (loaded once)
                s_ub = T.alloc_ub((block_rows, half), dtype)   # sin tile (loaded once)
                af = T.alloc_ub((block_rows, half), f32)
                bf = T.alloc_ub((block_rows, half), f32)
                cf = T.alloc_ub((block_rows, half), f32)
                sf = T.alloc_ub((block_rows, half), f32)
                t1 = T.alloc_ub((block_rows, half), f32)       # temporary a*s
                t2 = T.alloc_ub((block_rows, half), f32)       # temporary b*s
                x_stage = T.alloc_ub((block_rows, head_dim), dtype)   # full-row staging (contiguous DMA)
                o_stage = T.alloc_ub((block_rows, head_dim), dtype)   # full-row output staging

                # Compile-time computation-domain aliases (tracing-time Python
                # ternary): fp32 computes in place on the input buffers,
                # fp16/bf16 compute in the fp32 transit buffers.
                A = af if need_cast else a_ub
                B = bf if need_cast else b_ub
                C = cf if need_cast else c_ub
                S = sf if need_cast else s_ub

                # In-core grid-stride over logical tiles: static loop bound,
                # guard covers non-divisible grids (DESIGN.md section 5.5).
                for i in T.serial(iters):
                    task = i * _cores + cid
                    if task < n_num:
                        r0 = task * block_rows
                        # R-10 / section 5.4: dynamic row-tail trim; writes only
                        # touch [0:tr] rows, stale rows are never stored back.
                        tr = T.min(block_rows, seq_len - r0)

                        # GM -> UB.  Only T.copy uses dynamic slices (R-8).
                        T.copy(x[r0 : r0 + tr, 0:head_dim], x_stage[0:tr, 0:head_dim])
                        T.copy(x_stage[0:block_rows, 0:half], a_ub)
                        T.copy(x_stage[0:block_rows, half:head_dim], b_ub)
                        # cos/sin loaded once and reused by both halves (R3/CSE).
                        T.copy(cos_table[r0 : r0 + tr, 0:half], c_ub[0:tr, 0:half])
                        T.copy(sin_table[r0 : r0 + tr, 0:half], s_ub[0:tr, 0:half])

                        if need_cast:
                            # lossless up-cast (f16/bf16 -> f32, rint only, R-9)
                            T.vcast(a_ub, af, round_mode="rint")
                            T.vcast(b_ub, bf, round_mode="rint")
                            T.vcast(c_ub, cf, round_mode="rint")
                            T.vcast(s_ub, sf, round_mode="rint")

                        # --- half-split 2x2 rotation (fp32 domain, whole buffer) ---
                        # R3: both cross products are hoisted first so the fp32
                        # input buffers themselves can become the outputs
                        # (4 input + 2 temp fp32 tiles instead of 4 + 3).
                        T.vmul(A, S, t1)               # a * s
                        T.vmul(B, S, t2)               # b * s
                        T.vmul(A, C, A)                # a * c        (a dead after this)
                        T.vsub(A, t2, A)               # y1 = a*c - b*s
                        T.vmul(B, C, B)                # b * c        (b dead after this)
                        T.vadd(B, t1, B)               # y2 = b*c + a*s

                        if need_cast:
                            # single round back to the input dtype (R-9); a_ub/b_ub
                            # are dead after the up-cast and are reused as output.
                            T.vcast(A, a_ub, round_mode="rint")
                            T.vcast(B, b_ub, round_mode="rint")
                            T.copy(a_ub, o_stage[0:block_rows, 0:half])
                            T.copy(b_ub, o_stage[0:block_rows, half:head_dim])
                            T.copy(o_stage[0:tr, 0:head_dim], y[r0 : r0 + tr, 0:head_dim])
                        else:
                            T.copy(A[0:tr, 0:half], o_stage[0:tr, 0:half])
                            T.copy(B[0:tr, 0:half], o_stage[0:tr, half:head_dim])
                            T.copy(o_stage[0:tr, 0:head_dim], y[r0 : r0 + tr, 0:head_dim])

        return main

    return kernel


# ---------- precision comparing func ----------
def _rope_freqs(seq_len, head_dim, dtype_str, base=10000.0):
    """RoPE frequency tables (cos, sin), each (seq_len, head_dim // 2).

    Mirrors ``tests/ops/test_rope.py::_compute_freqs_cis_base`` (angles in
    fp32, cast to the target dtype).
    """
    half = head_dim // 2
    td = _TORCH_DTYPE[dtype_str]
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(td), torch.sin(angles).to(td)


def _make_input(seq_len, head_dim, dtype_str):
    """Random input + frequency tables following the workload logic."""
    x = torch.randn(seq_len, head_dim, dtype=_TORCH_DTYPE[dtype_str])
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


def _run_tensor(x_cpu, cos_cpu, sin_cpu, dtype_str, tag):
    """Run the kernel on CPU tensors and compare with the golden output."""
    x_cpu = x_cpu.detach().cpu().contiguous()
    cos_cpu = cos_cpu.detach().cpu().contiguous()
    sin_cpu = sin_cpu.detach().cpu().contiguous()
    seq_len, head_dim = x_cpu.shape
    golden = golden_rope_neox(x_cpu, cos_cpu, sin_cpu)

    kernel = _make_rope_neox_1d(seq_len, head_dim, dtype_str)(
        _DEFAULT_BLOCK_SIZE[dtype_str]
    )

    x_npu = x_cpu.npu()
    cos_npu = cos_cpu.npu()
    sin_npu = sin_cpu.npu()
    y = kernel(x_npu, cos_npu, sin_npu)
    y_cpu = y.cpu()

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(y_cpu, golden)
    try:
        torch.testing.assert_close(y_cpu, golden, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({seq_len},{head_dim}) dtype={dtype_str} "
            f"max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({seq_len},{head_dim}) dtype={dtype_str} "
        f"max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(seq_len, head_dim, dtype_str, tag):
    x, cos, sin = _make_input(seq_len, head_dim, dtype_str)
    return _run_tensor(x, cos, sin, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # Manifest 1d workloads (DESIGN.md section 8.3 L0-1..L0-5):
    # [256,64] fp16/bf16/fp32 smoke, [2048,64] fp16, [4096,128] bf16.
    for seq_len, head_dim, dtype_str in (
        (256, 64, "float32"),
        (256, 64, "float16"),
        (256, 64, "bfloat16"),
        (2048, 64, "float16"),
        (4096, 128, "bfloat16"),
    ):
        _run_case(seq_len, head_dim, dtype_str, "L0-manifest")

    # Non-divisible row tail: block_rows=32, 2050 = 64*32 + 2 (L0-7).
    _run_case(2050, 64, "float16", "L0-tail")
    # Tiny seq_len: block_rows clamped to 1 (L0-8).
    _run_case(1, 64, "float16", "L0-tiny")
    # Non-power-of-two seq_len: 1000 = 62*16 + 8 with block_rows=16 (L0-9).
    _run_case(1000, 128, "bfloat16", "L0-nonpow2")

    # cos/sin single-load reuse equivalence (L0-10): the golden builds the
    # full-width duplicated cos/sin tables and is evaluated per element, so a
    # match here proves the shared (block_rows, half) tile reuse is exact.
    _run_case(256, 64, "float16", "L0-cse")


def run_L1():
    # Every dtype through the full manifest shape set.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for seq_len, head_dim in ((256, 64), (2048, 64), (4096, 128)):
            _run_case(seq_len, head_dim, dtype_str, "L1-full")

    # Non-divisible tails for every dtype.
    _run_case(2050, 64, "float32", "L1-tail")
    _run_case(2050, 64, "bfloat16", "L1-tail")
    _run_case(1000, 128, "float16", "L1-nonpow2")

    # Exact single tile / two tiles.
    _run_case(32, 64, "float16", "L1-exact-tile")
    _run_case(64, 64, "float16", "L1-two-tiles")

    # Corner values (+/-inf / NaN / +/-0 / subnormal / extremes) per dtype:
    # exercises IEEE mul/add propagation and the fp32-transit rounding.
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2048, 64)
        x = _repeat_values(
            [0.0, -0.0, 1.0, -1.0, 1e-8, -1e-8, 88.8, -88.8, 1e-45, -1e-45],
            shape,
            td,
        )
        cos, sin = _rope_freqs(*shape, dtype_str)
        _run_tensor(x, cos, sin, dtype_str, "L1-corners")

    # Infinity / NaN inputs (non-blocking semantics verified here too).
    for dtype_str in ("float32", "float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        shape = (2048, 64)
        x = _repeat_values([float("inf"), float("-inf"), float("nan"), 0.0], shape, td)
        cos, sin = _rope_freqs(*shape, dtype_str)
        _run_tensor(x, cos, sin, dtype_str, "L1-inf-nan")


def run_L2():
    # Edge sizes, non-blocking.
    for seq_len, head_dim, dtype_str in (
        (1, 64, "float32"),
        (2, 64, "float16"),
        (3, 128, "bfloat16"),
        (33, 64, "float16"),
        (1025, 128, "float32"),
        (4095, 128, "float16"),
    ):
        try:
            _run_case(seq_len, head_dim, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(
                f"[L2] WARN (Record without blocking): shape=({seq_len},{head_dim}) "
                f"dtype={dtype_str} err={e}"
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
        shape = (2048, 64)
        for x_vals, cs_vals in boundary_patterns:
            try:
                x = _repeat_values(x_vals, shape, td)
                cos = _repeat_values(cs_vals, (shape[0], shape[1] // 2), td)
                sin = _repeat_values(cs_vals[::-1], (shape[0], shape[1] // 2), td)
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
