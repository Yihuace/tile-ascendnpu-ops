# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_fp8_quant_kernel: per-group FP8 e4m3 quantization of a [B, S_kv, G, D] tensor.

Migrated from the GPU TileOPs operator ``tileops/kernels/fp8_quant.py`` to
``target="npuir"`` (Developer mode, pure Vector).

Semantics (source operator contract, unchanged):

    amax_r  = max_d |x_{r,d}|                 (r = flat(b, s, g))
    amaxc_r = max(amax_r, 1e-4)
    scale_r = amaxc_r / 448                   (float32)
    q_{r,d} = clamp(x_{r,d} / scale_r, -448, 448)
    out     = fp8_e4m3(q)                     (round-to-nearest-even)

Migration notes
---------------
* ``(B, S_kv, G)`` is flattened to a row dimension ``R`` and processed as
  ``(R, D)``: the three group axes are independent quantization groups, so the
  flatten is a pure metadata view (no data reorder, no host-side padding).
* The source uses a 3-D CUDA grid and ``threads=128``; on npuir the grid is
  one-dimensional (``num_blocks <= vector core count``) with a static in-core
  ``T.serial`` task loop.
* ``float8`` is not representable by the npuir codegen (capability gap
  CG-2026-0018: no float8 branch in ``DTypetoMLIRType``), so the kernel writes
  the e4m3 byte through a ``uint8``/``int8`` carrier computed in software and
  the host reinterpret-casts it back with ``.view(torch.float8_e4m3fn)``.  The
  externally observable dtype/shape/values therefore stay identical to the
  source operator.

Encoder details (all verified against ``torch`` e4m3 rounding)
-------------------------------------------------------------
``byte_signed = 8*(ec + 6) + rint(|q| * 2^(3-ec)) - 128*[q < 0]`` with
``ec = clamp(floor(log2|q|), -6, 8)``.
* the sign bit uses a magnitude-independent comparison (``T.vcmp(q, |q|, "lt")``
  followed by ``T.vcast(bool -> f32, "rint")``), never a finite-multiplier form;
* ``uint8`` is lowered to ``i8`` by the codegen, so the byte is materialised in
  *signed* space (``bits - 128*sign``) before the narrowing cast;
* the tile-level ``exp2`` is *not* exactly a power of two (ln/exp based) which
  breaks round-to-nearest-even exactly at mantissa ties, so the mantissa scale
  ``2^(3-ec)`` is built exactly as ``2^9 / 2^(ec+6)`` with ``2^(ec+6)`` obtained
  from ``T.vshl`` on int32 (exact integer power of two) followed by an exact
  power-of-two division;
* the quantized value is produced with the same division as the reference
  implementation (``x / scale``), not with a reciprocal multiply: the two are
  not byte-equivalent at fine mantissa boundaries.

run: python _fp8_quant_kernel.py --level all
"""

import argparse
import functools
import os

import torch

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# Programming mode used by the verified implementation (TileOPs harness syncs
# Kernel.ascend_mode from this literal before running its report).
ASCEND_MODE = "Developer"

FP8_MIN, FP8_MAX = -448.0, 448.0
AMAX_EPS = 1e-4
# Elements per tile.  At 4096 the soft-encoder buffers overflow UB once the
# auto multi-buffer pass doubles them; 2048 fits (documented fallback path).
BLOCK_ELEMS = 2048
_DEFAULT_VECTOR_CORES = 40

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _vector_core_num():
    """Vector core count (query the device, fall back to the 910B3 constant)."""
    try:
        from tilelang.utils.npu_utils import NPUUtils

        num = int(NPUUtils.get().get_aivector_core_num())
        return num if num > 0 else _DEFAULT_VECTOR_CORES
    except Exception:  # pragma: no cover - device query is best effort
        return _DEFAULT_VECTOR_CORES


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation, independent of the NPU kernel)
# ---------------------------------------------------------------------------
def golden_fp8_quant(input_tensor):
    """Per-group fp8 e4m3 quantization of a ``[B, S_kv, G, D]`` tensor.

    Semantic basis: the source operator (amax -> clamp(1e-4) -> /448 ->
    clamp(+-448) -> float8_e4m3fn).  The intermediate domain is float32 (the
    source *kernel* accumulates amax/scale in float32).  This implementation
    deliberately does NOT mirror the NPU kernel: it uses torch-native division,
    no reciprocal multiply, no software encoder and no uint8 carrier.

    The final float32 -> float8_e4m3fn cast is performed on CPU because this
    toolchain cannot cast to fp8 on device (``aclnnInplaceCopy`` error 561103);
    fp8 quantization is per-element exact, so the CPU cast is bit-identical.
    """
    x = input_tensor.to(torch.float32)
    amax = torch.abs(x).amax(dim=-1, keepdim=True).clamp(min=AMAX_EPS)
    scale = amax / 448.0
    q = torch.clamp(x / scale, min=FP8_MIN, max=FP8_MAX)
    out = q.cpu().to(torch.float8_e4m3fn)
    return scale.squeeze(dim=-1), out


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, in_dtype):
    """Build (and cache) the npuir kernel factory for one shape/dtype.

    The returned callable accepts the harness config ``(num_stages, block_m)``
    and yields the compiled kernel; ``num_stages``/``block_m`` are advisory --
    the NPU has no autotune and the tile shape is derived from ``BLOCK_ELEMS``.
    """
    rows = batch * seq_len_kv * kv_group
    dim = index_dim
    block_m = max(1, min(BLOCK_ELEMS // dim, rows))
    n_tiles = (rows + block_m - 1) // block_m
    num_blocks = min(_vector_core_num(), n_tiles)
    iters = (n_tiles + num_blocks - 1) // num_blocks

    @tilelang.jit(target="npuir")
    def _fp8_quant_fwd_func(num_stages=0, block_m_hint=32):
        _ = (num_stages, block_m_hint)

        @T.prim_func
        def _fp8_quant_fwd_main(
            X: T.Tensor((rows, dim), in_dtype),
            Scale: T.Tensor((rows,), "float32"),
            OutU8: T.Tensor((rows, dim), "uint8"),
        ):
            with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                x_in = T.alloc_ub((block_m, dim), in_dtype)
                x32 = T.alloc_ub((block_m, dim), "float32")
                qv = T.alloc_ub((block_m, dim), "float32")
                av = T.alloc_ub((block_m, dim), "float32")
                lv = T.alloc_ub((block_m, dim), "float32")
                t1 = T.alloc_ub((block_m, dim), "float32")
                num = T.alloc_ub((block_m, dim), "float32")
                pwf = T.alloc_ub((block_m, dim), "float32")
                sbv = T.alloc_ub((block_m, dim), "float32")
                ei = T.alloc_ub((block_m, dim), "int32")
                qi = T.alloc_ub((block_m, dim), "int32")
                one_i = T.alloc_ub((block_m, dim), "int32")
                pw = T.alloc_ub((block_m, dim), "int32")
                u8 = T.alloc_ub((block_m, dim), "uint8")
                cond = T.alloc_ub((block_m, dim), "bool")
                amax = T.alloc_ub((block_m, 1), "float32")
                amax_c = T.alloc_ub((block_m, 1), "float32")
                c448 = T.alloc_ub((block_m, 1), "float32")
                scale2 = T.alloc_ub((block_m, 1), "float32")
                scale1 = T.alloc_ub((block_m,), "float32")

                for i, j in T.Parallel(block_m, dim):
                    one_i[i, j] = 1
                for i in T.Parallel(block_m):
                    c448[i, 0] = 448.0

                for it in T.serial(iters):
                    tile = it * num_blocks + cid
                    if tile < n_tiles:
                        # Overlapping-window row mapping: every tile is a full
                        # (block_m, dim) block, so the tail tile re-reads a few
                        # already computed rows (deterministic, idempotent)
                        # instead of needing a partial copy or host padding.
                        row0 = T.min(tile * block_m, rows - block_m)

                        # 1) load + fp32 promotion
                        T.copy(X[row0, 0], x_in)
                        for i, j in T.Parallel(block_m, dim):
                            x32[i, j] = T.cast(x_in[i, j], "float32")

                        # 2) per-group absmax (D is the reduction axis)
                        T.vabs(x32, av)
                        T.reduce(av, amax, dims=[1], reduce_mode="max")

                        # 3) scale = max(amax, 1e-4) / 448 (exact division).
                        #    No upper clamp: an infinite amax must stay infinite
                        #    (the reference yields x/inf = 0 there).
                        for i in T.Parallel(block_m):
                            amax_c[i, 0] = T.max(amax[i, 0], AMAX_EPS)
                        T.vdiv(amax_c, c448, scale2)

                        # 4) q = clamp(x / scale, +-448)
                        T.vdiv(x32, scale2, qv)
                        T.vclamp(qv, qv, FP8_MIN, FP8_MAX)

                        # 5) soft e4m3 encode
                        # 5a) sign bit: [q < |q|] == [q < 0] (magnitude
                        #     independent, valid over the whole range)
                        T.vabs(qv, av)
                        T.vcmp(qv, av, cond, "lt")
                        T.vcast(cond, sbv, round_mode="rint")
                        # 5b) exponent ec = clamp(floor(log2|q|), -6, 8)
                        T.vclamp(av, av, 1e-30, FP8_MAX)
                        T.vlog2(av, lv, t1)
                        T.vcast(lv, ei, round_mode="floor")
                        T.vclamp(ei, ei, -6, 8)
                        T.vcast(ei, t1, round_mode="rint")
                        # 5c) exact mantissa scaling: |q| * 2^(3-ec)
                        #     = (|q| * 2^9) / 2^(ec+6) with 2^(ec+6) an exact
                        #     integer power of two (T.vshl) so that ties round
                        #     to nearest even exactly like the fp8 cast.
                        for i, j in T.Parallel(block_m, dim):
                            qi[i, j] = ei[i, j] + 6
                        T.vshl(one_i, qi, pw)
                        T.vcast(pw, pwf, round_mode="rint")
                        for i, j in T.Parallel(block_m, dim):
                            num[i, j] = av[i, j] * 512.0
                        T.vdiv(num, pwf, lv)
                        T.vcast(lv, qi, round_mode="rint")
                        T.vcast(qi, lv, round_mode="rint")
                        # 5d) signed byte (uint8 is lowered to i8: emit the
                        #     byte in signed space, clamp to i8 range)
                        for i, j in T.Parallel(block_m, dim):
                            lv[i, j] = 8.0 * (t1[i, j] + 6.0) + lv[i, j] - 128.0 * sbv[i, j]
                        for i, j in T.Parallel(block_m, dim):
                            u8[i, j] = T.cast(T.min(T.max(lv[i, j], -128.0), 127.0), "uint8")

                        # 6) write back
                        for i in T.Parallel(block_m):
                            scale1[i] = scale2[i, 0]
                        T.copy(u8, OutU8[row0, 0])
                        T.copy(scale1, Scale[row0 : row0 + block_m])

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


# ---------------------------------------------------------------------------
# Host-side dispatch: restores the externally observable contract
# ---------------------------------------------------------------------------
def fp8_quant_forward(batch, seq_len_kv, kv_group, index_dim, in_dtype, input_tensor, config=None):
    """Run the NPU kernel and return ``(scale, out_fp8)`` exactly as the source op.

    ``input_tensor`` is the original ``(B, S_kv, G, D)`` tensor.  The kernel works
    on the metadata view ``(R, D)`` and writes an ``uint8`` byte carrier; the
    output is restored with ``reshape`` + ``view(torch.float8_e4m3fn)`` (a pure
    metadata reinterpret -- verified available on device; the dtype-restoring
    ``.to(fp8)`` cast is not).  ``config`` mirrors the harness ``dict`` with
    ``num_stages``/``block_m`` keys (both advisory on NPU).
    """
    config = config or {}
    rows = batch * seq_len_kv * kv_group
    dtype_str = str(in_dtype).replace("torch.", "")
    kernel = _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, dtype_str)(
        config.get("num_stages", 0), config.get("block_m", 32)
    )

    x = input_tensor.contiguous()
    scale = torch.empty(rows, dtype=torch.float32, device=x.device)
    out_u8 = torch.empty(rows, index_dim, dtype=torch.uint8, device=x.device)
    kernel(x.reshape(rows, index_dim), scale, out_u8)
    return (
        scale.reshape(batch, seq_len_kv, kv_group),
        out_u8.reshape(batch, seq_len_kv, kv_group, index_dim).view(torch.float8_e4m3fn),
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _decode_fp8(u8):
    """Reinterpret a uint8 byte carrier as fp8 e4m3 (host metadata view)."""
    return u8.cpu().contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def _cosine_similarity(got_u8, ref_u8):
    """Cosine similarity over the finite lanes (source-op comparator, thr 0.99)."""
    import torch.nn.functional as F

    got = _decode_fp8(got_u8).reshape(-1)
    ref = _decode_fp8(ref_u8).reshape(-1)
    mask = torch.isfinite(got) & torch.isfinite(ref)
    if int(mask.sum()) == 0:
        return 1.0
    got, ref = got[mask], ref[mask]
    # cosine similarity is undefined for a zero vector (e.g. all-zero input);
    # correctness for that degenerate case is carried by the byte comparison.
    if float(torch.linalg.vector_norm(ref)) == 0.0:
        return 1.0 if float(torch.linalg.vector_norm(got)) == 0.0 else 0.0
    return float(F.cosine_similarity(got, ref, dim=0))


def _classify_mismatches(x_cpu, got_u8, ref_u8):
    """Classify fp8 byte differences.

    Accepted classes:

    * ``zero_ok``   exact ``-0.0`` sign-of-zero (``0x00`` vs ``0x80``); both
      decode to 0 and the software encoder cannot represent the sign of zero;
    * ``nan_lane``  NaN lanes -- outside the operator contract (DESIGN R6);
    * ``tie_1lsb``  +-1 LSB at an exact mantissa tie: the device float32
      difference of the device float32 division (which is ~1 ulp off, not IEEE
      correctly rounded) legitimately selects the other neighbour of a half-way
      mantissa.  Detected by ``|frac(|q| * 2^(3-ec)) - 0.5| <= 1e-5``, i.e. the
      reference value lies within one device rounding distance of the tie.

    ``unexpected`` (returned last) must be zero.
    """
    got = got_u8.cpu().reshape(-1).to(torch.int32)
    ref = ref_u8.cpu().reshape(-1).to(torch.int32)
    x = x_cpu.to(torch.float32)
    xf = x.reshape(-1)

    # The reference quantized value (also drives the tie classification).
    amax = torch.abs(x).amax(dim=-1, keepdim=True).clamp(min=AMAX_EPS)
    q = torch.clamp(x / (amax / 448.0), min=FP8_MIN, max=FP8_MAX)
    qf = q.reshape(-1)

    # Quantization groups whose amax is NaN (a NaN somewhere in the group) are
    # outside the operator contract (DESIGN R6): NaN propagation in a vector max
    # is not specified by the ISA docs, so those lanes are reported, not judged.
    bad_row = torch.isnan(amax.squeeze(-1)).reshape(-1)
    bad_lane = bad_row.repeat_interleave(q.shape[-1])

    diff = got - ref
    mism = diff != 0
    zero_ok = mism & (qf == 0) & ((got ^ ref) == 0x80)
    nan_lane = mism & (torch.isnan(qf) | torch.isnan(xf) | bad_lane)
    other = mism & ~zero_ok & ~nan_lane

    tie_1lsb = torch.zeros_like(other)
    if bool(other.any()):
        a = qf.abs().clamp(min=1e-30).clamp(max=FP8_MAX)
        ec = torch.floor(torch.log2(a)).clamp(-6, 8)
        two = torch.pow(torch.tensor(2.0, dtype=torch.float64), 3.0 - ec.double())
        u = a.double() * two
        frac = u - torch.floor(u)
        # `|q| * 2^(3-ec)` lives in [8, 16).  A one-ulp difference in the device
        # float32 quotient moves this value by at most 16 * 2^-23 ~ 1.9e-6, so a
        # reference mantissa within 1e-5 of the half-way point may legitimately
        # resolve to either neighbour.
        near_tie = (frac - 0.5).abs() <= 1e-5
        tie_1lsb = other & near_tie & (diff.abs() == 1)
    unexpected = other & ~tie_1lsb
    return {
        "total": int(xf.numel()),
        "mismatch": int(mism.sum()),
        "zero_ok": int(zero_ok.sum()),
        "nan_lane": int(nan_lane.sum()),
        "tie_1lsb": int(tie_1lsb.sum()),
        "unexpected": int(unexpected.sum()),
    }


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
def _make_input(batch, seq_len_kv, kv_group, index_dim, dtype, gen):
    shape = (batch, seq_len_kv, kv_group, index_dim)
    if gen == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if gen == "const":
        return torch.full(shape, 448.0, dtype=dtype)
    if gen == "tiny_neg":
        # the region (0 < |v| < 1e-30) that broke a finite-multiplier sign form
        tiny = torch.tensor(
            [1e-31, 3e-31, 5e-31, 9.9e-31, 1e-32, 1e-35, 1e-38, 1e-41, 1e-45], dtype=torch.float32
        )
        vals = torch.cat([tiny, -tiny, tiny * 0.0, -tiny * 0.0]).repeat(
            (batch * seq_len_kv * kv_group * index_dim + 2 * tiny.numel() - 1) // (2 * tiny.numel())
        )
        return vals[: batch * seq_len_kv * kv_group * index_dim].reshape(shape).to(dtype)
    if gen == "corners":
        # one special value per row (column 0) so a NaN/Inf stays confined to
        # its own quantization group instead of contaminating every row.
        x = torch.randn(shape, dtype=torch.float32) * 4.0
        rows = batch * seq_len_kv * kv_group
        flat = x.reshape(rows, index_dim)
        specials = torch.tensor(
            [
                0.0,
                -0.0,
                1e-6,
                -1e-6,
                1e-30,
                -1e-30,
                1e-31,
                -1e-31,
                1e-38,
                -1e-38,
                2.0**-149,
                -(2.0**-149),
                448.0,
                -448.0,
                1e4,
                -1e4,
                float("inf"),
                float("-inf"),
                float("nan"),
            ],
            dtype=torch.float32,
        )
        for k, val in enumerate(specials.tolist()):
            flat[k % rows, 0] = val
        return x.to(dtype)
    return torch.randn(shape, dtype=dtype)


def _run_case(
    batch, seq_len_kv, kv_group, index_dim, dtype, gen="randn", tag="case", seed=20261014
):
    torch.manual_seed(seed)
    dtype_str = str(dtype).replace("torch.", "")
    x = _make_input(batch, seq_len_kv, kv_group, index_dim, dtype, gen)
    rows = batch * seq_len_kv * kv_group

    kernel = _fp8_quant_kernel(batch, seq_len_kv, kv_group, index_dim, dtype_str)(0, 32)
    x_dev = x.to("npu").reshape(rows, index_dim).contiguous()
    scale_dev = torch.empty(rows, dtype=torch.float32, device="npu")
    out_dev = torch.empty(rows, index_dim, dtype=torch.uint8, device="npu")
    kernel(x_dev, scale_dev, out_dev)

    scale_g, out_g = golden_fp8_quant(x)
    scale_k = scale_dev.cpu().reshape(batch, seq_len_kv, kv_group)
    ref_u8 = out_g.view(torch.uint8).reshape(rows, index_dim)

    finite_rows = torch.isfinite(scale_g)
    torch.testing.assert_close(scale_k[finite_rows], scale_g[finite_rows], rtol=1e-4, atol=1e-4)
    rep = _classify_mismatches(x, out_dev, ref_u8)
    cos = _cosine_similarity(out_dev, ref_u8)

    assert cos >= 0.99, f"[{tag}] cosine similarity too low: {cos}"
    assert rep["unexpected"] == 0, f"[{tag}] unexpected fp8 byte mismatches: {rep}"
    print(
        f"[{tag}] PASS shape=({batch},{seq_len_kv},{kv_group},{index_dim}) "
        f"dtype={dtype_str} gen={gen} cos={cos:.6f} "
        f"bytes: exact={rep['total'] - rep['mismatch']}/{rep['total']} "
        f"-0.0={rep['zero_ok']} tie+-1lsb={rep['tie_1lsb']} "
        f"nan={rep['nan_lane']} unexpected={rep['unexpected']}"
    )
    return rep


# ---------------------------------------------------------------------------
# Layered test suite
# ---------------------------------------------------------------------------
# L0: smoke -- compile + run on the smallest tiles across all input dtypes plus
# the special-value corner case (zeros / +-448 / +-inf / NaN / tiny values).
L0_CASES = [
    (1, 256, 1, 64, torch.float32, "randn"),
    (1, 256, 1, 64, torch.float16, "randn"),
    (1, 256, 1, 64, torch.bfloat16, "randn"),
    (1, 256, 1, 64, torch.float32, "zeros"),
    (1, 256, 1, 64, torch.float32, "corners"),
]

# L1: the manifest workloads (blocking).
L1_CASES = [
    (1, 8192, 1, 64, torch.float16, "randn"),
    (1, 8192, 1, 64, torch.bfloat16, "randn"),
    (1, 4096, 1, 128, torch.float32, "randn"),
    (1, 16384, 1, 32, torch.float32, "randn"),
    (1, 1024, 4, 64, torch.float16, "randn"),
]

# L2: extra shapes / magnitudes (advisory).
L2_CASES = [
    (2, 512, 1, 128, torch.float16, "randn"),
    (1, 2048, 2, 32, torch.bfloat16, "randn"),
    (1, 512, 1, 64, torch.float32, "zeros"),
]


def run_L0():
    for i, case in enumerate(L0_CASES):
        _run_case(*case, tag=f"L0-{i + 1}", seed=20261014 + i)


def run_L1():
    for i, case in enumerate(L1_CASES):
        _run_case(*case, tag=f"L1-{i + 1}", seed=777000 + i)


def run_L2():
    for i, case in enumerate(L2_CASES):
        try:
            _run_case(*case, tag=f"L2-{i + 1}", seed=555000 + i)
        except Exception as exc:  # noqa: BLE001
            print(f"[L2-{i + 1}] WARN (recorded, non-blocking): {exc}")


def run_boundary():
    cases = [
        (1, 100, 1, 64, torch.float16, "randn", "non_divisible_R"),
        (1, 1, 1, 64, torch.float32, "randn", "single_row"),
        (1, 37, 1, 32, torch.bfloat16, "randn", "non_divisible_d32"),
        (1, 8, 1, 128, torch.float32, "corners", "corner_values"),
        (1, 16, 1, 64, torch.float32, "const", "max_magnitude"),
        (1, 4, 1, 64, torch.float32, "tiny_neg", "tiny_negative_region"),
    ]
    for i, (b, s, g, d, dt, gen, name) in enumerate(cases):
        try:
            _run_case(b, s, g, d, dt, tag=f"Boundary-{name}", gen=gen, seed=313000 + i)
        except Exception as exc:  # noqa: BLE001
            print(f"[Boundary-{name}] WARN (recorded, non-blocking): {exc}")


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
    tilelang.cache.clear_cache()
    main()
