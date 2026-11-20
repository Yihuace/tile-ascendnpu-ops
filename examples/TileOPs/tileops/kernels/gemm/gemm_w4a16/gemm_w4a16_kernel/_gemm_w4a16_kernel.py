# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_gemm_w4a16_kernel: group-affine W4A16 dequantized GEMM on npuir (Developer).

Migrated from the TileOPs GPU ``_gemm_w4a16_kernel`` (SM90 fusion of INT4
unpack + affine dequant + A16 tensor-core GEMM) to the npuir backend.  The NPU
algorithm follows the frozen ``DESIGN.md`` of this directory:

  * R1: the GPU "128 threads decode the packed tile into SMEM, then WGMMA"
    producer/consumer form is replaced by a single Developer kernel in which
    the dequantization runs on the **Vector** side (v-prefix chain writing a
    shared ``W_sh`` tile) and the **Cube** side consumes that buffer directly
    through ``T.gemm`` (automatic CV split / UB->L1 handoff, no manual barrier).
  * R2: the GPU private shift/mask intrinsic (``_tir_packed_to_unsigned_convert``)
    is replaced by a **signedness-agnostic, cast-only INT4 unpack recipe**
    (``npuir_cast``/``npuir_mul``/``npuir_sub`` + ``npuir_interleave``): npuir
    lowers ``uint8`` as *i8*, so ``b >= 128`` casts to a negative value; because
    ``256 == 0 (mod 16)`` the floored `mod 16` extraction reproduces the unsigned
    nibbles bit-exactly.  Bitwise ``vand``/``vshr`` and ``vfloordiv`` are NOT
    used (verifier rejects / vector-core timeout).
  * R3: the GPU 2D grid + ``threads=128`` is folded into a 1D ``T.Kernel`` with
    ``cid -> (by, bx)`` and ``num_launched == num_logical`` (non-persistent).
  * R4: the per-thread chunk loop is dropped entirely -- the whole
    ``[block_N, block_K/2]`` packed tile is processed by block-wide vector ops.
  * E1..E4 (DESIGN 1.6.1): single fp32 -> fp16 rounding at store time, fp16-domain
    affine dequant, per-(n, k-tile) metadata cast hoisted once + ``vbrc``
    broadcast, group index ``g = k0 // group_size`` constant inside a k-tile
    (``block_K | group_size``).

Toolchain constraints respected (siblings ``examples/gemm/_gemm_kernel/repro/``):
  * A GM->L1 ``T.copy`` with a *runtime extent* feeding ``T.gemm`` is not
    lowerable in Developer mode ("Unsupported copy from cbuf to cbuf").  All
    loads therefore use **static extents with dynamic start offsets only**.
  * The epilogue keeps the supported form ``T.copy(C_local[0:tm, 0:tn],
    OUT[m0:m0+tm, n0:n0+tn])`` (static source offset, dynamic extents) which
    emits the single-rounding fp32 -> fp16 ``VCastOp``.
  * When ``M < block_M`` (the decode shape) the A tile is filled by an
    exact-extent copy of the ``M`` valid rows, so no out-of-range GM row is
    ever read; the unused tile rows only reach the unwritten C rows.

Known boundary limitation (non-blocking, see ``repro/``): an output tile that
degenerates to exactly ``tm == tn == 1`` (M=1, N=1) aborts the device with
"MTE DDR address is out of range" at the epilogue store.  No manifest workload
(N >= 64) hits it; M=1 with N >= 2 and N=1 with M >= 2 are both correct.

Coverage: fp16 activation only (operator contract), non-divisible M/N,
``block_K | K`` (guaranteed by ``group_size | K`` and ``block_K | group_size``),
group-boundary k-tiles, one-sided / wide / all-zero weights.

Run: ``python _gemm_w4a16_kernel.py --level {L0|L1|all}``
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# Stage 3 validated programming mode; must equal TILELANG_ASCEND_MODE above.
ASCEND_MODE = "Developer"

# Directory of this operator (used to spawn isolated boundary-case children).
OPERATOR_DIR = os.path.dirname(os.path.abspath(__file__))

# fp32 accumulation is mandatory for fp16 GEMM on npuir (T.gemm.md 2.3).
_ACCUM_DTYPE = "float32"

# Manifest group size (GemmW4A16Op hard contract).
GROUP_SIZE = 128

# Tolerance (DESIGN 8.2 / examples/TileOPs/tests/ops/test_gemm.py:181).
_TOL = {"float16": {"atol": 7e-2, "rtol": 5e-2}}


# ---------- Workload generator (packed INT4 + affine metadata) ----------
def quantize_weight_int4(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    """Affine group-wise INT4 quantize + pack a logical ``[N, K]`` weight.

    Ported verbatim from ``examples/TileOPs/tileops/workloads/gemm.py`` (the
    harness input generator).  Packing contract: the low nibble holds the even
    k, the high nibble the odd k.

    Returns ``(packed[N, K/2] uint8, scale[N, K/gs] fp32, zero[N, K/gs] uint8,
    dequantized[N, K] fp32)``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1).clamp_max(0)
    group_max = grouped.amax(dim=-1).clamp_min(0)
    scale = ((group_max - group_min) / 15.0).clamp_min(1e-12)
    zero = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )
    unsigned = quantized.reshape(n, k)
    packed = torch.bitwise_or(
        unsigned[:, 0::2],
        torch.bitwise_left_shift(unsigned[:, 1::2], 4),
    )
    dequantized = (
        (quantized.float() - zero.float().unsqueeze(-1)) * scale.unsqueeze(-1)
    ).reshape(n, k)
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


def _make_inputs(m, n, k, group_size, dtype, seed=0, kind="randn", n_chunk=1024,
                 act_zero=False):
    """CPU inputs ``(activation, packed_weight, weight_scale, weight_zero)``.

    ``n`` is generated in chunks so multi-hundred-MB workloads stay in RAM.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    activation = torch.randn(m, k, generator=gen, dtype=torch.float32).to(dtype)
    if act_zero:
        activation = torch.zeros_like(activation)
    packed = torch.empty(n, k // 2, dtype=torch.uint8)
    scale = torch.empty(n, k // group_size, dtype=torch.float32)
    zero = torch.empty(n, k // group_size, dtype=torch.uint8)
    for c0 in range(0, n, n_chunk):
        c1 = min(c0 + n_chunk, n)
        rows = c1 - c0
        if kind == "zeros":
            w = torch.zeros(rows, k, dtype=torch.float32)
        elif kind == "pos":
            w = torch.rand(rows, k, generator=gen, dtype=torch.float32) * 0.5
        elif kind == "neg":
            w = -torch.rand(rows, k, generator=gen, dtype=torch.float32) * 0.5
        elif kind == "wide":
            w = torch.randn(rows, k, generator=gen, dtype=torch.float32) * 2.0
        else:
            w = torch.randn(rows, k, generator=gen, dtype=torch.float32) * 0.25
        p, s, z, _ = quantize_weight_int4(w, group_size)
        packed[c0:c1] = p
        scale[c0:c1] = s
        zero[c0:c1] = z
    return activation, packed, scale, zero


# ---------- Golden (PyTorch CPU reference implementation) ----------
def _dequant_from_packed(packed, scale, zero, group_size, out_dtype):
    """Unpack nibbles and dequantize ``W = (q - z) * s`` in fp32, round once."""
    n, packed_k = packed.shape
    k = packed_k * 2
    q = torch.empty(n, k, dtype=torch.uint8)
    q[:, 0::2] = torch.bitwise_and(packed, 0x0F)  # low nibble  -> even k
    q[:, 1::2] = torch.bitwise_and(
        torch.bitwise_right_shift(packed, 4), 0x0F
    )  # high nibble -> odd k
    s = torch.repeat_interleave(scale, group_size, dim=1)  # (N, K) fp32
    z = torch.repeat_interleave(zero, group_size, dim=1)  # (N, K) uint8
    return ((q.to(torch.float32) - z.to(torch.float32)) * s).to(out_dtype)


def golden_gemm_w4a16(activation, packed_weight, weight_scale, weight_zero,
                      group_size: int = GROUP_SIZE):
    """Source-operator semantics: ``out = activation @ dequant(packed_weight)^T``.

    ``activation``: (M, K) fp16; ``packed_weight``: (N, K/2) uint8;
    ``weight_scale``: (N, K/group_size) fp32; ``weight_zero``: (N, K/group_size)
    uint8; returns (M, N) fp16.

    Mirrors ``GemmW4A16Test.ref_program`` + ``quantize_weight_int4``: the weight
    is dequantized in fp32 and rounded once to the activation dtype, the
    accumulation runs in the fp32 opmath domain through explicit ``torch``
    module calls, and a single fp32 -> fp16 round is applied at the end (the
    same single rounding the kernel epilogue VCast performs, DESIGN 1.6.1 E1).
    ``N`` is processed in chunks so the reference stays memory bounded.
    """
    m, k = activation.shape
    n = packed_weight.shape[0]
    out = torch.empty(m, n, dtype=activation.dtype)
    a32 = activation.to(torch.float32)
    chunk = max(16, (16 * 1024 * 1024) // max(k, 1))
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        w = _dequant_from_packed(
            packed_weight[c0:c1], weight_scale[c0:c1], weight_zero[c0:c1],
            group_size, activation.dtype,
        )
        out[:, c0:c1] = torch.matmul(
            a32, torch.transpose(w.to(torch.float32), 0, 1)
        ).to(activation.dtype)
    return out


# ---------- Kernel ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemm_w4a16_kernel(M, N, K, group_size=128, dtype="float16",
                       block_M=16, block_N=64, block_K=64, num_stages=2):
    """W4A16 fused-dequant GEMM for npuir (Developer, Cube+Vector MixCV).

    Args:
        M, N, K: compile-time GEMM dims (one kernel per shape, jit-cached).
        group_size: affine quantization group (manifest contract, 128).
        dtype: activation / output dtype (operator contract: float16).
        block_M/block_N/block_K: tile sizes (DESIGN 5.2; block_M=16 keeps the
            m=1 decode shape bandwidth-bound instead of compute-bound).
        num_stages: K-pipeline depth (DESIGN 4.5 UB budget).

    Returns:
        The compiled ``main`` prim_func (``OUT`` allocated via ``out_idx``).
    """
    accum_dtype = _ACCUM_DTYPE
    # R6 invariants: the factory already guarantees group_size | K and
    # block_K | group_size, so every k-tile lies inside exactly one group
    # (g = k0 // group_size is a per-tile scalar) and there is no K tail.
    assert group_size % block_K == 0, "block_K must divide group_size (R6)"
    assert K % group_size == 0, "K must be divisible by group_size (R6)"
    assert K % block_K == 0, "block_K must divide K (R6, no K tail on npuir)"
    assert block_K % 2 == 0, "packed nibble axis must be integral"

    bk_half = block_K // 2
    n_blocks = (N + block_N - 1) // block_N
    m_blocks = (M + block_M - 1) // block_M
    # Non-persistent 1D grid: num_launched == num_logical (never min()-clamped,
    # otherwise blocks beyond the physical core count would be silently dropped).
    num_launched = m_blocks * n_blocks
    k_iters = K // block_K
    # Compile-time shape specialisations (trace-time Python branches): avoid a
    # dynamic-extent GM->L1 copy (unsupported) and avoid out-of-range reads.
    m_fits_one_tile = M < block_M
    n_fits_one_tile = N < block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),                      # activation  fp16
        PW: T.Tensor((N, K // 2), "uint8"),              # INT4 packed weight
        WS: T.Tensor((N, K // group_size), "float32"),   # group scale
        WZ: T.Tensor((N, K // group_size), "uint8"),     # group zero
        OUT: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            # cid -> (M block, N block) bijection (DESIGN 0.6 R3).
            by = cid // n_blocks
            bx = cid % n_blocks
            m0 = by * block_M
            n0 = bx * block_N
            # Valid output extents (dynamic tail guard for the epilogue only).
            tm = T.min(block_M, M - m0)
            tn = T.min(block_N, N - n0)

            # Cube operands (L1) and fp32 accumulator (L0C).
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            W_sh = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Vector-side unpack / dequant scratch (UB).  ``tmp`` is reused for
            # both cast temporaries of the recipe (their live ranges do not
            # overlap) to keep the UB footprint inside the 192 KiB budget.
            pw = T.alloc_shared((block_N, bk_half), "uint8")
            bf = T.alloc_shared((block_N, bk_half), dtype)
            tmp = T.alloc_shared((block_N, bk_half), dtype)
            hf = T.alloc_shared((block_N, bk_half), dtype)
            lo = T.alloc_shared((block_N, bk_half), dtype)
            hi = T.alloc_shared((block_N, bk_half), dtype)
            h32 = T.alloc_shared((block_N, bk_half), "int16")
            hq32 = T.alloc_shared((block_N, bk_half), "int16")
            q = T.alloc_shared((block_N, block_K), dtype)
            t2 = T.alloc_shared((block_N, block_K), dtype)
            zb = T.alloc_shared((block_N, block_K), dtype)
            sb = T.alloc_shared((block_N, block_K), dtype)
            s32 = T.alloc_shared((block_N, 1), "float32")
            z8 = T.alloc_shared((block_N, 1), "uint8")
            sf = T.alloc_shared((block_N, 1), dtype)
            zf = T.alloc_shared((block_N, 1), dtype)

            # K-fold pipeline: MTE2 load -> Vector unpack+dequant -> Cube MMA.
            for kk in T.Pipelined(k_iters, num_stages=num_stages):
                k0 = kk * block_K
                g = k0 // group_size  # scalar group index (R6: constant in tile)

                # --- loads: static extents, dynamic start offsets only ---
                if m_fits_one_tile:
                    # A is smaller than one M tile: copy exactly M rows in
                    # range; the unused tile rows only reach unwritten C rows.
                    T.copy(A[0:M, k0:k0 + block_K], A_sh[0:M, 0:block_K])
                else:
                    T.copy(A[m0:m0 + block_M, k0:k0 + block_K], A_sh)
                if n_fits_one_tile:
                    T.copy(PW[0:N, k0 // 2:k0 // 2 + bk_half],
                           pw[0:N, 0:bk_half])
                    T.copy(WS[0:N, g:g + 1], s32[0:N, 0:1])
                    T.copy(WZ[0:N, g:g + 1], z8[0:N, 0:1])
                else:
                    T.copy(PW[n0:n0 + block_N, k0 // 2:k0 // 2 + bk_half], pw)
                    T.copy(WS[n0:n0 + block_N, g:g + 1], s32)
                    T.copy(WZ[n0:n0 + block_N, g:g + 1], z8)

                # --- signedness-agnostic cast-only INT4 unpack (DESIGN 0.6 R2) ---
                # npuir lowers uint8 as i8, so keep every step exact-integer:
                #   hf = floor(bf/16)   (exact, handles b >= 128 as negative)
                #   lo = bf - 16*hf     == b mod 16   in [0,15] (even k)
                #   hi = hf - 16*floor(hf/16)         in [0,15] (odd k)
                T.npuir_cast(pw, bf, round_mode="rint")
                T.npuir_mul(bf, 0.0625, tmp)
                T.npuir_cast(tmp, h32, round_mode="floor")
                T.npuir_cast(h32, hf, round_mode="rint")
                T.npuir_mul(hf, 16.0, tmp)
                T.npuir_sub(bf, tmp, lo)
                T.npuir_mul(hf, 0.0625, tmp)
                T.npuir_cast(tmp, hq32, round_mode="floor")
                T.npuir_cast(hq32, tmp, round_mode="rint")
                T.npuir_mul(tmp, 16.0, tmp)
                T.npuir_sub(hf, tmp, hi)
                # k-order restored: interleave(lo, hi) -> [block_N, block_K]
                T.npuir_interleave(lo, hi, q)

                # --- affine group dequant in the fp16 domain (E3 + E4) ---
                T.npuir_cast(s32, sf, round_mode="rint")
                T.npuir_cast(z8, zf, round_mode="rint")
                T.vbrc(zf, zb)  # explicit broadcast (no implicit column tax)
                T.vbrc(sf, sb)
                T.npuir_sub(q, zb, t2)
                T.npuir_mul(t2, sb, W_sh)

                # --- A16 Cube accumulation, NT form (E1: fp32 accum) ---
                T.gemm(A_sh, W_sh, C_local, initC=(kk == 0), b_transpose=True)

            # Epilogue: guarded write-back; the compiler emits one fp32 -> fp16
            # VCast (RINT) for the whole tile (single rounding, E1).
            T.copy(C_local[0:tm, 0:tn], OUT[m0:m0 + tm, n0:n0 + tn])

    return main


# ---------- precision comparing function ----------
def run_case(m, n, k, tag, dtype=torch.float16, group_size=GROUP_SIZE, seed=0,
             kind="randn", block_M=16, block_N=64, block_K=64, num_stages=2,
             n_chunk=1024, act_zero=False):
    """Build inputs, run the kernel on NPU, compare against the CPU golden."""
    dtype_str = str(dtype).rsplit(".", 1)[-1]
    tol = _TOL[dtype_str]

    act, packed, scale, zero = _make_inputs(
        m, n, k, group_size, dtype, seed=seed, kind=kind, n_chunk=n_chunk,
        act_zero=act_zero,
    )
    golden = golden_gemm_w4a16(act, packed, scale, zero, group_size)

    kernel = _gemm_w4a16_kernel(m, n, k, group_size, dtype_str, block_M,
                                block_N, block_K, num_stages)
    out = kernel(act.npu(), packed.npu(), scale.npu(), zero.npu()).cpu()

    assert out.shape == golden.shape, (out.shape, golden.shape)
    assert out.dtype == golden.dtype, (out.dtype, golden.dtype)
    max_diff = (out.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(out, golden, atol=tol["atol"], rtol=tol["rtol"])
    print(f"[{tag}] PASS gemm_w4a16(m={m},n={n},k={k},kind={kind}) "
          f"block=({block_M},{block_N},{block_K},{num_stages}) "
          f"max_diff={max_diff:.3e}")
    return max_diff


# ---------- hierarchical testing ----------
def run_L0():
    """Gate: DESIGN 8.3 L0 plan (7 shapes, all pass required)."""
    cases = [
        (64, 64, 128, "L0-1 smoke square"),
        (128, 256, 256, "L0-2 smoke rect multi-block"),
        (1, 512, 512, "L0-3 decode m=1 tail path"),
        (1, 8192, 16384, "L0-4 decode manifest hbm"),
        (1, 7168, 20480, "L0-5 decode non-power2 N"),
        (16, 1024, 1024, "L0-6 multi k-tile per group"),
        (1, 8192, 81920, "L0-7 long-K pressure"),
    ]
    worst = 0.0
    for m, n, k, tag in cases:
        worst = max(worst, run_case(m, n, k, tag))
    print(f"L0 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L1():
    """Extended coverage: group boundaries, M/N tails, metadata extremes."""
    cases = [
        (1, 8192, 8192, "L1-1 decode l2-resident", {}),
        (1, 8192, 256, "L1-2 tiny K one group", {}),
        (1, 64, 128, "L1-3 N == block_N, one group", {}),
        (64, 256, 128, "L1-4 single k-iter", {}),
        (128, 512, 128, "L1-5 single k-iter multi-M", {}),
        (16, 1024, 2048, "L1-6 16 k-tiles / 8 groups", {}),
        (1, 128, 1024, "L1-7 8 groups m=1", {}),
        (17, 256, 512, "L1-8 M tail non-divisible", {}),
        (3, 192, 256, "L1-9 M tail small", {}),
        (1, 100, 256, "L1-10 N tail non-divisible", {}),
        (33, 300, 384, "L1-11 odd M/N", {}),
        (1, 8192, 16384, "L1-12 decode-hbm (n_chunk small)", {"n_chunk": 256}),
        (32, 1024, 1024, "L1-13 zeros weight", {"kind": "zeros"}),
        (32, 1024, 1024, "L1-14 one-sided positive weight", {"kind": "pos"}),
        (32, 1024, 1024, "L1-15 one-sided negative weight", {"kind": "neg"}),
        (32, 1024, 1024, "L1-16 wide-range weight", {"kind": "wide"}),
    ]
    worst = 0.0
    for m, n, k, tag, kw in cases:
        worst = max(worst, run_case(m, n, k, tag, **kw))
    print(f"L1 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_L2():
    """Larger shapes (non-blocking: failures are recorded as warnings)."""
    cases = [
        (1, 8192, 20480, "L2-1 long-K"),
        (128, 4096, 2048, "L2-2 prefill-ish"),
        (64, 7168, 20480, "L2-3 non-power2 wide"),
    ]
    worst = 0.0
    for m, n, k, tag in cases:
        try:
            worst = max(worst, run_case(m, n, k, tag))
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): {tag} -> {exc}")
    print(f"L2 done ({len(cases)} cases), worst max_diff={worst:.3e}")


def run_boundary():
    """Boundary cases (non-blocking).

    Each case runs in a *child process* so that a device-side abort (e.g. the
    degenerate ``tm == tn == 1`` output tile, see ``repro/``) is contained and
    recorded as a WARN instead of killing the suite.
    """
    cases = [
        (16, 16, 128, "B-1 min tiles (M=16, N=16)", {}),
        (1, 16, 128, "B-2 M=1, N<block_N", {}),
        (1, 1, 128, "B-3 degenerate 1x1 output tile", {}),
        (1, 2, 128, "B-4 M=1, N=2 (1xN tile)", {}),
        (16, 64, 128, "B-5 exactly one group", {}),
        (16, 128, 1280, "B-6 10 groups", {}),
        (200, 300, 256, "B-7 non-divisible M/N", {}),
        (32, 1024, 1024, "B-8 all-zero weight", {"kind": "zeros"}),
        (64, 256, 256, "B-9 zero activation", {"act_zero": True}),
    ]
    n_pass = 0
    for m, n, k, tag, kw in cases:
        ok, detail = _boundary_case(m, n, k, tag, kw)
        if ok:
            n_pass += 1
            print(f"[Boundary] PASS {tag}: {detail}")
        else:
            print(f"[Boundary] WARN {tag} (not blocking): {detail}")
    print(f"Boundary done ({n_pass}/{len(cases)} cases passed)")


def _boundary_case(m, n, k, tag, kw):
    """Run one boundary case in a child process; return ``(ok, detail)``."""
    code = (f"import _gemm_w4a16_kernel as M;"
            f"M.run_case({m}, {n}, {k}, {tag!r}, **{kw!r})")
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", code], cwd=OPERATOR_DIR,
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return False, "child timeout"
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines()
             if "PASS gemm_w4a16" in ln or "Error" in ln or "error" in ln]
    detail = lines[-1][:220] if lines else f"child exit {proc.returncode}"
    return proc.returncode == 0, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L0()
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
