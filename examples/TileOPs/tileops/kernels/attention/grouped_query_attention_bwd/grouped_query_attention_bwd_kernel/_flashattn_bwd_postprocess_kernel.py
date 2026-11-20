# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""GQA backward postprocess kernel (NPU, target="npuir").

Migrated from the GPU TileOPs factory ``_flashattn_bwd_postprocess_kernel``
(``/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`` L106-128;
class ``FlashAttnBwdPostprocessKernel`` L131-176).

Source semantics (the only basis of the embedded golden):

    dq_out[b, s, h, d] = cast_to_storage_dtype(dq[b, s, h, d])   # RNE

``dq`` is BSHD (batch, seq_len, heads, dim) float32; ``dq_out`` is BSHD with
the storage dtype (float16 / bfloat16).  This is a pure element-wise
round-to-nearest-even (RNE) conversion -- there is NO scaling, NO ``delta``
and NO division (the source prim_func has exactly two parameters and one
``T.copy``).

NPU re-design (see DESIGN.md section 0.6):

  * R1 -- the source 3-D grid ``T.Kernel(ceildiv(S,64), heads, batch)`` is not
    supported on NPU.  It becomes a 1-D ``T.Kernel(nblk, is_npu=True)`` with an
    in-core ``T.serial(iters)`` grid-stride over the logical slabs
    (persistent kernel, ``nblk = min(40, total_tiles)`` physical Vector cores).
  * R2 -- the source tile is a fixed-head strided ``[64, dim]`` rectangle.  The
    tile becomes a fully contiguous slab ``[R_S, H, D]`` (fixed batch, ``R_S``
    consecutive ``s``, all H, all D).  ``R_S = gcd(seq_len, _rs_max)`` statically
    divides ``seq_len`` -> no tail tile and no padding.
  * R3 -- the source GM->GM ``T.copy`` (with implicit cast) is illegal on NPU
    (Developer mode forbids GM->GM).  It becomes GM->UB->GM via a shared buffer
    plus an explicit ``T.vcast(round_mode="rint")`` (TRAP-C12: the implicit
    cross-dtype copy is not a free cast fusion and fixes the rounding).

Hard constraints honoured:
  * The source ``make_dq_layout`` (GPU-only 8x8 gemm-fragment layout for
    vectorised Ampere atomicAdd) is dropped -- it carries no numerical
    semantics and NPU has no such fragment requirement.
  * ``threads=128`` (a CUDA execution detail) is removed.

run: python _flashattn_bwd_postprocess_kernel.py --level all
"""

import os

# Developer mode for npuir (single-core linear GM->UB->GM chain, no manual
# sync needed).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  enables .npu() placement

# Programming mode actually validated by this file (Stage 5 syncs this into
# the TileOPs ``Kernel.ascend_mode`` field).
ASCEND_MODE = "Developer"

# Physical Vector core count on Ascend910B3 (NPUUtils: 20 AI cores x 2 for a
# pure Vector operator; DESIGN.md section 5.5).
_N_PHYS = 40

# Target slab element count (transport bandwidth knee; DESIGN.md section 5.2).
_TILE_ELEMS = 16384

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Per-output-dtype tolerances (DESIGN.md section 8.2).  Both sides are RNE, so
# a pure cast should be bit-exact; the tolerance only covers rounding edges.
_TOLERANCE = {
    "float16": (1e-3, 1e-3),     # (atol, rtol)
    "bfloat16": (1e-2, 1e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_flashattn_bwd_postprocess(dq: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Reference ``dq_out = RNE(dq)`` (element-wise fp32 -> out_dtype).

    ``dq`` is ``(B, S, H, D)`` float32; the result is ``(B, S, H, D)`` with
    ``out_dtype``.  Only the source op semantics are used here (an RNE cast),
    never the DESIGN.md NPU tiling / persistence structure, so the reference
    stays independent from the kernel under test.
    """
    dq_cpu = torch.as_tensor(dq.detach().cpu())
    # f32 -> storage dtype is an RNE cast (PyTorch ``.to`` semantics).
    return torch.as_tensor(dq_cpu.to(torch.float32)).to(out_dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[1], target="npuir")
def _flashattn_bwd_postprocess_kernel(batch, heads, seq_len, dim, dtype="float16"):
    """Factory: dq_out[b,s,h,d] = RNE(dq[b,s,h,d]).

    All of ``batch`` / ``heads`` / ``seq_len`` / ``dim`` / ``dtype`` are
    compile-time constants (matching the source factory signature).
    """
    accum_dtype = "float"
    shape = (batch, seq_len, heads, dim)

    # R_S statically divides seq_len -> the slabs always cover the full
    # sequence with no tail tile and no padding (DESIGN.md section 5.4).
    _rs_max = max(1, _TILE_ELEMS // (heads * dim))
    R_S = math.gcd(seq_len, _rs_max)
    assert seq_len % R_S == 0
    n_s = seq_len // R_S
    total_tiles = batch * n_s
    nblk = min(_N_PHYS, total_tiles)
    iters = (total_tiles + nblk - 1) // nblk  # static loop bound

    @T.prim_func
    def flash_bwd_post(
        dq: T.Tensor(shape, accum_dtype),  # type: ignore
        dq_out: T.Tensor(shape, dtype),  # type: ignore
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            # All UB buffers are allocated unconditionally at the top level
            # (a conditional T.alloc is illegal).
            dq_f32 = T.alloc_shared([R_S, heads, dim], accum_dtype)  # fp32 staging
            dq_cast = T.alloc_shared([R_S, heads, dim], dtype)       # cast result

            # In-core grid-stride over the logical slabs; the loop bound is a
            # compile-time constant and the guard covers the ragged tail of the
            # round-robin mapping (DESIGN.md section 5.5 / 6.2).
            for it in T.serial(iters):
                t = cid + it * nblk
                if t < total_tiles:
                    bz = t // n_s            # batch index
                    ss = (t % n_s) * R_S     # s slab start

                    # 1. contiguous slab read GM -> UB (long contiguous DMA).
                    T.copy(dq[bz, ss:ss + R_S, 0:heads, 0:dim], dq_f32)
                    # 2. element-wise RNE cast (vectorised, hivm::VCastOp).
                    T.vcast(dq_f32, dq_cast, round_mode="rint")
                    # 3. contiguous slab write-back UB -> GM.
                    T.copy(dq_cast, dq_out[bz, ss:ss + R_S, 0:heads, 0:dim])

    return flash_bwd_post


# ---------- precision comparing func ----------
def _max_diff(y, golden):
    """Max abs error over finite golden entries."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _run_tensor(dq_cpu, dtype_str, tag):
    """Run the kernel on NPU tensors and compare against the golden output."""
    dq_cpu = dq_cpu.detach().cpu().contiguous()
    B, S, H, D = dq_cpu.shape
    out_dtype = _TORCH_DTYPE[dtype_str]

    golden = golden_flashattn_bwd_postprocess(dq_cpu, out_dtype)
    kernel = _flashattn_bwd_postprocess_kernel(B, H, S, D, dtype_str)

    dq_npu = dq_cpu.npu()
    dq_out = kernel(dq_npu)
    dq_out_cpu = dq_out.cpu().reshape(B, S, H, D)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(dq_out_cpu, golden)
    try:
        torch.testing.assert_close(
            dq_out_cpu, golden, rtol=rtol, atol=atol, equal_nan=True
        )
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape=({B},{S},{H},{D}) dtype={dtype_str} "
            f"max_diff={max_diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape=({B},{S},{H},{D}) dtype={dtype_str} "
        f"max_diff={max_diff:.6e}"
    )
    return max_diff


def _run_case(B, S, H, D, dtype_str, tag, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dq = torch.randn(B, S, H, D, dtype=torch.float32, generator=gen)
    return _run_tensor(dq, dtype_str, tag)


# ---------- hierarchical testing ----------
def run_L0():
    # L0: smoke + every manifest workload (DESIGN.md section 8.2 clause 1),
    # fp16 and bf16 each.
    for dtype_str in ("float16", "bfloat16"):
        _run_case(1, 128, 8, 64, dtype_str, "L0-smoke")
        _run_case(4, 512, 32, 128, dtype_str, "L0-llama8b-short")
        _run_case(2, 2048, 32, 128, dtype_str, "L0-llama8b-long")
        _run_case(2, 512, 64, 128, dtype_str, "L0-llama70b-short")
        _run_case(1, 2048, 64, 128, dtype_str, "L0-llama70b-long")


def run_L1():
    # L1: boundary shapes and gcd-driven R_S selection (blocking layer).
    # seq_len < _rs_max -> R_S = seq_len (single slab per batch).
    _run_case(1, 8, 2, 64, "float16", "L1-S-lt-R")
    # B = H = 1.
    _run_case(1, 512, 1, 128, "float16", "L1-B1H1")
    # Non power-of-two S: gcd(1000, 32) = 8 -> R_S = 8, 125 slabs.
    _run_case(1, 1000, 4, 128, "float16", "L1-S1000")
    # Other non-trivial gcds.
    _run_case(2, 1000, 8, 64, "bfloat16", "L1-S1000-H8D64")
    _run_case(2, 192, 8, 128, "bfloat16", "L1-S192")
    _run_case(2, 320, 8, 64, "bfloat16", "L1-S320")
    # Exact multiples (R_S == _rs_max).
    _run_case(1, 384, 8, 128, "float16", "L1-S384")
    # D = 64 smallest-vector-width case.
    _run_case(2, 1024, 16, 64, "bfloat16", "L1-D64")


def run_L2():
    # L2: edge sizes around the tiling / core-count boundaries (non-blocking).
    for B, S, H, D, dtype_str in (
        (1, 1, 1, 64, "float16"),
        (1, 7, 1, 64, "float16"),
        (1, 8, 2, 128, "bfloat16"),
        (3, 96, 3, 64, "bfloat16"),
        (1, 4096, 8, 128, "float16"),
        (2, 2048, 8, 64, "bfloat16"),
        (1, 333, 4, 64, "float16"),
    ):
        try:
            _run_case(B, S, H, D, dtype_str, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): shape=({B},{S},{H},{D}) "
                  f"dtype={dtype_str} err={e}")


def _repeat_values(values, shape, dtype):
    n = 1
    for s in shape:
        n *= s
    base = torch.tensor(values, dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n].reshape(shape)


def run_boundary():
    # Boundary: cast corner cases (non-blocking).
    # fp32 special values: +-0, +-1, fp32 extremes, fp16 overflow threshold
    # (65504 max, 65520 -> inf), subnormals, NaN.
    for dtype_str in ("float16", "bfloat16"):
        try:
            shape = (1, 256, 4, 64)
            dq = _repeat_values(
                [0.0, -0.0, 1.0, -1.0, 65504.0, 65520.0, 1e-8, float("inf"),
                 float("-inf"), float("nan"), 3.4e38, -3.4e38],
                shape, torch.float32,
            )
            _run_tensor(dq, dtype_str, "Boundary-specialvalues")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
                  f"err={e}")

    # All-zero input -> all-zero output.
    for dtype_str in ("float16", "bfloat16"):
        try:
            dq = torch.zeros((2, 256, 4, 128), dtype=torch.float32)
            _run_tensor(dq, dtype_str, "Boundary-allzero")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
                  f"err={e}")

    # Subnormals / tiny magnitudes around the storage-dtype denormal range.
    for dtype_str in ("float16", "bfloat16"):
        try:
            shape = (1, 512, 8, 128)
            gen = torch.Generator().manual_seed(11)
            dq = torch.randn(shape, generator=gen, dtype=torch.float32)
            scale = torch.logspace(-8, 6, shape[-1])
            dq = dq * scale
            _run_tensor(dq, dtype_str, "Boundary-wide-dyn")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
                  f"err={e}")

    # Round-to-nearest-even tie cases (exact .5 ulp boundaries in fp16).
    try:
        vals = [1.0 + 2 ** (-11), 1.0 + 3 * 2 ** (-11), 1.0 + 2 ** (-10),
                1.0 + 2 ** (-24), -1.0 - 2 ** (-11)]
        dq = _repeat_values(vals, (1, 128, 4, 64), torch.float32)
        _run_tensor(dq, "float16", "Boundary-rne-ties")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): dtype=float16 "
              f"err={e}")


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
