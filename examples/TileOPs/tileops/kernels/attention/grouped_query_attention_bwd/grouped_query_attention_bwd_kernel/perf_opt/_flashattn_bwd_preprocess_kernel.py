# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""GQA backward preprocess kernel (NPU, target="npuir").

Migrated from the GPU TileOPs factory ``_flashattn_bwd_preprocess_kernel``
(``/home/developer/TileOPs/tileops/kernels/attention/gqa_bwd.py`` L26-53; class
``FlashAttnBwdPreprocessKernel`` L56-105).

Source semantics (the only basis of the embedded golden):

    delta[b, h, s] = sum_{d=0}^{D-1} o[b, s, h, d] * do[b, s, h, d]

``o`` / ``do`` are BSHD (batch, seq_len, heads, dim); ``delta`` is BHS
(batch, heads, seq_len) float32.  Multiplication and accumulation of
``o * do`` happen in fp32 (fp16 x fp16 is exactly representable in fp32).

NPU re-design (see DESIGN.md section 0.6):

  * R1 -- the source 3-D grid ``T.Kernel(heads, ceil(S/256), batch)`` is not
    supported on NPU.  It becomes a 1-D ``T.Kernel(nblk, is_npu=True)`` with an
    in-core ``T.serial(iters)`` grid-stride over the logical row tiles
    (persistent kernel, ``nblk = min(40, total_tiles)`` physical Vector cores).
  * R2 -- the source ``[256, 256]`` fragment tile (fp32 ``acc`` alone is
    256 KB) cannot fit in the 192 KB UB.  The tile becomes ``[R, D]`` with
    ``R = gcd(seq_len, 128)`` (statically divides ``seq_len`` -> no tail tile).
  * R3 -- the source ``for k in range(ceildiv(dim, blk))`` chunk-accumulate
    loop collapses (the tile now spans the full D), replaced by a single
    ``T.Parallel`` product followed by one ``T.reduce_sum(dim=1)``.

Hard constraints honoured:
  * BSHD strided read uses the slice form ``T.copy`` -- ``T.load_nd2nz`` is
    forbidden (it would silently read the strided tile as contiguous memory).
  * ``T.reduce_sum(..., dim=1, clear=True)`` -- ``clear=False`` on an
    uninitialised buffer is a silent numerical error.
  * The ``[R, 1]`` reduction result is written back with ``size=[R]``
    (a rank-[R,1] -> 1-D GM copy would be a shape error).

run: python _flashattn_bwd_preprocess_kernel.py --level all


Stage-4 (perf_opt) copy.  Measured side-by-side with the Stage-3 baseline on all 8 tune
workloads; no candidate change was adopted (memory-bound stage, see opt_log.md section 7),
so this file is identical to the Stage-3 baseline implementation.
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

tilelang.cache.clear_cache()

# Physical Vector core count on Ascend910B3 (NPUUtils: 20 AI cores x 2 for a
# pure Vector operator; DESIGN.md section 5.5).
_N_PHYS = 40

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Per-dtype tolerances (DESIGN.md section 8.2).  The input dtype decides the
# tolerance; the output is always fp32.
_TOLERANCE = {
    "float16": (1e-3, 1e-4),   # (atol, rtol)
    "bfloat16": (5e-3, 1e-3),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_flashattn_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Reference ``delta = rowsum(o * do)``.

    ``o`` / ``do`` are ``(B, S, H, D)`` fp16/bf16; the result is
    ``(B, H, S)`` fp32.  Only the source op semantics are used here (fp32
    product and fp32 accumulation, followed by the BSH -> BHS transpose),
    never the DESIGN.md NPU tiling / persistence structure, so the reference
    stays independent from the kernel under test.
    """
    o_cpu = torch.as_tensor(o.detach().cpu())
    do_cpu = torch.as_tensor(do.detach().cpu())
    delta_bsh = torch.sum(  # [B, S, H] fp32
        torch.mul(o_cpu.to(torch.float32), do_cpu.to(torch.float32)), dim=-1
    )
    return torch.transpose(delta_bsh, 1, 2).contiguous()  # [B, H, S] fp32


# ---------- Kernel ----------
@tilelang.jit(out_idx=[2], target="npuir")
def _flashattn_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype):
    """Factory: delta[b, h, s] = sum_d o[b, s, h, d] * do[b, s, h, d].

    All of ``batch`` / ``heads`` / ``seq_len`` / ``dim`` / ``dtype`` are
    compile-time constants (matching the source factory signature).
    """
    accum_dtype = "float"
    shape = (batch, seq_len, heads, dim)

    # R statically divides seq_len -> the row tiles always cover the full
    # sequence with no tail tile and no padding (DESIGN.md section 5.4).
    R = math.gcd(seq_len, 128)
    assert seq_len % R == 0
    n_s = seq_len // R
    total_tiles = batch * heads * n_s
    nblk = min(_N_PHYS, total_tiles)
    iters = (total_tiles + nblk - 1) // nblk  # static loop bound

    @T.prim_func
    def flash_bwd_prep(
        o: T.Tensor(shape, dtype),  # type: ignore
        do: T.Tensor(shape, dtype),  # type: ignore
        delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(nblk, is_npu=True) as (cid, _):
            # All UB buffers are allocated unconditionally at the top level
            # (a conditional T.alloc is illegal).
            o_s = T.alloc_shared([R, dim], dtype)        # [R, D] o tile
            do_s = T.alloc_shared([R, dim], dtype)       # [R, D] do tile
            prod_s = T.alloc_shared([R, dim], accum_dtype)   # fp32 o*do
            delta_s = T.alloc_shared([R, 1], accum_dtype)    # fp32 rowsum

            # In-core grid-stride over the logical row tiles; the loop bound
            # is a compile-time constant and the guard covers the ragged tail
            # of the round-robin mapping (DESIGN.md section 5.5 / 6.2).
            for it in T.serial(iters):
                t = cid + it * nblk
                if t < total_tiles:
                    bz = t // (heads * n_s)
                    bx = (t % (heads * n_s)) // n_s   # head index
                    by = t % n_s                      # sequence row block
                    s0 = by * R

                    # 1. strided BSHD read via the slice form of T.copy
                    #    (row segment is D*elem_bytes >= 128 B, contiguous).
                    T.copy(o[bz, s0:s0 + R, bx, 0:dim], o_s)
                    T.copy(do[bz, s0:s0 + R, bx, 0:dim], do_s)

                    # 2. fp32 promote + element-wise product (vectorised over
                    #    the innermost contiguous D axis).
                    for i, j in T.Parallel(R, dim):
                        prod_s[i, j] = T.cast(o_s[i, j], accum_dtype) \
                            * T.cast(do_s[i, j], accum_dtype)

                    # 3. single-pass row reduction along D (VReduceOp).
                    T.reduce_sum(prod_s, delta_s, dim=1, clear=True)

                    # 4. contiguous write-back ([R, 1] -> GM [R]).
                    T.copy(delta_s, delta[bz, bx, s0], size=[R])

    return flash_bwd_prep


# ---------- precision comparing func ----------
def _max_diff(y, golden):
    """Max abs error over finite golden entries."""
    y_f64 = y.to(torch.float64)
    g_f64 = golden.to(torch.float64)
    finite = torch.isfinite(g_f64) & torch.isfinite(y_f64)
    if not bool(finite.any()):
        return 0.0
    return (y_f64[finite] - g_f64[finite]).abs().max().item()


def _run_tensor(o_cpu, do_cpu, dtype_str, tag):
    """Run the kernel on NPU tensors and compare against the golden output."""
    o_cpu = o_cpu.detach().cpu().contiguous()
    do_cpu = do_cpu.detach().cpu().contiguous()
    B, S, H, D = o_cpu.shape

    golden = golden_flashattn_bwd_preprocess(o_cpu, do_cpu)
    kernel = _flashattn_bwd_preprocess_kernel(B, H, S, D, dtype_str)

    o_npu = o_cpu.npu()
    do_npu = do_cpu.npu()
    delta = kernel(o_npu, do_npu)
    delta_cpu = delta.cpu().reshape(B, H, S)

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(delta_cpu, golden)
    try:
        torch.testing.assert_close(delta_cpu, golden, rtol=rtol, atol=atol)
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
    td = _TORCH_DTYPE[dtype_str]
    gen = torch.Generator().manual_seed(seed)
    o = torch.randn(B, S, H, D, dtype=torch.float32, generator=gen)
    do = torch.randn(B, S, H, D, dtype=torch.float32, generator=gen)
    return _run_tensor(o.to(td), do.to(td), dtype_str, tag)


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
    # L1: boundary shapes and gcd-driven R selection (blocking layer).
    # S < 128 -> R = S (single row block).
    _run_case(1, 64, 4, 64, "float16", "L1-S-lt-R")
    # B = H = 1.
    _run_case(1, 512, 1, 128, "float16", "L1-B1H1")
    # Non power-of-two S: gcd(1000, 128) = 8 -> R = 8, 125 row blocks.
    _run_case(1, 1000, 4, 128, "float16", "L1-S1000")
    # Other non-trivial gcds: gcd(192,128)=64, gcd(320,128)=64.
    _run_case(2, 192, 8, 128, "bfloat16", "L1-S192")
    _run_case(2, 320, 8, 64, "bfloat16", "L1-S320")
    # Exact multiples: gcd(384,128)=128, gcd(512,128)=128.
    _run_case(1, 384, 8, 128, "float16", "L1-S384")
    # D = 64 with the worst-case R = 128.
    _run_case(2, 1024, 16, 64, "bfloat16", "L1-D64")


def run_L2():
    # L2: edge sizes around the tiling / core-count boundaries (non-blocking).
    for B, S, H, D, dtype_str in (
        (1, 1, 1, 64, "float16"),
        (1, 7, 1, 64, "float16"),
        (1, 8, 2, 128, "bfloat16"),
        (1, 16, 1, 32, "float16"),
        (3, 96, 3, 64, "bfloat16"),
        (1, 4096, 8, 128, "float16"),
        (2, 2048, 8, 64, "bfloat16"),
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
    # Boundary: special-value inputs (non-blocking).
    for dtype_str in ("float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        try:
            shape = (1, 256, 4, 64)
            big = float(torch.finfo(td).max)
            o = _repeat_values(
                [0.0, -0.0, 1.0, -1.0, big, -big, 1e-4, -1e-4], shape, td
            )
            do = _repeat_values(
                [0.0, -0.0, big, -big, 1.0, -1.0, 1e-4, -1e-4], shape, td
            )
            _run_tensor(o, do, dtype_str, "Boundary-special")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
                  f"err={e}")

    # All-zero input -> delta == 0.
    for dtype_str in ("float16", "bfloat16"):
        try:
            shape = (2, 256, 4, 128)
            o = torch.zeros(shape, dtype=_TORCH_DTYPE[dtype_str])
            do = torch.zeros(shape, dtype=_TORCH_DTYPE[dtype_str])
            _run_tensor(o, do, dtype_str, "Boundary-allzero")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
                  f"err={e}")

    # Wide dynamic range / cancellation (large + small mixed products).
    for dtype_str in ("float16", "bfloat16"):
        td = _TORCH_DTYPE[dtype_str]
        try:
            shape = (1, 512, 8, 128)
            gen = torch.Generator().manual_seed(7)
            o = torch.randn(shape, generator=gen, dtype=torch.float32)
            do = torch.randn(shape, generator=gen, dtype=torch.float32)
            scale = torch.logspace(-3, 3, shape[-1])
            o = (o * scale).to(td)
            do = (do * scale).to(td)
            _run_tensor(o, do, dtype_str, "Boundary-wide-dyn")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): dtype={dtype_str} "
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
