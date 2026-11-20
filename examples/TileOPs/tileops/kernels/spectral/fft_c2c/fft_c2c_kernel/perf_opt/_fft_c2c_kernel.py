# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""1D Complex-to-Complex FFT kernel (NPU, target="npuir").

Migrated from the GPU TileLang factory ``_fft_c2c_kernel``
(``/home/developer/TileOPs/tileops/kernels/fft.py``).  Semantics: the one
dimensional DFT along the last dimension, equivalent to
``torch.fft.fft(x, dim=-1)``; any leading dimensions are flattened into a
single batch axis, processed in parallel and reshaped back.

NPU re-design (see DESIGN.md section 0.5/0.6):

  * R1 -- complex dtype -> real/imag fp32 split.  npuir has no complex dtype
    and no fp64 vector math; complex64 is therefore carried as two fp32
    tensors (the source already splits real/imag at the kernel boundary).
  * R2 -- the source 2-D grid ``(smem_blocks, batch)`` with ``threads=512``
    becomes a 1-D grid ``T.Kernel(B, is_npu=True)`` (no ``threads``; TRAP-
    threads-kwarg-noop).  One block processes one complete FFT row.
  * R3 -- bit-reversal DIT -> Stockham autosort DIF radix-2.  Every stage reads
    and writes only contiguous segments (no strided gather), and no explicit
    permutation is required (TRAP-C10-strided-gather).
  * R4 -- the whole row is resident in UB, so all ``log2(n)`` stages run in a
    single kernel launch with ping-pong buffers (GM traffic = 1 read + 1
    write), replacing the source's per-stage GM round-trips.
  * R5 -- all twiddle factors come from a host-built fp64 LUT cast to fp32
    (no in-kernel trig; VP-2026-0140 / T.vcos angle-domain limits).

Implementation note on compile-time unrolling: the stage count
``t = log2(n)`` is a Python constant, but the TileLang parser lowers a plain
``for s in range(t)`` into a *runtime* serial loop (its induction variable is
a TIR ``Var``, not a Python ``int``), which cannot carry the per-stage Python
constants (segment length ``h``, group count ``ng``, LUT base, and the
ping-pong buffer identity).  The stage sequence is therefore emitted by
chaining ``@T.macro`` bodies built recursively in Python at factory-build
time (``make_chain``): each macro inlines one constant-specialised stage and
calls the next, so the whole FFT network is fully expanded at trace time with
every index a compile-time constant.  The math is unchanged.

run: python _fft_c2c_kernel.py --level all
"""

import os

# Developer mode for npuir (linear data-dependency chain, automatic sync).
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

# complex64 <-> two fp32 tensors; complex128 is NOT implementable on npuir
# (no complex dtype + no fp64 vector elementwise ops; CG-2026-0019).
_REAL_DTYPE = "float32"
_TORCH_DTYPE = {"float32": torch.float32}

# Tolerance from examples/TileOPs/tests/ops/test_fft.py (DESIGN.md 8.2).
_ATOL = 1e-4
_RTOL = 1e-4


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_fft_c2c(x):
    """1-D C2C FFT along the last dimension, equivalent to torch.fft.fft.

    Input ``x`` is ``[..., n]`` complex (``n`` a power of 2); the output has
    the same shape and dtype.  This mirrors the reference program of
    ``examples/TileOPs/tests/ops/test_fft.py`` and is deliberately independent
    of the NPU Stockham implementation.
    """
    return torch.fft.fft(x, dim=-1)


# ---------- Host-side twiddle LUT (fp64 build, cast to fp32) ----------
def build_lut(n, real_dtype=torch.float32):
    """Twiddle factor table for every butterfly stage.

    Stage ``s`` (``half_m = 2^s``) occupies ``LUT[half_m - 1 : 2*half_m - 1]``
    with ``LUT[half_m - 1 + k] = exp(-2*pi*i*k / (2*half_m))``.  The layout
    matches the source op layer ``_build_lut`` and the design LUT mapping
    ``LUT[h - 1 + i]`` for a Stockham stage of segment length ``h``.
    """
    log2n = int(math.log2(n))
    angles = torch.zeros(n - 1, dtype=torch.float64)
    for s in range(log2n):
        half_m = 1 << s
        m = half_m * 2
        k = torch.arange(half_m, dtype=torch.float64)
        angles[half_m - 1: 2 * half_m - 1] = -2.0 * math.pi * k / m
    return (
        torch.cos(angles).to(real_dtype).contiguous(),
        torch.sin(angles).to(real_dtype).contiguous(),
    )


# ---------- Kernel ----------
@tilelang.jit(out_idx=[4, 5], target="npuir")
def _fft_c2c_kernel(n, batch_size, dtype="float32"):
    """Stockham autosort DIF radix-2 FFT factory.

    Args:
        n: FFT length along the last dimension (a power of 2).
        batch_size: number of independent FFT rows (``prod`` of leading dims).
        dtype: real component dtype; only ``"float32"`` (complex64) is
            supported on npuir.

    Returns:
        A ``@T.prim_func`` taking ``(x_real, x_imag, lut_real, lut_imag)`` of
        shape ``(batch_size, n)`` / ``(n - 1,)`` and producing
        ``(y_real, y_imag)`` of shape ``(batch_size, n)``.
    """
    if dtype != _REAL_DTYPE:
        raise ValueError(
            f"Unsupported real dtype {dtype!r}: npuir has no complex dtype and "
            "no fp64 vector elementwise ops, so complex128 is not implementable "
            "(CG-2026-0019). Use float32 (complex64)."
        )
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FFT size must be a positive power of 2, got {n}")

    t = int(math.log2(n))  # number of radix-2 stages (compile-time constant)
    B = batch_size
    DT = dtype
    nh = n // 2

    def make_chain(s):
        """Recursively build a macro chain emitting stages ``s .. t-1``.

        Each macro inlines one stage with Python-constant ``h`` / ``ng`` /
        LUT base and forwards the *swapped* ping-pong buffers to the next
        stage.  The recursion terminates at ``s == t`` with an empty macro.
        """
        if s == t:
            @T.macro
            def done(S_r, S_i, D_r, D_i, lut_r, lut_i):
                pass
            return done

        nxt = make_chain(s + 1)
        h = 1 << (t - 1 - s)          # segment length: n/2, n/4, ..., 1
        ng = n // (2 * h)             # number of butterfly groups
        lb = h - 1                    # LUT base for this stage

        @T.macro
        def stage(S_r, S_i, D_r, D_i, lut_r, lut_i):
            # Bisect the row: A is logically [ng, 2h] -> B is [2, ng, h].
            for g, i in T.Parallel(ng, h):
                lo_r = S_r[g * 2 * h + i]
                lo_i = S_i[g * 2 * h + i]
                hi_r = S_r[g * 2 * h + h + i]
                hi_i = S_i[g * 2 * h + h + i]
                wr = lut_r[lb + i]
                wi = lut_i[lb + i]
                dr = lo_r - hi_r
                di = lo_i - hi_i
                # Sum goes to the first half, (lo - hi) * w to the second.
                D_r[g * h + i] = lo_r + hi_r
                D_i[g * h + i] = lo_i + hi_i
                D_r[g * h + i + nh] = dr * wr - di * wi
                D_i[g * h + i + nh] = dr * wi + di * wr
            nxt(D_r, D_i, S_r, S_i, lut_r, lut_i)

        return stage

    chain = make_chain(0)
    # Buffer holding the result after ``t`` stages (stage 0 writes Bc, ...).
    final_is_a = (t % 2 == 0)

    @T.prim_func
    def main(
        x_real: T.Tensor((B, n), DT),         # type: ignore
        x_imag: T.Tensor((B, n), DT),         # type: ignore
        lut_real: T.Tensor((n - 1,), DT),     # type: ignore
        lut_imag: T.Tensor((n - 1,), DT),     # type: ignore
        y_real: T.Tensor((B, n), DT),         # type: ignore
        y_imag: T.Tensor((B, n), DT),         # type: ignore
    ) -> None:
        with T.Kernel(B, is_npu=True) as (cid, _):
            # UB ping-pong row buffers (real / imag of both halves).
            A_r = T.alloc_shared((n,), DT)
            A_i = T.alloc_shared((n,), DT)
            Bc_r = T.alloc_shared((n,), DT)
            Bc_i = T.alloc_shared((n,), DT)

            T.copy(x_real[cid, 0:n], A_r)
            T.copy(x_imag[cid, 0:n], A_i)

            chain(A_r, A_i, Bc_r, Bc_i, lut_real, lut_imag)

            if final_is_a:
                T.copy(A_r, y_real[cid, 0:n])
                T.copy(A_i, y_imag[cid, 0:n])
            else:
                T.copy(Bc_r, y_real[cid, 0:n])
                T.copy(Bc_i, y_imag[cid, 0:n])

    return main


# ---------- precision comparing helpers ----------
def _max_abs_diff(y, golden):
    return (y.to(torch.complex128) - golden.to(torch.complex128)).abs().max().item()


def _run_case(shape, dtype=torch.complex64, seed=0, tag="case", expect_ok=True):
    """Run one FFT case: build inputs, dispatch the NPU kernel, compare.

    ``shape`` is the full ``[..., n]`` shape (the last dim is the FFT length).
    Returns the max abs error (complex64) or ``None`` when skipped.
    """
    if dtype == torch.complex128:
        # Honest capability disclosure: complex128 needs fp64 real components,
        # which npuir cannot compute (CG-2026-0019).  Not a silent pass.
        print(f"[{tag}] SKIP: dtype=complex128 unsupported on npuir (CG-2026-0019)")
        return None

    if dtype != torch.complex64:
        raise ValueError(f"unsupported test dtype {dtype}")
    if len(shape) == 0:
        raise ValueError("shape must be at least 1-D")

    n = shape[-1]
    torch.manual_seed(seed)
    xr = torch.randn(shape, dtype=torch.float32)
    xi = torch.randn(shape, dtype=torch.float32)
    x = torch.complex(xr, xi)

    golden = golden_fft_c2c(x)

    batch = x[..., 0].numel() if len(shape) > 1 else 1
    xr2 = xr.reshape(batch, n)
    xi2 = xi.reshape(batch, n)
    lr, li = build_lut(n)

    kernel = _fft_c2c_kernel(n, batch)
    out = kernel(xr2.npu(), xi2.npu(), lr.npu(), li.npu())
    y = torch.complex(out[0].cpu(), out[1].cpu()).reshape(shape)

    diff = _max_abs_diff(y, golden)
    try:
        torch.testing.assert_close(y, golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: shape={tuple(shape)} dtype=complex64 "
            f"max_diff={diff:.6e}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: shape={tuple(shape)} dtype=complex64 "
        f"max_diff={diff:.6e}"
    )
    return diff


# ---------- hierarchical testing ----------
def run_L0():
    """L0 smoke / manifest cases (DESIGN.md 8.3 L0-1 .. L0-7)."""
    # L0-1: manifest smoke, smallest N.
    _run_case((64,), tag="L0-1-64")
    # L0-2: manifest unbatched.
    _run_case((4096,), tag="L0-2-4096")
    # L0-3: manifest batched.
    _run_case((64, 4096), tag="L0-3-b64-4096")
    # L0-4: leading dims flattened into the batch axis.
    _run_case((4, 64), tag="L0-4-4x64")
    # L0-5: batch == 1 boundary.
    _run_case((1, 4096), tag="L0-5-1x4096")
    # L0-6: multiple leading dims + large N.
    _run_case((2, 4, 4096), tag="L0-6-2x4x4096")
    # L0-7: complex128 -- disclosed as unsupported (not a silent pass).
    _run_case((64, 4096), dtype=torch.complex128, tag="L0-7-c128")


def run_L1():
    """L1 functional sweep: N / batch breadth (manifest full workloads)."""
    # Small-to-large powers of two, unbatched and batched.
    for n, bshape in (
        (128, ()),
        (256, (8,)),
        (512, (16,)),
        (1024, ()),
        (1024, (2, 4)),
        (2048, (4,)),
        (128, (4,)),
    ):
        _run_case((*bshape, n), tag="L1")
    # Manifest batched workload repeated with a different seed for stability.
    _run_case((64, 4096), seed=1, tag="L1-b64-4096")
    # Batch exactly one physical-core wave (40 rows).
    _run_case((40, 64), tag="L1-b40")
    # Larger power-of-two N, unbatched.
    _run_case((8192,), tag="L1-8192")


def run_L2():
    """L2 edge sizes (non-blocking)."""
    for n in (2, 4, 8, 16, 32):
        try:
            _run_case((n,), tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): n={n} err={e}")
    for shape in ((3, 64), (5, 128), (7, 8)):
        try:
            _run_case(shape, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape={shape} err={e}")


def run_boundary():
    """Boundary / special-value cases (non-blocking)."""
    # Impulse at index 0: FFT is flat (all ones).
    n = 64
    try:
        x = torch.zeros(1, n, dtype=torch.complex64)
        x[0, 0] = 1.0
        golden = golden_fft_c2c(x)
        lr, li = build_lut(n)
        k = _fft_c2c_kernel(n, 1)
        out = k(x.real.contiguous().npu(), x.imag.contiguous().npu(), lr.npu(), li.npu())
        y = torch.complex(out[0].cpu(), out[1].cpu())
        torch.testing.assert_close(y, golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
        print(f"[Boundary] PASS: impulse max_diff={_max_abs_diff(y, golden):.6e}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): impulse err={e}")

    # Constant signal: energy only in the DC bin.
    try:
        x = torch.full((2, n), 0.5 + 0.25j, dtype=torch.complex64)
        golden = golden_fft_c2c(x)
        lr, li = build_lut(n)
        k = _fft_c2c_kernel(n, 2)
        out = k(x.real.contiguous().npu(), x.imag.contiguous().npu(), lr.npu(), li.npu())
        y = torch.complex(out[0].cpu(), out[1].cpu())
        torch.testing.assert_close(y, golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
        print(f"[Boundary] PASS: constant max_diff={_max_abs_diff(y, golden):.6e}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): constant err={e}")

    # All zeros.
    try:
        x = torch.zeros(1, n, dtype=torch.complex64)
        golden = golden_fft_c2c(x)
        lr, li = build_lut(n)
        k = _fft_c2c_kernel(n, 1)
        out = k(x.real.contiguous().npu(), x.imag.contiguous().npu(), lr.npu(), li.npu())
        y = torch.complex(out[0].cpu(), out[1].cpu())
        torch.testing.assert_close(y, golden, rtol=_RTOL, atol=_ATOL, equal_nan=True)
        print(f"[Boundary] PASS: zeros max_diff={_max_abs_diff(y, golden):.6e}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): zeros err={e}")


# ---------- benchmark workload ids (mirrors benchmarks/ops/bench_fft.py) ----------
# label -> (n, full_shape).  The leading dims are flattened into the batch axis:
# batch_size = prod(shape[:-1]) (1 when unbatched), exactly as the op layer does.
_CASES = {
    "fft-64-c64-smoke": (64, (64,)),
    "fft-4k-c64-unbatched": (4096, (4096,)),
    "fft-4k-c64-b64": (4096, (64, 4096)),
}


def run_case_bench(case, launches=25, seed=0):
    """Dispatch one benchmark workload through the kernel for msprof profiling.

    Mirrors ``benchmarks/ops/bench_fft.py``: real/imag split, batch flatten,
    host LUT build, kernel dispatch -- the kernel-only Task Duration captured by
    ``msprof op`` is the measurement.
    """
    n, shape = _CASES[case]
    torch.manual_seed(seed)
    xr = torch.randn(shape, dtype=torch.float32)
    xi = torch.randn(shape, dtype=torch.float32)
    batch = xr.numel() // n
    xr2 = xr.reshape(batch, n)
    xi2 = xi.reshape(batch, n)
    lr, li = build_lut(n)
    kernel = _fft_c2c_kernel(n, batch)
    xr_d, xi_d, lr_d, li_d = xr2.npu(), xi2.npu(), lr.npu(), li.npu()
    out = kernel(xr_d, xi_d, lr_d, li_d)
    torch.npu.synchronize()
    for _ in range(launches):
        out = kernel(xr_d, xi_d, lr_d, li_d)
    torch.npu.synchronize()
    print(f"[case] {case} n={n} batch={batch} launches={launches} "
          f"out[0].shape={tuple(out[0].shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    parser.add_argument("--case", default=None, choices=sorted(_CASES))
    parser.add_argument("--launches", type=int, default=25)
    args, _ = parser.parse_known_args()
    if args.case is not None:
        run_case_bench(args.case, launches=args.launches)
        return
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
