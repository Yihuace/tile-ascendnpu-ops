# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""GroupNorm forward kernel, affine primary path (NPU, Developer mode).

Migrated from the GPU repo's ``_group_norm_kernel`` (TileOPs
``GroupNormFwdOp``, affine variant) to TileLang ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    The input ``x`` of shape ``(N, C, *spatial)`` is viewed as a 2-D
    matrix of ``M = N * num_groups`` rows by ``D = (C/num_groups) *
    spatial_size`` columns, and each row is normalised independently::

        mean_r  = sum_j x_rj / D
        var_r   = sum_j (x_rj - mean_r)^2 / D      (biased, ddof=0)
        rstd_r  = 1 / sqrt(var_r + eps)
        y_rj    = (x_rj - mean_r) * rstd_r          (kernel layer)

    per-channel affine ``y * weight[c] + bias[c]`` is then applied by the
    op layer, outside the kernel.

Affine layering (preserved from GPU, do not collapse):
    - Kernel layer: ``main(x, weight, bias, y)`` (four tensors,
      ``out_idx=[3]``) where ``weight``/``bias`` are **row-broadcast**
      vectors of shape ``(D_padded,)``.  On the only migration call path
      (``GroupNormFwdOp``) the wrapper always passes a **unit** affine
      (ones / zeros), so the kernel-layer multiply-add is an IEEE
      bit-exact no-op and is omitted (DESIGN.md section 1.6.1 O5).  The
      ``weight``/``bias`` parameters are kept purely for the interface
      contract (per-channel ``(C,)`` weights must never be fed to the
      kernel).
    - Op layer: ``GroupNormFwdOp.forward(x, weight, bias)`` takes
      per-channel ``(C,)`` weight/bias and applies
      ``y * weight.reshape((1,C)+(1,)*len(spatial)) + bias.reshape(...)``
      outside the kernel (replicated, unchanged, in the test harness).

NPU redesign (DESIGN.md section 0.6):
    R1: persistent core split ``num_kernels = min(num_logical,
        vector_cores)`` with an in-core ``T.serial`` grid-stride loop and
        an if-guard for out-of-range row blocks (static bounds).
    R2: ``block_m`` driven by the UB budget table (DESIGN.md section
        4.5), not the GPU structural ``block_m=1``.
    R3: tail handling via ``T.min`` + src/dst dual explicit slices +
        garbage-row drop (rows are independent; ``reduce dim=1`` and the
        row-broadcast v-ops never mix rows; only valid rows are copied
        out).
    R4: explicit fp32 transit chain for fp16/bf16: ``T.vcast(rint)`` up to
        fp32 -> fp32 v-prefix chain -> ``T.vcast(rint)`` single
        round-back (bf16 has no Vector arithmetic; fp16 transit matches
        the golden's fp32 opmath).
    R5: ``rstd = 1/sqrt(var)`` composed as ``T.vsqrt`` + ``T.vdiv``
        (``sqrt(v)/v``); the raw ``T.vrsqrt`` instruction is approximate
        on npuir (measured max rel err ~2.87e-3) and violates the fp32
        gate (TRAP-vrsqrt-plain-precision).
    R6: kernel-layer unit affine elided; per-channel affine stays in the
        op layer.
    R7: exact ``D`` (``ALIGNMENT = 1``, no row padding, no host
        ``F.pad``).

Dropped GPU mechanisms (DESIGN.md section 0.5, intent handed over):
    256-alignment padding + pad variance correction (exact-D identity
    O4), host ``F.pad`` of x/weight/bias, ``threads`` / block-thread
    two-level parallelism (Vector lanes + persistent split).

Factory contract (source interface unchanged):
    _group_norm_kernel(M, D, eps, dtype)
      -> _func(block_m=UB-table default) -> main
    main(x, weight, bias, y) returns y (out_idx=[3]); weight/bias are
    row-broadcast ``(D_padded,)`` and are not read on this path.
    ``ALIGNMENT`` / ``_align_up`` are re-exported for the wrapper's
    extracted-import contract (``ALIGNMENT = 1`` on NPU).

Run the embedded hierarchical tests:
    python _group_norm_kernel.py --level L0
    python _group_norm_kernel.py --level all
"""

import os

# Developer mode: alloc_shared maps to UB, compiler auto-sync
# (DESIGN.md sections 2 and 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming-mode literal (TileOPs harness migration contract, Stage 5
# syncs ``Kernel.ascend_mode`` from this value before running the report).
ASCEND_MODE = "Developer"

__all__ = [
    "ALIGNMENT",
    "ASCEND_MODE",
    "_align_up",
    "_group_norm_kernel",
    "golden_group_norm",
]


# ---------------------------------------------------------------------------
# Backend alignment helpers (integration glue, Stage 5)
#
# Re-exported for the wrapper's extracted-import contract (the NPU wrapper
# ``group_norm.py`` imports ``ALIGNMENT`` / ``_align_up`` / ``_group_norm_kernel``
# from ``._group_norm_fwd_kernels``; Stage 5 rewrites that import to this
# module).  On CUDA ``ALIGNMENT`` is 256 and drives an SMEM row-padding plus
# a host-side ``F.pad``; NPU has no such alignment constraint, so it is set to
# 1 -- ``D_padded == D``, ``pad == 0`` and the wrapper's ``F.pad`` / output
# truncation branches never fire (DESIGN.md sections 0.5 and 0.6 R7).
# ---------------------------------------------------------------------------

ALIGNMENT = 1


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported dtypes (factory validation, identical to the source).
_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# UB capacity per Vector core (Ascend910B3, ``AscendArch().ub_cap`` == 196608).
_UB_CAPACITY_BYTES = 196_608

# Measured multi-buffered UB cost per resident element for this buffer
# structure (stage + x_f32 + sq_f32 + 5 x (block_m,1)), on BOTH dtype
# paths, whenever the persistent serial loop is active
# (num_local_tasks >= 2).  Same family/structure as the ada_layer_norm
# migration, whose compile-probe overflow messages read
# 245760 B = 20*3*4096 and 240000 B = 20*4*3000 (TRAP-UB-multibuffer-inflation
# third proof; DESIGN.md section 4.5).  The guard is expressed in resident
# elements: 196608 / 20 = 9830, kept at 9216 (= the design's largest
# sanctioned config bm=8 x D=1152, >= 12 KB margin).
_UB_BYTES_PER_ELEM = 20
_UB_MAX_ELEMS = 9_216

# block_m default cap (DESIGN.md section 4.5).
_BLOCK_M_CAP = 8

# Precision tolerances (DESIGN.md section 8.2, identical to the source repo
# test_group_norm.py ``_get_tolerances``).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


def _affine_for(dtype_str):
    """Gate affine mode per dtype.

    fp32/bf16 run the full op-layer replica with a real random per-channel
    (C,) affine: the op layer's dtype-affine double rounding is negligible
    for fp32 (atol 1e-5) and absorbed by the bf16 tolerance (1.6e-2).

    fp16 uses the kernel-level unit affine (DESIGN.md section 8.1): the op
    layer applies the per-channel affine *after* the kernel's fp16
    round-trip, so an fp16 affine amplifies the normalised tensor's
    quantisation by |w| and can cancel catastrophically when the result is
    near zero -- an interface property, not a kernel error.  The
    informational fp16 real-affine replica lives in run_L2 (warn only).
    """
    return "unit" if dtype_str == "float16" else "op"


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_group_norm(x, weight, bias, num_groups, eps=1e-5):
    """PyTorch reference: ``F.group_norm(x, num_groups, weight, bias, eps)``.

    Port of the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_group_norm.py ``GroupNormTest.ref_program``).  Three tensor
    inputs (x / weight / bias): ``x`` is ``(N, C, *spatial)``,
    ``weight``/``bias`` are per-channel ``(C,)``.  fp32 opmath over the
    whole call and a single round-back to the input dtype.

    Independent of the NPU algorithm (no fp32-transit chain, no
    persistent structure, no vsqrt+vdiv composition).  Note: this golden
    maps the affine in fp32, whereas the op layer applies it after the
    kernel's dtype round-trip; the resulting double-rounding is an
    inherent property of the (unchanged) op-layer interface and is
    covered by the dtype tolerances.

    Runs on CPU and does not require an NPU device.
    """
    return F.group_norm(
        x.float(),
        num_groups,
        weight=weight.float(),
        bias=bias.float(),
        eps=eps,
    ).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _default_block_m(D):
    """UB-budget default block_m (DESIGN.md section 4.5 formula)."""
    return min(_BLOCK_M_CAP, max(1, _UB_MAX_ELEMS // D))


def _group_norm_kernel(M, D, eps, dtype):
    """Build the GroupNorm forward kernel factory (source signature).

    Returns a JIT callable taking a single ``block_m`` (``threads``
    removed, DESIGN.md section 3.3 note), which compiles and returns
    ``main``; ``main(x, weight, bias, y)`` returns ``y`` (``out_idx=[3]``,
    source contract).
    """
    if M < 1 or D < 1:
        raise ValueError(f"M and D must be positive, got M={M}, D={D}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )

    # R7: NPU has no row-length alignment requirement; D_padded == D and
    # pad_count == 0 (kept for structural parity with the source).
    D_padded = _align_up(D, ALIGNMENT)
    pad_count = D_padded - D
    del pad_count  # exact D: no padding contribution to correct for (O4)

    # O1 (DESIGN.md section 1.6.1): compile-time reciprocal, used as the
    # scalar src2 of T.vmul (division -> multiplication by fl(1/D)).
    D_inv = 1.0 / float(D)

    # R4: fp32 transit for both half-precision dtypes (bf16 has no Vector
    # arithmetic; fp16 transit matches the golden's fp32 opmath rounding
    # path, TRAP-fp16-opmath-golden).
    use_fp32_transit = dtype in ("float16", "bfloat16")
    block_m_default = _default_block_m(D)

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; measured 20 -> 40 on Ascend910B3).  Re-queried per
    # factory call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[3], target="npuir")
    def _func(block_m=block_m_default):
        # Host-side UB budget guard (DESIGN.md sections 3.3 and 9.2).
        # ValueError (not a bare assert) so the guard survives `python -O`.
        if block_m < 1:
            raise ValueError(f"block_m must be positive, got block_m={block_m}")
        if block_m * D > _UB_MAX_ELEMS:
            raise ValueError(
                f"block_m={block_m} exceeds the UB budget for D={D} "
                f"(max {_UB_MAX_ELEMS} resident elements at "
                f"{_UB_BYTES_PER_ELEM} B/elem multi-buffered, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5).  All quantities
        # are compile-time Python constants (fold to IntImm at trace time,
        # PL-1.5); the in-core serial bound is therefore static.
        num_logical = (M + block_m - 1) // block_m
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = (num_logical + num_kernels - 1) // num_kernels

        # Compile-time dtype dispatch outside the traced body (avoids the
        # TVM-script parser if-block variable-table scoping limitation).
        if use_fp32_transit:

            @T.prim_func
            def main(
                x: T.Tensor[(M, D_padded), dtype],
                # Interface-only row-broadcast affine (D_padded,).  This
                # migration path always receives the identity affine
                # (unit weight / zero bias) from the wrapper, so the
                # multiply-add is an IEEE bit-exact no-op and is omitted
                # (O5).  Per-channel affine is applied by the op layer.
                weight: T.Tensor[(D_padded,), dtype],
                bias: T.Tensor[(D_padded,), dtype],
                y: T.Tensor[(M, D_padded), dtype],  # out_idx=[3]
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (DESIGN.md section 4.3), allocated once
                    # per core and reused across serial tasks:
                    #   stage : x load -> rounded result (2 roles)
                    #   x_f32 : x fp32 -> d (kept, O3) -> d*rstd
                    #   sq_f32: d^2 -> sqrt(var) scratch (2 roles)
                    stage = T.alloc_shared((block_m, D_padded), dtype)
                    x_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    sq_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # row blocks).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            # Tail row count.  The v-ops below run on the
                            # full (block_m, D) buffers; garbage tail rows
                            # are row-isolated (reduce dim=1 / row
                            # broadcast) and never copied out.
                            real_m = T.min(block_m, M - off_m)

                            # --- 1. Load + upcast (R4) ---
                            # Dual explicit slices, dst zero-origin
                            # (TRAP-T-copy-region-semantics /
                            # TRAP-UB-dst-align recommended form).
                            T.copy(
                                x[off_m : off_m + real_m, 0:D],
                                stage[0:real_m, 0:D],
                            )
                            T.vcast(stage, x_f32, round_mode="rint")

                            # --- 2. Mean (O1: S1 * fl(1/D)) ---
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, D_inv, mean)

                            # --- 3. Centered two-pass variance ---
                            # d is kept for the epilogue (O3); exact D, no
                            # pad accounting (O4 identity).
                            T.vsub(x_f32, mean, x_f32)  # d = x - mean
                            T.vmul(x_f32, x_f32, sq_f32)  # sq = d^2
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, D_inv, var)
                            T.vadd(var, eps, var)
                            # rstd = 1/sqrt(var): vsqrt+vdiv composition
                            # (the raw T.vrsqrt instruction carries
                            # ~2.87e-3 relative error on npuir; R5).
                            T.vsqrt(var, s2)  # s2 reused: sqrt(var)
                            T.vdiv(s2, var, rstd)  # rstd = sqrt(var)/var

                            # --- 4. Normalise (unit kernel affine, O5) ---
                            T.vmul(x_f32, rstd, x_f32)  # d * rstd

                            # --- 5. Round back once + store valid rows ---
                            T.vcast(x_f32, stage, round_mode="rint")
                            T.copy(
                                stage[0:real_m, 0:D],
                                y[off_m : off_m + real_m, 0:D],
                            )

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(M, D_padded), dtype],
                weight: T.Tensor[(D_padded,), dtype],
                bias: T.Tensor[(D_padded,), dtype],
                y: T.Tensor[(M, D_padded), dtype],
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # fp32 path (DESIGN.md section 4.3): no transit
                    # buffers, same-dtype direct copies, no vcast.
                    x_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    sq_f32 = T.alloc_shared((block_m, D_padded), "float32")
                    s1 = T.alloc_shared((block_m, 1), "float32")
                    mean = T.alloc_shared((block_m, 1), "float32")
                    s2 = T.alloc_shared((block_m, 1), "float32")
                    var = T.alloc_shared((block_m, 1), "float32")
                    rstd = T.alloc_shared((block_m, 1), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)

                            T.copy(
                                x[off_m : off_m + real_m, 0:D],
                                x_f32[0:real_m, 0:D],
                            )
                            T.reduce_sum(x_f32, s1, dim=1)
                            T.vmul(s1, D_inv, mean)
                            T.vsub(x_f32, mean, x_f32)
                            T.vmul(x_f32, x_f32, sq_f32)
                            T.reduce_sum(sq_f32, s2, dim=1)
                            T.vmul(s2, D_inv, var)
                            T.vadd(var, eps, var)
                            T.vsqrt(var, s2)  # s2 reused: sqrt(var)
                            T.vdiv(s2, var, rstd)  # rstd = sqrt(var)/var
                            T.vmul(x_f32, rstd, x_f32)
                            # Direct fp32 write-back, valid rows only.
                            T.copy(
                                x_f32[0:real_m, 0:D],
                                y[off_m : off_m + real_m, 0:D],
                            )

        return main

    return _func


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------

_FACTORY_CACHE = {}


def _get_factory(M, D, eps, dtype_str):
    """Cache factory instances (mirrors the wrapper's one-factory-per-op
    usage; the jit wrapper then caches compiled mains per block_m)."""
    key = (M, D, eps, dtype_str)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _group_norm_kernel(M, D, eps, dtype_str)
    return _FACTORY_CACHE[key]


# Per-dtype worst max_diff across the whole run (for the final summary).
_STATS_MAX_DIFF = {}


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware: positions where both are NaN count as equal when
    equal_nan is set; equal infinities count as equal.
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _resolve_spec(x_shape, num_groups):
    """(N, C, *spatial) + num_groups -> derived (M, D, spatial, cpg)."""
    N, C = int(x_shape[0]), int(x_shape[1])
    spatial = tuple(int(s) for s in x_shape[2:])
    spatial_size = int(math.prod(spatial)) if spatial else 1
    if C % num_groups != 0:
        raise ValueError(f"C={C} must be divisible by num_groups={num_groups}")
    cpg = C // num_groups
    D = cpg * spatial_size
    M = N * num_groups
    return N, C, spatial, spatial_size, cpg, D, M


def _run_case(
    dtype_str,
    x_shape,
    g,
    block_m=None,
    tag="L0",
    x=None,
    weight=None,
    bias=None,
    eps=1e-5,
    affine="op",
    equal_nan=False,
):
    """Run one 3-input GroupNorm case and compare against the golden.

    ``affine`` selects the per-channel ``(C,)`` affine used:
      "op"   : randn weight/bias, applied in the input dtype exactly like
               ``GroupNormFwdOp.forward`` (``y * w.reshape(...) + b``).
      "unit" : unit weight / zero bias; the op-layer affine is an exact
               identity, so the kernel output is compared directly
               against pure normalisation (DESIGN.md section 8.1: "the
               kernel-level output carries no affine, so compare with
               unit weight / bias").

    The flow still replicates ``GroupNormFwdOp.forward``: x
    (N,C,*spatial) -> contiguous -> view (M, D) -> kernel with the
    **unit row-broadcast** ``(D,)`` affine -> view back -> per-channel
    affine outside the kernel -> golden compare.  Per-channel ``(C,)``
    weights are never fed to the kernel.

    ``x`` / ``weight`` / ``bias`` may be pre-built CPU tensors
    (special-value cases); otherwise randn inputs are generated on CPU,
    cast, and moved to NPU.

    Raises AssertionError on tolerance violation.
    """
    if affine not in ("op", "unit"):
        raise ValueError(f"affine must be 'op' or 'unit', got {affine!r}")
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    N, C, spatial, spatial_size, cpg, D, M = _resolve_spec(x_shape, g)

    if x is None:
        x = torch.randn(*x_shape, dtype=torch.float32).to(torch_dtype)
    else:
        assert tuple(x.shape) == tuple(x_shape), f"x.shape {tuple(x.shape)} != {tuple(x_shape)}"
        assert x.dtype == torch_dtype, f"x.dtype {x.dtype} != {torch_dtype}"
    if weight is None:
        if affine == "unit":
            weight = torch.ones(C, dtype=torch_dtype)
        else:
            weight = torch.randn(C, dtype=torch.float32).to(torch_dtype)
    if bias is None:
        if affine == "unit":
            bias = torch.zeros(C, dtype=torch_dtype)
        else:
            bias = torch.randn(C, dtype=torch.float32).to(torch_dtype)

    if block_m is None:
        block_m = _default_block_m(D)

    x = x.contiguous()

    # Kernel-level row-broadcast affine: always the identity on this path
    # (per-channel (C,) weights must never be fed to the kernel).
    kernel_weight = torch.ones(D, dtype=torch_dtype).npu()
    kernel_bias = torch.zeros(D, dtype=torch_dtype).npu()

    x_2d = x.reshape(M, D).npu()
    kernel = _get_factory(M, D, eps, dtype_str)(block_m)
    y_2d = kernel(x_2d, kernel_weight, kernel_bias)

    # Op-layer reshape back + per-channel affine (same dtype / device as
    # the op layer).  The kernel itself never sees the per-channel affine.
    y = y_2d.reshape(N, g, cpg, *spatial).reshape(x_shape)
    affine_shape = (1, C) + (1,) * len(spatial)
    y = y * weight.npu().reshape(affine_shape) + bias.npu().reshape(affine_shape)

    ref = golden_group_norm(x, weight, bias, g, eps)
    y_cpu = y.cpu()

    max_diff, violations = _compare(y_cpu, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
        best = _STATS_MAX_DIFF.get(dtype_str, 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(y_cpu, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: x_shape={tuple(x_shape)} g={g} dtype={dtype_str} "
            f"(M,D)=({M},{D}) block_m={block_m} max_diff={max_diff:.3e} "
            f"violations={violations}/{y_cpu.numel()}"
        )
        raise
    print(
        f"[{tag}] PASS: x_shape={tuple(x_shape)} g={g} dtype={dtype_str} "
        f"(M,D)=({M},{D}) block_m={block_m} max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other
    exception is recorded as a runtime error (both block the gate).
    """
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, 17 items).

    Per DESIGN.md section 8.1 the kernel-level cases compare against the
    golden with **unit** per-channel affine (the kernel's row-broadcast
    affine is the identity on this path), while fp32/bf16 cases also carry
    a real random per-channel affine through the op-layer replica.  fp16
    real per-channel affine is deliberately NOT a blocking L0/L1 case: the
    op layer applies it *after* the kernel's dtype round-trip, so its fp16
    cancellation error is bounded by the square of the affine magnitude and
    is an interface property of the (unchanged) op layer, not of this
    kernel (see run_L2 for the informational fp16 real-affine case).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # (case_id, x_shape, num_groups, dtype, block_m, kwargs)
    cases = [
        # manifest smoke-g8 (2,32,8,8) g=8 -> (M, D) = (16, 256)
        ("L0-1", (2, 32, 8, 8), 8, "float32", 8, {}),
        ("L0-2", (2, 32, 8, 8), 8, "float16", 8, {"affine": "unit"}),
        ("L0-3", (2, 32, 8, 8), 8, "bfloat16", 8, {}),
        # manifest image-g32 (8,128,32,32) g=32 -> (256, 4096)
        ("L0-4", (8, 128, 32, 32), 32, "float16", 2, {"affine": "unit"}),
        ("L0-5", (8, 128, 32, 32), 32, "bfloat16", 2, {}),
        ("L0-6", (8, 128, 32, 32), 32, "float32", 2, {}),
        # manifest wider-channel-g32 (4,256,28,28) g=32 -> (128, 6272)
        ("L0-7", (4, 256, 28, 28), 32, "float16", 1, {"affine": "unit"}),
        # manifest tail-spatial-g16 (4,128,30,30) g=16 -> (64, 7200)
        ("L0-8", (4, 128, 30, 30), 16, "float16", 1, {"affine": "unit"}),
        # row tail A: 17 = 8*2 + 1 -> (M, D) = (17, 1152)
        ("L0-9", (17, 1, 1152), 1, "float16", 8, {"affine": "unit"}),
        # row tail B: 257 = 2*128 + 1, representative D
        ("L0-10", (257, 1, 4096), 1, "float16", 2, {"affine": "unit"}),
        # row tail + bf16, block_m = 1 (no row tail: M % 1 == 0)
        ("L0-11", (65, 1, 7200), 1, "bfloat16", 1, {}),
        # tiny odd shape: D = 13 (partial vector lane), M < block_m
        ("L0-12", (5, 1, 13), 1, "float16", 8, {"affine": "unit"}),
        # non-power-of-two cpg + non-aligned spatial (7,7): (16, 196)
        ("L0-13", (2, 32, 7, 7), 8, "float16", 8, {"affine": "unit"}),
        # 3D spatial (2,16,4,4,4) g=4 -> (8, 256)
        ("L0-14", (2, 16, 4, 4, 4), 4, "float16", 8, {"affine": "unit"}),
        # 1D spatial (2,32,16) g=8 -> (16, 64)
        ("L0-15", (2, 32, 16), 8, "float16", 8, {"affine": "unit"}),
        # large-mean row statistics in fp32 (mean ~3.7, unit var scale):
        # exercises the mean/centring path without the tiny-variance
        # conditioning corner (that corner is a Boundary case).
        (
            "L0-16",
            (2, 32, 8, 8),
            8,
            "float32",
            8,
            {
                "x": (3.7 + torch.randn(2, 32, 8, 8)).to(torch.float32),
                "affine": "unit",
            },
        ),
        # op-level end-to-end, real per-channel (3-input) affine + reshape
        # (migrated fixture shape (4,16,(4,4)), g=4).
        ("L0-17", (4, 16, 4, 4), 4, "float32", 8, {}),
    ]

    for case_id, x_shape, g, dtype_str, block_m, kw in cases:
        _try_case(
            failures,
            case_id,
            dtype_str=dtype_str,
            x_shape=x_shape,
            g=g,
            block_m=block_m,
            tag=case_id,
            **kw,
        )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print(f"[L0] ALL PASS ({len(cases)} items)")
    return failures


def _run_L1_contract(failures):
    """L1 contract checks (factory guards, no precision comparison)."""
    # UB budget guard rejects over-budget block_m values.
    for M, D, dtype_str, bm in (
        (64, 4096, "float32", 3),  # 3*4096 = 12288 > 9216
        (64, 4096, "float16", 3),  # 12288 > 9216
        (64, 6272, "float16", 2),  # 12544 > 9216
    ):
        try:
            _group_norm_kernel(M, D, 1e-5, dtype_str)(bm)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} bm={bm} D={D} ({exc})")
        else:
            failures.append(
                (
                    f"L1/UB-guard/{dtype_str}/bm={bm}/D={D}",
                    AssertionError(f"{dtype_str} bm={bm} D={D} not rejected"),
                )
            )

    # Out-of-domain D (beyond the residency bound) is intercepted by the
    # same guard through the default block_m (DESIGN.md section 9.3).  The
    # guard fires when _func(block_m) is invoked, not at factory build.
    try:
        _group_norm_kernel(4, 20000, 1e-5, "float16")()
    except ValueError as exc:
        print(f"[L1] PASS: domain guard rejects D=20000 fp16 ({exc})")
    else:
        failures.append(("L1/domain-guard-D20000", AssertionError("D=20000 fp16 not rejected")))

    # Non-positive M / D are rejected.
    for bad_m, bad_d in ((0, 1152), (64, 0)):
        try:
            _group_norm_kernel(bad_m, bad_d, 1e-5, "float16")
        except ValueError:
            print(f"[L1] PASS: M/D guard rejects M={bad_m}, D={bad_d}")
        else:
            failures.append(
                (
                    f"L1/md-guard/{bad_m}x{bad_d}",
                    AssertionError(f"M={bad_m}, D={bad_d} not rejected"),
                )
            )

    # Unsupported dtype is rejected.
    try:
        _group_norm_kernel(64, 1152, 1e-5, "int32")
    except ValueError:
        print("[L1] PASS: dtype guard rejects int32")
    else:
        failures.append(("L1/dtype-guard", AssertionError("int32 not rejected")))

    # ALIGNMENT = 1 keeps the wrapper contract padding-free (pad == 0).
    assert ALIGNMENT == 1, f"NPU ALIGNMENT must be 1, got {ALIGNMENT}"
    assert _align_up(13, ALIGNMENT) == 13
    print("[L1] PASS: ALIGNMENT=1 (D_padded == D, wrapper pad == 0)")


def run_L1():
    """L1: full functional coverage (dtype x shape x block_m + contracts).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # dtype x shape grid: aligned / M-tail / unaligned spatial / tiny.
    for dtype_str in ("float32", "float16", "bfloat16"):
        for x_shape, g, bm in (
            ((8, 64, 8, 8), 8, 4),  # (64, 512) aligned
            ((2, 32, 8, 8), 8, 8),  # (16, 256), M < block_m
            ((2, 64, 5, 5), 16, 2),  # (32, 100), unaligned spatial
            ((2, 32, 16), 8, 2),  # 1D spatial -> (16, 64)
        ):
            _try_case(
                failures,
                f"L1/{dtype_str}/{x_shape}/g={g}/bm={bm}",
                dtype_str=dtype_str,
                x_shape=x_shape,
                g=g,
                block_m=bm,
                affine=_affine_for(dtype_str),
                tag="L1",
            )

    # 3-input coverage from the migrated test fixture: 1D/3D spatial,
    # non-power-of-two cpg (C=30, g=5), non-aligned spatial (7,7).
    for x_shape, g, dtype_str in (
        ((2, 32, 16), 8, "float16"),  # 1D spatial
        ((2, 16, 4, 4, 4), 4, "float16"),  # 3D spatial
        ((2, 30, 4, 4), 5, "float16"),  # cpg = 6 (non-power-of-two)
        ((2, 30, 4, 4), 5, "float32"),
        ((2, 30, 4, 4), 5, "bfloat16"),
        ((2, 32, 7, 7), 8, "bfloat16"),  # non-aligned spatial, bf16
        ((2, 32, 7, 7), 8, "float32"),  # non-aligned spatial, real affine
        ((2, 32, 4, 4), 1, "float16"),  # g = 1
        ((2, 32, 4, 4), 32, "float16"),  # g = C
    ):
        _try_case(
            failures,
            f"L1/3in/{x_shape}/g={g}/{dtype_str}",
            dtype_str=dtype_str,
            x_shape=x_shape,
            g=g,
            affine=_affine_for(dtype_str),
            tag="L1",
        )

    # block_m sweep on one shape (wrapper config coverage).
    for bm in (1, 2, 4, 8):
        _try_case(
            failures,
            f"L1/fp16/96x1152/bm={bm}",
            dtype_str="float16",
            x_shape=(96, 1, 1152),
            g=1,
            block_m=bm,
            affine="unit",
            tag="L1",
        )

    # Persistent core-count switch boundary (num_logical around 40 Vector
    # cores; 397 = 49*8 + 5 also carries an M tail).
    for M, logical in ((312, 39), (320, 40), (328, 41), (397, 50)):
        _try_case(
            failures,
            f"L1/persistent/num_logical={logical}",
            dtype_str="float16",
            x_shape=(M, 1, 1152),
            g=1,
            block_m=8,
            affine="unit",
            tag="L1",
        )

    # Factory-default block_m path (UB-budget table value).
    _try_case(
        failures,
        "L1/fp16/50x3000/default-bm",
        dtype_str="float16",
        x_shape=(50, 1, 3000),
        g=1,
        affine="unit",
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/50x3000/default-bm",
        dtype_str="float32",
        x_shape=(50, 1, 3000),
        g=1,
        affine="op",
        tag="L1",
    )

    # Non-default eps (exercises the eps plumbing: var + eps placement).
    _try_case(
        failures,
        "L1/fp16/eps=1e-2",
        dtype_str="float16",
        x_shape=(2, 32, 8, 8),
        g=8,
        block_m=8,
        eps=1e-2,
        affine="unit",
        tag="L1",
    )
    _try_case(
        failures,
        "L1/bf16/eps=1e-3",
        dtype_str="bfloat16",
        x_shape=(2, 32, 8, 8),
        g=8,
        block_m=8,
        eps=1e-3,
        affine="op",
        tag="L1",
    )

    # Unit per-channel affine in fp32 (kernel validated directly against
    # pure normalisation, op-layer affine an exact identity).
    _try_case(
        failures,
        "L1/kernel-only-unit-affine/fp32",
        dtype_str="float32",
        x_shape=(4, 64, 8, 8),
        g=8,
        block_m=4,
        affine="unit",
        tag="L1",
    )
    _try_case(
        failures,
        "L1/kernel-only-unit-affine/bf16",
        dtype_str="bfloat16",
        x_shape=(4, 64, 8, 8),
        g=8,
        block_m=4,
        affine="unit",
        tag="L1",
    )

    # Non-contiguous input (the op layer calls ``x.contiguous()`` first).
    try:
        x_full = torch.randn(2, 64, 8, 8, dtype=torch.float32).to(torch.float16)
        x_nc = x_full[:, :32]  # non-contiguous slice (stride != contiguous)
        w = torch.ones(32, dtype=torch.float16)
        b = torch.zeros(32, dtype=torch.float16)
        _run_case(
            dtype_str="float16",
            x_shape=(2, 32, 8, 8),
            g=8,
            block_m=8,
            affine="unit",
            tag="L1",
            x=x_nc,
            weight=w,
            bias=b,
        )
        print("[L1] PASS: non-contiguous input via op-layer contiguous()")
    except AssertionError as exc:
        failures.append(("L1/non-contiguous", exc))
        print(f"[collected] L1/non-contiguous FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/non-contiguous", exc))
        print(f"[collected] L1/non-contiguous RUNTIME-ERROR: {exc!r}")

    # E2 factory contract: one factory callable serving two block_m values
    # (the jit wrapper caches the compiled main per block_m).
    try:
        dtype_str = "float16"
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        M, D, g = 96, 1152, 1
        x = torch.randn(96, 1, 1152, dtype=torch.float32).to(torch_dtype)
        w = torch.ones(1, dtype=torch_dtype)
        b = torch.zeros(1, dtype=torch_dtype)
        ref = golden_group_norm(x, w, b, g, 1e-5)
        fn = _get_factory(M, D, 1e-5, dtype_str)
        kw = torch.ones(D, dtype=torch_dtype).npu()
        kb = torch.zeros(D, dtype=torch_dtype).npu()
        x_2d = x.reshape(M, D).npu()
        for bm in (8, 4):
            out = fn(bm)(x_2d, kw, kb).cpu().reshape(96, 1, 1152)
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L1] PASS: E2 factory(block_m) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # Factory contract guards (no NPU run).
    _run_L1_contract(failures)

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes and informational interface cases (warn only)."""
    cases = [
        ((1, 1, 1), 1, "float16", 1),  # minimal
        ((2, 3, 3), 1, "bfloat16", 1),  # tiny odd
        ((1, 1, 4096), 1, "float32", None),  # single row, default bm
        ((3, 1, 8192), 1, "float16", 1),  # D beyond the manifest max
        ((5000, 1, 64), 1, "float16", 8),  # large M, small D (deep serial)
        ((1, 1, 9216), 1, "bfloat16", 1),  # single row, at the UB element cap
        ((1, 1, 9216), 1, "float32", 1),
    ]
    for x_shape, g, dtype_str, bm in cases:
        try:
            _run_case(
                dtype_str=dtype_str,
                x_shape=x_shape,
                g=g,
                block_m=bm,
                affine=_affine_for(dtype_str),
                tag="L2",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): x_shape={x_shape} {dtype_str}: {e}")

    # Informational (non-blocking): the EXACT op-layer replica for fp16,
    # i.e. the real randn per-channel affine applied in fp16 after the
    # kernel's fp16 round-trip.  This reproduces the op layer bit-for-bit
    # but its fp16 cancellation error is an interface property (bounded by
    # |w| * ulp(normalised)), not a kernel error -- the kernel itself is
    # validated by the unit-affine L0/L1 gate.  Expected to warn.
    for x_shape, g in (((2, 32, 8, 8), 8), ((4, 16, 4, 4), 4)):
        try:
            _run_case(
                dtype_str="float16",
                x_shape=x_shape,
                g=g,
                affine="op",
                tag="L2/fp16-op-replica",
            )
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): fp16 op replica {x_shape}: {e}")

    # Informational (non-blocking): large mean / tiny variance in fp32.
    # The comparison error is dominated by the fp32 representation of the
    # large mean (~4.8e-7 at 3.7) amplified by rstd ~ 1e2, i.e. it
    # exceeds atol 1e-5 for ANY implementation whose reduction order
    # differs from torch's.  Kept as a documented conditioning corner.
    try:
        _run_case(
            dtype_str="float32",
            x_shape=(2, 32, 8, 8),
            g=8,
            affine="unit",
            tag="L2/large-mean-tiny-var",
            x=(3.7 + torch.randn(2, 32, 8, 8) * 1e-2).to(torch.float32),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[L2] WARN (Record without blocking): large-mean-tiny-var fp32: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    shape = (8, 16, 8, 8)  # (N, C, *spatial)
    g = 8

    def _mk(dtype=torch.float16, unit=True):
        C = shape[1]
        ww = torch.ones(C, dtype=dtype) if unit else torch.randn(C).to(dtype)
        bb = torch.zeros(C, dtype=dtype) if unit else torch.randn(C).to(dtype)
        return ww, bb

    # Zeros: var = 0 -> rstd = 1/sqrt(eps); pure kernel path (unit affine).
    try:
        ww, bb = _mk(unit=True)
        _run_case(
            dtype_str="float16",
            x_shape=shape,
            g=g,
            tag="Boundary-zeros",
            x=torch.zeros(*shape, dtype=torch.float16),
            weight=ww,
            bias=bb,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # Constant rows (var = 0 as well, non-zero mean).
    try:
        ww, bb = _mk(unit=True)
        _run_case(
            dtype_str="float32",
            x_shape=shape,
            g=g,
            tag="Boundary-const-rows",
            x=torch.full(shape, 3.7, dtype=torch.float32),
            weight=ww,
            bias=bb,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): const-rows: {e}")

    # Large fp16 magnitudes (x100 scale).
    try:
        ww, bb = _mk(dtype=torch.float16)
        _run_case(
            dtype_str="float16",
            x_shape=shape,
            g=g,
            tag="Boundary-large",
            x=(torch.randn(*shape) * 100.0).to(torch.float16),
            weight=ww,
            bias=bb,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # Subnormal fp16 magnitudes (~1e-5 scale; exact fp32 upcast).
    try:
        ww, bb = _mk(dtype=torch.float16)
        _run_case(
            dtype_str="float16",
            x_shape=shape,
            g=g,
            tag="Boundary-subnormal",
            x=(torch.randn(*shape) * 1e-5).to(torch.float16),
            weight=ww,
            bias=bb,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # Extreme per-channel magnitude (fp16 max 65504; saturation corners
    # compare equal whenever both sides saturate to the same inf).
    try:
        ww, bb = _mk(unit=True)
        xx = torch.randn(*shape).to(torch.float16)
        _run_case(
            dtype_str="float16",
            x_shape=shape,
            g=g,
            tag="Boundary-extreme",
            x=xx,
            weight=ww,
            bias=bb,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): extreme: {e}")

    # inf/NaN natural propagation (DESIGN.md section 0.1: no special
    # handling; a full-inf row yields NaN output on both sides).
    try:
        xx = torch.randn(*shape, dtype=torch.float32)
        xx[0, :, :, :] = float("inf")  # full-inf rows -> NaN rows
        xx[1, ::4] = float("nan")  # scattered NaN -> NaN rows
        xx[2, ::4] = float("inf")  # mixed inf rows -> NaN rows
        ww, bb = _mk(unit=True)
        _run_case(
            dtype_str="float16",
            x_shape=shape,
            g=g,
            tag="Boundary-inf-nan",
            x=xx.to(torch.float16),
            weight=ww,
            bias=bb,
            equal_nan=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf-nan: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
