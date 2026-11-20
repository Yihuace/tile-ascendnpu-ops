# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Floor-divide forward kernel (NPU, Developer mode).

Migrated from the GPU TileOPs operator ``floor_divide_fwd_kernel``
(``_make_binary_explicit`` factory specialised with
``op_func(a, b) = T.Cast(a.dtype, T.floor(T.cast(a, "float32") /
T.cast(b, "float32")))``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = floor(a[off_a(i)] / b[off_b(i)])
on two already-flattened contiguous ``(a_numel,)`` / ``(b_numel,)`` tensors
of the same float dtype (float16 / bfloat16 / float32); the output is the
flat ``(N_total,)`` view of ``broadcast_shapes(a_shape, b_shape)``.
``off_a`` / ``off_b`` are the broadcast index maps; for same-shape inputs
they are the identity.  Division and floor are evaluated in the fp32
domain, then the single result is cast back to the native dtype.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element unrolled divmod chain (used only on the
        broadcast path) is replaced by an outer-row scalar base (one
        unrolled mixed-radix divmod per task) plus innermost-dimension
        vectorisation.  Rows whose last-dim stride is 0 (a scalar within
        the row) are materialised with ``T.vbrc``.
    R2: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial`` grid-stride loop and an if-guard,
        replacing the CUDA grid x block parallel structure.  Element
        parallelism is handled by the Vector unit.
    R4: there is no usable one-shot floor-divide primitive (see the
        runtime traps below); the elementwise work is the two-instruction
        chain ``T.vdiv`` then ``T.vcast(round_mode="floor")``, executed in
        the fp32 domain.  fp16/bf16 are promoted losslessly with
        ``T.vcast(round_mode="rint")`` and the result is rounded back once.

Implementation notes / toolchain traps (DESIGN.md section 9; probed and
registered in capability-gaps):
    - ``T.vfloordiv`` (``npuir_floordiv``) and ``T.vfloor`` (``npuir_floor``)
      compile successfully but HANG the Vector core at runtime on this
      toolchain (fp32/fp16/bf16 same-shape, sliced and ``alloc_shared``
      forms all hang; CG-2026-0023).  They must NOT be used here; the
      two-instruction ``vdiv`` + ``vcast(floor)`` chain is the only
      verified floor-divide path.
    - ``T.vcast`` with an identity-dtype cast (f32->f32, any rounding mode)
      produces NaN on this toolchain even though
      docs/Tilelang.language/数据类型转换操作/T.vcast.md lists it as
      supported (CG-2026-0024).  The fp32 path therefore does no
      promote/demote cast at all -- the dtype branches below skip the
      casts statically instead of running a uniform cast chain.
    - The fp32 transit for fp16/bf16 is a correctness requirement, not an
      optimisation: computing in the native half domain rounds the
      quotient before the floor (e.g. ``2.999..`` -> ``3.0`` ->
      ``floor=3``), diverging from the golden.  DESIGN.md E3 measured 66
      fp16 / 430 bf16 divergences for the no-transit variant.
    - The kernel body's ``block_size`` must be the same value used to
      derive the task decomposition (``chunks`` / ``num_logical`` /
      ``num_kernels`` / ``num_local_tasks``).  Those are computed *inside*
      the jit function (not at factory-build time) so the returned
      callable stays consistent for every block_size it is invoked with.

Factory contract (frozen; see examples/TileOPs/tileops/kernels/elementwise/
floor_divide/floor_divide.py):
    floor_divide_fwd_kernel(N_total, dtype, coalesced_shape, a_strides,
                            b_strides, a_numel, b_numel,
                            output_dtype=None, block_size=4096)
        -> kernel(block_size) -> main(a, b) -> y   (``out_idx=[2]``)

Run the embedded hierarchical tests:
    python floor_divide_fwd_kernel.py --level L0
    python floor_divide_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per task, no manual
# sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import math
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "ASCEND_MODE",
    "coalesce_broadcast_dims",
    "floor_divide_fwd_kernel",
    "golden_floor_divide",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

# dtypes handled natively by the kernel (float-only; integers/bool are
# rejected by the Op layer, matching the source _FLOAT_DTYPES contract).
_ALLOWED_DTYPES = ("float16", "bfloat16", "float32")

_DEFAULT_BLOCK = 4096

# Bytes of full-size UB buffer per element, worst case (DESIGN.md 4.5):
#   float32        : a32 + b32 + q + yf                      = 4 x 4 = 16
#   float16/bf16   : a_in + b_in + a32 + b32 + q + yf + y_out = 22
_UB_BYTES_PER_ELEM = {
    "float32": 4 * 4,
    "float16": 3 * 2 + 4 * 4,
    "bfloat16": 3 * 2 + 4 * 4,
}

# Upper bound on block_size from the UB budget (DESIGN.md section 4.5,
# conservative x1.7 inflation; fp32 ~7228, fp16/bf16 ~5257).
_UB_MAX_BLOCK = {
    dt: int(_UB_CAPACITY_BYTES // (_UB_BYTES_PER_ELEM[dt] * _UB_INFLATION))
    for dt in _ALLOWED_DTYPES
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_floor_divide(a, b):
    """Floor-divide reference (source-operator semantics, direct broadcast).

    ``torch.floor(a.float() / b.float()).to(a.dtype)`` -- the source-repo
    ``test_floor_divide.py`` ``ref_program`` (line 41), which mirrors the GPU
    op body ``T.Cast(a.dtype, T.floor(T.cast(a, "float32") /
    T.cast(b, "float32")))``: division and floor are evaluated in the fp32
    domain, then the single result is cast back.  Inputs keep their original
    broadcast shapes; the output shape is ``broadcast_shapes(a.shape,
    b.shape)`` and the dtype is ``same_as(a)``.  Independent of the NPU
    algorithm (no row/inner indexing, no vdiv/vcast, no persistent
    structure) and runs on CPU without an NPU device.

    NOTE (Stage 3, D-class finding): plain ``torch.floor_divide(a, b)`` is
    deliberately NOT used as the reference.  On CPU it floors a
    higher-precision quotient, so it disagrees with the source's own fp32
    double-rounding semantics at boundary points.  Measured on a
    (4096, 4096) / (1, 4096) fp32 row-broadcast: ``torch.floor_divide``
    differs from the explicit fp32 formula at 3 / 16.78M points -- e.g.
    a=1.6938038 b=0.11292025: the fp32 quotient rounds to exactly 15.0
    (explicit floor = 15) while ``floor_divide`` returns 14.  The frozen
    operator semantics is the fp32-domain one, so the explicit formula is
    the golden.
    """
    return torch.floor(a.detach().cpu().float() / b.detach().cpu().float()).to(a.dtype)


# ---------------------------------------------------------------------------
# Broadcast lowering (verbatim from the source operator's helper)
# ---------------------------------------------------------------------------
def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod
    operations inside the kernel body.  This is a lowering decision, not
    part of the op contract: the caller hands the factory two operand
    shapes and the kernel picks its own index representation.

    Returns ``(out_shape, coalesced_shape, a_strides, b_strides)`` where
    strides use 0 for broadcast dimensions.
    """
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        return [
            0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i]
            for i in range(ndim)
        ]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


# ---------------------------------------------------------------------------
# Outer-row scalar base (compile-time unrolled mixed-radix divmod)
# ---------------------------------------------------------------------------
def _row_base(row, outer_shape, strides):
    """Build the outer-row base offset expression for one operand.

    ``row`` is the runtime row index; ``outer_shape`` and ``strides`` are
    Python constants, so the loop unrolls at trace time into a chain of
    ``T.floordiv`` / ``T.floormod`` ops (one divmod per outer dimension,
    executed once per task rather than once per element -- DESIGN.md R1).
    Returns the Python int 0 when there is no outer dimension (k == 1).
    """
    base = 0
    rem = row
    n_outer = len(outer_shape)
    for j in range(n_outer):
        div = 1
        for t in range(j + 1, n_outer):
            div *= outer_shape[t]
        coord = T.floordiv(rem, div)
        rem = T.floormod(rem, div)
        base = base + coord * strides[j]
    return base


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def _normalize_dtype(dtype):
    """Return the TileLang dtype string, rejecting unsupported dtypes.

    The source operator is float-only (``_FLOAT_DTYPES``); integer and bool
    dtypes are rejected at the Op layer (test_floor_divide_rejects_integer_dtype).
    """
    if isinstance(dtype, torch.dtype):
        name = str(dtype).replace("torch.", "")
    else:
        name = str(dtype)
    if name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; floor_divide is float-only, "
            f"expected one of {_ALLOWED_DTYPES}"
        )
    return name


def floor_divide_fwd_kernel(
    N_total,
    dtype,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    block_size=_DEFAULT_BLOCK,
):
    """Build the floor-divide kernel factory.

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(a, b)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).
    """
    dtype = _normalize_dtype(dtype)
    out_dtype = _normalize_dtype(output_dtype) if output_dtype is not None else dtype
    coalesced_shape = tuple(int(x) for x in coalesced_shape)
    a_strides = tuple(int(x) for x in a_strides)
    b_strides = tuple(int(x) for x in b_strides)
    N_total = int(N_total)
    a_numel = int(a_numel)
    b_numel = int(b_numel)

    if N_total <= 0:
        raise ValueError(f"N_total must be a positive int, got {N_total}")
    if len(coalesced_shape) == 0 or len(a_strides) != len(coalesced_shape) or len(
        b_strides
    ) != len(coalesced_shape):
        raise ValueError("coalesced_shape / a_strides / b_strides rank mismatch")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op).  Re-queried per factory call so other deployments
    # adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    L = coalesced_shape[-1]
    outer = list(coalesced_shape[:-1])
    R = math.prod(outer) if outer else 1
    sa, sb = a_strides[-1], b_strides[-1]
    if sa not in (0, 1) or sb not in (0, 1):
        raise ValueError(f"innermost strides must be 0 or 1, got sa={sa}, sb={sb}")

    # Static dtype branch: fp16/bf16 need the fp32 transit (R4); the fp32
    # path must skip every identity cast (CG-2026-0024).  Resolved at trace
    # time so no runtime branch is emitted.
    needs_upcast = dtype != "float32"
    needs_downcast = out_dtype != "float32"
    max_block = _UB_MAX_BLOCK[dtype]

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        # Host-side UB budget guard (DESIGN.md section 4.5).
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {dtype} "
                f"(max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # block_size-dependent task decomposition, evaluated here (not at
        # factory time) so the compiled kernel always matches the block_size
        # it is invoked with -- the returned callable is cached per
        # block_size (wrapper K11 contract).  block_size folds to a
        # compile-time constant.
        chunks = T.ceildiv(L, block_size)
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), dtype),
            b: T.Tensor((b_numel,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # fp32 运算域缓冲：整 buffer 一条向量指令覆盖。
                a32 = T.alloc_ub((block_size,), "float32")
                b32 = T.alloc_ub((block_size,), "float32")
                q = T.alloc_ub((block_size,), "float32")
                yf = T.alloc_ub((block_size,), "float32")
                # 半精度中转缓冲：仅 fp16/bf16 使用（编译期分支，fp32 上被
                # DCE 消除）。注意：解析器会解析所有分支体，buffer 必须无条件
                # 声明，因此这里全部无条件 alloc_ub。
                a_in = T.alloc_ub((block_size,), dtype)
                b_in = T.alloc_ub((block_size,), dtype)
                y_out = T.alloc_ub((block_size,), out_dtype)
                # 标量广播源：仅当某 operand 末维 stride==0（该 operand 在行内
                # 是标量）时使用，先单元素搬入再 T.vbrc 物化成 (block_size,)。
                a1 = T.alloc_ub((1,), dtype)
                b1 = T.alloc_ub((1,), dtype)
                a1f = T.alloc_ub((1,), "float32")
                b1f = T.alloc_ub((1,), "float32")

                # persistent 核内串行循环：边界为编译期静态值（num_local_tasks
                # 由 num_logical/num_kernels 常量除得），grid-stride 映射
                # task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # 行 base 标量：每 task 仅一次 unrolled divmod（非每元素，
                        # 较源逐元素 divmod 降 ~L 倍），末维仅在 stride==1 时
                        # 贡献 inner_off。
                        a_base = _row_base(row, outer, a_strides) + (
                            inner_off if sa == 1 else 0
                        )
                        b_base = _row_base(row, outer, b_strides) + (
                            inner_off if sb == 1 else 0
                        )
                        tail = T.min(block_size, L - inner_off)

                        # --- 物化 a 到 fp32 域 (block_size,) ---
                        if sa == 1:
                            # a 末维连续：连续段搬入。
                            if needs_upcast:
                                T.copy(a[a_base : a_base + tail], a_in[0:tail])
                                T.vcast(a_in, a32, round_mode="rint")
                            else:
                                # fp32 输入直接搬入 a32，禁止 f32->f32 恒等 cast
                                # （CG-2026-0024，会产生 NaN）。
                                T.copy(a[a_base : a_base + tail], a32[0:tail])
                        else:
                            # a 行内标量：单元素搬入 + vbrc 广播到整 buffer。
                            T.copy(a[a_base : a_base + 1], a1[0:1])
                            if needs_upcast:
                                T.vcast(a1, a1f, round_mode="rint")
                                T.vbrc(a1f, a32)
                            else:
                                T.vbrc(a1, a32)

                        # --- 物化 b 到 fp32 域 (block_size,)（同上）---
                        if sb == 1:
                            if needs_upcast:
                                T.copy(b[b_base : b_base + tail], b_in[0:tail])
                                T.vcast(b_in, b32, round_mode="rint")
                            else:
                                T.copy(b[b_base : b_base + tail], b32[0:tail])
                        else:
                            T.copy(b[b_base : b_base + 1], b1[0:1])
                            if needs_upcast:
                                T.vcast(b1, b1f, round_mode="rint")
                                T.vbrc(b1f, b32)
                            else:
                                T.vbrc(b1, b32)

                        # --- 除法 + floor + 落盘 ---
                        # 禁用 T.vfloordiv / T.vfloor（编译成功但运行期挂死，
                        # CG-2026-0023）：唯一可用链是 vdiv + vcast(floor)，
                        # 且必须在 fp32 域（半精度商先舍入再 floor 会偏）。
                        T.vdiv(a32, b32, q)
                        T.vcast(q, yf, round_mode="floor")
                        flat0 = row * L + inner_off
                        if needs_downcast:
                            T.vcast(yf, y_out, round_mode="rint")
                            # 尾块 [tail:] 的陈旧 lane 一并参与向量计算，但只
                            # 写回 [0:tail]，垃圾不落 GM（vec_add_1d.py 同款）。
                            T.copy(y_out[0:tail], y[flat0 : flat0 + tail])
                        else:
                            T.copy(yf[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
# Per-dtype tolerance (DESIGN.md section 8.2, identical to the source repo
# test_floor_divide.py: atol = 1.0 if dtype != float32 else 1e-5).
# The fp32-transit chain is measured bit-exact vs the golden; the half
# tolerances are the source contract.
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1.0, 0.0),
    "bfloat16": (1.0, 0.0),
}

_FACTORY_CACHE = {}

# Per-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel):
    key = (N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = floor_divide_fwd_kernel(
            N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str, mode="positive"):
    """Random CPU data (source tests/bench use positive operands).

    mode: ``positive`` -> [0.1, 4.0); ``negative`` -> (-4.0, -0.1];
    ``signed`` -> randn (exercises the toward-minus-inf floor branch).
    """
    if mode == "positive":
        base = torch.rand(shape, dtype=torch.float32) * 3.9 + 0.1
    elif mode == "negative":
        base = -(torch.rand(shape, dtype=torch.float32) * 3.9 + 0.1)
    else:
        base = torch.randn(shape, dtype=torch.float32)
    return base.to(_TORCH_DTYPE[dtype_str])


def _compare(out, ref, dtype_str, equal_nan=False):
    """Return (max_diff, violations); NaN/Inf aware when equal_nan is set."""
    atol, rtol = _TOLERANCE[dtype_str]
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(f"shape mismatch: out {tuple(out.shape)} != ref {tuple(ref.shape)}")
    if out.dtype != ref.dtype:
        raise AssertionError(f"dtype mismatch: out {out.dtype} != ref {ref.dtype}")
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


def _run_case(
    a_shape,
    b_shape,
    dtype_str,
    tag,
    block_size=None,
    a_cpu=None,
    b_cpu=None,
    mode="positive",
    equal_nan=False,
    exact=False,
):
    """Run one kernel case and compare against the golden.

    ``a_cpu``/``b_cpu`` may be supplied verbatim (special-value cases);
    otherwise random data is generated (``mode``: positive/negative/signed).
    ``exact=True`` additionally requires bit-exact equality (fp32 path).
    """
    if block_size is None:
        block_size = _DEFAULT_BLOCK

    out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
    N_total = math.prod(out_shape)
    a_numel = math.prod(a_shape) if len(a_shape) else 1
    b_numel = math.prod(b_shape) if len(b_shape) else 1

    if a_cpu is None:
        a_cpu = _rand_cpu(a_shape, dtype_str, mode)
        b_cpu = _rand_cpu(b_shape, dtype_str, mode)
    assert b_cpu is not None, "a_cpu and b_cpu must be supplied together"

    ref = golden_floor_divide(a_cpu, b_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )
    assert ref.dtype == _TORCH_DTYPE[dtype_str], f"golden dtype {ref.dtype} unexpected"

    a_npu = a_cpu.reshape(-1).npu()
    b_npu = b_cpu.reshape(-1).npu()

    fn = _get_factory(N_total, dtype_str, cshape, a_strides, b_strides, a_numel, b_numel)
    y = fn(block_size)(a_npu, b_npu)
    out = y.cpu().reshape(out_shape)

    if exact and not torch.equal(out, ref):
        viol = int((out != ref).sum().item())
        md = (out.float() - ref.float()).abs().max().item()
        raise AssertionError(f"exact compare failed: {viol} violation(s), max_diff={md}")

    max_diff, viol = _compare(out, ref, dtype_str, equal_nan=equal_nan)
    torch.testing.assert_close(
        out, ref, rtol=_TOLERANCE[dtype_str][1], atol=_TOLERANCE[dtype_str][0],
        equal_nan=equal_nan,
    )
    if max_diff == max_diff and max_diff > _STATS_MAX_DIFF.get(dtype_str, -1.0):
        _STATS_MAX_DIFF[dtype_str] = max_diff  # max_diff == max_diff skips NaN

    print(
        f"[{tag}] PASS: a={tuple(a_shape)} b={tuple(b_shape)} dtype={dtype_str} "
        f"block={block_size} max_diff={max_diff}"
    )
    return max_diff, viol


def _try_case(failures, case_id, **kwargs):
    """Run ``_run_case``, collecting (not re-raising) per-case failures."""
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, twelve items)."""
    failures = []

    # L0-1: smoke same-shape, all float dtypes (manifest smoke-1k).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-1/{dtype_str}/4096", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L0-1")
    _try_case(failures, "L0-1/float32/1024", a_shape=(1024,), b_shape=(1024,), dtype_str="float32", tag="L0-1")

    # L0-2: large same-shape (manifest hidden-state-prefill, chunks=2048).
    for dtype_str in ("float32", "float16"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            a_shape=(2048, 4096),
            b_shape=(2048, 4096),
            dtype_str=dtype_str,
            tag="L0-2",
        )

    # L0-3: broadcast three modes (manifest cnn-feat + row-broadcast + bias).
    for dtype_str in ("float32", "float16"):
        for a_shape, b_shape, label in (
            ((16, 256, 56, 56), (256, 1, 1), "cnn-feat"),
            ((2, 64, 128), (1, 1, 128), "bias"),
        ):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/{label}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L0-3",
            )
    _try_case(
        failures,
        "L0-3/float32/row-broadcast",
        a_shape=(4096, 4096),
        b_shape=(1, 4096),
        dtype_str="float32",
        tag="L0-3",
    )

    # L0-4: a-side broadcast (roles swapped, exercises sa == 0).
    _try_case(failures, "L0-4/1x4096|4096x4096", a_shape=(1, 4096), b_shape=(4096, 4096), dtype_str="float32", tag="L0-4")
    _try_case(failures, "L0-4/2x64x128|2x64x1", a_shape=(2, 64, 128), b_shape=(2, 64, 1), dtype_str="float32", tag="L0-4")

    # L0-5: scalar broadcast / full scalar (L == 1 both-vbrc degenerate).
    _try_case(failures, "L0-5/scalar-broadcast", a_shape=(64, 128), b_shape=(1, 1), dtype_str="float32", tag="L0-5")
    _try_case(failures, "L0-5/full-scalar", a_shape=(1,), b_shape=(1,), dtype_str="float32", tag="L0-5")

    # L0-6: non-divisible tails (single partial tile / non-32B-aligned tail).
    for N in (1000, 3136, 4097):
        _try_case(failures, f"L0-6/N={N}", a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="L0-6")

    # L0-7: integer / bool dtype rejection (source float-only contract).
    for bad in ("int8", "int16", "int32", "int64", "uint8", "bool"):
        try:
            _normalize_dtype(bad)
        except ValueError:
            print(f"[L0-7] PASS: dtype {bad!r} rejected")
        else:
            failures.append((f"L0-7/{bad}", AssertionError(f"{bad} not rejected")))
    try:
        floor_divide_fwd_kernel(4096, "int32", (4096,), (1,), (1,), 4096, 4096)
    except ValueError:
        print("[L0-7] PASS: factory rejects int32")
    else:
        failures.append(("L0-7/factory-int32", AssertionError("factory int32 not rejected")))

    # L0-8: fp32 no-identity-cast regression (DESIGN.md note 3, CG-2026-0024).
    _try_case(failures, "L0-8/float32-exact", a_shape=(4096,), b_shape=(4096,), dtype_str="float32", tag="L0-8", exact=True)

    # L0-9: persistent core-count switch boundary (num_logical in 39/40/41).
    bs = _DEFAULT_BLOCK
    for num_logical, N in ((39, 39 * bs), (40, 40 * bs), (41, 41 * bs)):
        _try_case(
            failures,
            f"L0-9/num_logical={num_logical}",
            a_shape=(N,),
            b_shape=(N,),
            dtype_str="float32",
            tag="L0-9",
        )

    # L0-10: float corners (NaN / +-Inf / +-0 / subnormal / integer boundary).
    _run_L0_corners(failures)

    # L0-11: vdiv broadcast capability regression ([M,N]/[1,N], [M,N]/[M,1],
    # [M,N]/[1,1]) -- covers the R1 materialisation for both operand roles.
    for a_shape, b_shape, label in (
        ((64, 128), (1, 128), "col"),
        ((64, 128), (64, 1), "row"),
        ((64, 128), (1, 1), "scalar"),
    ):
        _try_case(
            failures,
            f"L0-11/{label}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="float32",
            tag="L0-11",
        )

    # L0-12: divisor boundary (small positive divisor, b == 1.0).
    try:
        a = _rand_cpu((4096,), "float32")
        b = torch.full((4096,), 1.0, dtype=torch.float32)
        _try_case(
            failures,
            "L0-12/b=1.0",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str="float32",
            tag="L0-12",
            a_cpu=a,
            b_cpu=b,
        )
        a2 = _rand_cpu((4096,), "float32")
        b2 = torch.full((4096,), 0.1, dtype=torch.float32)
        _try_case(
            failures,
            "L0-12/b=0.1",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str="float32",
            tag="L0-12",
            a_cpu=a2,
            b_cpu=b2,
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-12/setup", exc))

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (twelve items)")
    return failures


def _run_L0_corners(failures):
    """L0-10: float special values, fp32 + fp16 (IEEE propagation)."""
    nan = float("nan")
    inf = float("inf")
    a_pat = [6.0, 7.0, -7.0, inf, -inf, nan, 0.0, -0.0, 2.9999, -3.0001, 1.0, 1e-8, 5.5, -5.5, 8.0, 9.0]
    b_pat = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, inf, 3.0, 2.0, 2.0, -2.0, 0.5]
    reps = 4096 // len(a_pat)
    for dtype_str in ("float32", "float16"):
        a = torch.tensor(a_pat * reps, dtype=_TORCH_DTYPE[dtype_str])
        b = torch.tensor(b_pat * reps, dtype=_TORCH_DTYPE[dtype_str])
        _try_case(
            failures,
            f"L0-10/{dtype_str}",
            a_shape=(4096,),
            b_shape=(4096,),
            dtype_str=dtype_str,
            tag="L0-10",
            a_cpu=a,
            b_cpu=b,
            equal_nan=True,
        )


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size)."""
    failures = []

    # dtype x {exact multiple, tail==1-ish, non-32B-aligned tail}.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 1000, 4097):
            _try_case(failures, f"L1/{dtype_str}/N={N}", a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L1")

    # Broadcast variants across dtypes and shapes.
    for dtype_str in ("float16", "float32"):
        for a_shape, b_shape in (
            ((4, 8, 16), (1, 8, 1)),
            ((3, 5), (1, 1)),
            ((2, 3, 4), (2, 3, 4)),
            ((1, 4096), (4096, 4096)),
            ((16, 256, 56, 56), (256, 1, 1)),
        ):
            _try_case(
                failures,
                f"L1/{dtype_str}/{a_shape}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L1",
            )

    # Signed operands: the toward-minus-infinity floor branch.
    for dtype_str in ("float16", "float32"):
        _try_case(failures, f"L1/{dtype_str}/signed", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L1", mode="signed")
        _try_case(failures, f"L1/{dtype_str}/negative", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L1", mode="negative")

    # Non-default block sizes within the UB budget.
    _try_case(failures, "L1/fp32/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="float32", tag="L1", block_size=2048)
    _try_case(failures, "L1/float16/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="float16", tag="L1", block_size=2048)
    _try_case(failures, "L1/bfloat16/N=2^18/bs=1024", a_shape=(2**18,), b_shape=(2**18,), dtype_str="bfloat16", tag="L1", block_size=1024)

    # E2 factory contract: one factory callable serving two block_sizes.
    try:
        dtype_str = "float32"
        a_shape = (8192,)
        out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, a_shape)
        a = _rand_cpu(a_shape, dtype_str).npu()
        b = _rand_cpu(a_shape, dtype_str).npu()
        ref = golden_floor_divide(a.cpu(), b.cpu())
        fn = _get_factory(8192, dtype_str, cshape, a_strides, b_strides, 8192, 8192)
        for bs in (4096, 2048):
            out = fn(bs)(a, b).cpu()
            max_diff, _ = _compare(out, ref, dtype_str)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard.
    try:
        floor_divide_fwd_kernel(4096, "float16", (4096,), (1,), (1,), 4096, 4096)(6000)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects float16 block_size=6000 ({exc})")
    else:
        failures.append(("L1/UB-guard-float16", AssertionError("float16 6000 not rejected")))
    try:
        floor_divide_fwd_kernel(4096, "float32", (4096,), (1,), (1,), 4096, 4096)(8000)
    except ValueError as exc:
        print(f"[L1] PASS: UB guard rejects float32 block_size=8000 ({exc})")
    else:
        failures.append(("L1/UB-guard-float32", AssertionError("float32 8000 not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in ((1, "float32"), (2, "float16"), (3, "bfloat16"), (2**22, "float16"), (2**24, "float32")):
        try:
            _run_case(a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # all zeros: 0 / 0 -> NaN (IEEE), matches the golden.
    try:
        a = torch.zeros(N, dtype=torch.float32)
        b = torch.zeros(N, dtype=torch.float32)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="Boundary-zeros", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # a == b -> floor(1) == 1 (positive), 0 -> -1 (negative) etc.
    try:
        a = _rand_cpu((N,), "float32", "signed")
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="Boundary-a-eq-b", a_cpu=a, b_cpu=a.clone(), equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): a==b: {e}")

    # subnormal magnitudes.
    try:
        a = (torch.randn(N, dtype=torch.float32) * 1e-8).to(torch.float16)
        b = (torch.rand(N, dtype=torch.float32) * 0.9 + 0.1).to(torch.float16)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float16", tag="Boundary-subnormal", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # large magnitudes.
    try:
        a = (torch.rand(N, dtype=torch.float32) * 100.0 + 1.0).to(torch.float16)
        b = (torch.rand(N, dtype=torch.float32) * 3.0 + 0.1).to(torch.float16)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float16", tag="Boundary-large", a_cpu=a, b_cpu=b, equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # scalar broadcast with a special divisor (x / scalar).
    try:
        a = _rand_cpu((N,), "float32")
        b = torch.tensor([2.5], dtype=torch.float32)
        _run_case(a_shape=(N,), b_shape=(1,), dtype_str="float32", tag="Boundary-scalar-div", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): scalar-div: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    if args.level in ("L0", "all"):
        t0 = time.time()
        failures += run_L0()
        print(f"[L0] elapsed {time.time() - t0:.1f}s")
    if args.level in ("L1", "all"):
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
    if args.level in ("L2", "all"):
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
    if args.level in ("boundary", "all"):
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
