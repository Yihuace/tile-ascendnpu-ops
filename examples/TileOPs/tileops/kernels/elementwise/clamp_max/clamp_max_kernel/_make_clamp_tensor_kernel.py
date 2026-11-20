# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Clamp-max forward kernel (``_make_clamp_tensor_kernel`` / ``ClampMaxFwdOp``).

Migrated from the GPU TileOPs operator ``_make_clamp_tensor_kernel`` (the
``ClampTensorFwdKernel`` builder factory; ``ClampMaxFwdOp`` uses its
``has_max``-only branch, i.e. ``y = min(x, hi)``) to TileLang
``target="npuir"`` on Ascend NPU.  Programming mode: **Developer**.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = min(x[off_x(i)], hi[off_hi(i)])           (element-wise)
on two tensors that the Op layer has flattened to contiguous 1-D; the public
result has shape ``broadcast_shapes(x.shape, hi.shape)`` and dtype
``same_as(x)`` (float16 / bfloat16 / float32 -- **not** bool: clamp_max is a
numeric truncation, not a comparison).  ``off`` are the broadcast index maps.
NaN semantics match ``torch.clamp_max``: a NaN in ``x`` or ``hi`` at a
position yields NaN there.  ``min`` is an operand *selection* (no arithmetic),
so the result is bit-exact in the native dtype.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element scalar chain (fp32 upcast -> ``T.min`` -> two
        ``isnan`` guards -> cast back, ~8 scalar ops/element) is replaced by a
        **single native-dtype vector ``T.vmin(x_ub, hi_ub, y_ub)``**.  Both of
        the source's redundant steps are dropped after machine verification
        (verify_equiv.py) and design probes:
          * the explicit NaN guard -- NPU ``hivm::VMinOp`` already propagates
            NaN exactly like ``torch.clamp_max`` (probe Q-B + E1);
          * the fp32 transit -- ``min`` has no accumulation and its result is
            always one of the operands, so native-dtype min is bit-exact
            (E2).  Per-task vector op count drops 8 -> 1.
    R2: CUDA grid x block two-level parallelism becomes a one-dimensional
        ``T.Kernel(num_kernels, is_npu=True)`` with a persistent in-core
        ``T.serial`` grid-stride loop.  ``num_kernels = min(num_logical,
        aicore*2)`` (pure-Vector cores; this machine: 20 AI cores -> 40).
    R3: the upper-bound broadcast is moved **into the kernel** (row
        decomposition + native ``T.vmin`` scalar / one-dim broadcast), so the
        Op layer never has to host-materialise the expanded bound (which
        would violate ascend-constraints.md section 4 and add +2N*E GM
        traffic).  Two innermost-stride cases:
          * ``hi_strides[-1] == 1`` (row-broadcast / same-shape): load the
            contiguous bound segment and use a same-shape ``T.vmin``;
          * ``hi_strides[-1] == 0`` (innermost broadcast, e.g. cnn-feat
            ``[256,1,1]``): load the single per-row bound element into
            ``hi1`` and let ``T.vmin`` broadcast it (``T.vbrc`` fallback
            documented below).
        ``x`` is always the full flattened output shape (manifest has no
        x-broadcast workload); the Op layer keeps its contract for that case.

Implementation notes:
    - ``T.vmin`` accepts bf16 natively even though
      ``docs/Tilelang.language/比较操作/T.vmin.md`` section 2.2.1 marks bf16
      "x"; this was re-confirmed on hardware by the L0-1 bf16 case (R1).
    - The one-dim broadcast form ``(block_size,)`` vs ``(1,)`` used by the
      innermost-broadcast path is verified natively by L0-2/L0-6 (REVIEW.md
      dimension 1 note).  ``T.vbrc`` is an equivalent fallback.
    - ``block_size`` is folded inside the jit function so the returned
      callable stays consistent for every ``block_size`` it is invoked with
      (TRAP-jit-blocksize-decomposition).

Factory contract (frozen; wrapper
``examples/TileOPs/tileops/kernels/elementwise/clamp_max/clamp_max.py``):
    _make_clamp_tensor_kernel(N_total, dtype, has_min, has_max,
                              coalesced_shape=None, hi_strides=None,
                              hi_numel=None, block_size=4096)
        -> kernel(block_size) -> main(x, hi, y)          (``out_idx=[2]``)
    ``has_min=False`` / ``has_max=True`` selects the clamp_max branch.

Run the embedded hierarchical tests:
    python _make_clamp_tensor_kernel.py --level L0
    python _make_clamp_tensor_kernel.py --level all
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
    "_make_clamp_tensor_kernel",
    "clamp_max_fwd_kernel",
    "coalesce_broadcast_dims",
    "golden_clamp_max",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

# Wrapper K11 default: block_size = GPU threads * npt.
_DEFAULT_BLOCK = 4096

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

# Accuracy standard (DESIGN.md section 8.2): min is an operand selection, so
# same-dtype kernel and golden are expected bit-exact; these are margins.
_ATOL_RTOL = {
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "float32": (1e-5, 1e-5),
}

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _dtype_name(dtype):
    if isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    return str(dtype)


def _default_block_size(dtype):
    """Wrapper K11 default: 2048 for fp32, 4096 for fp16/bf16."""
    return 2048 if _dtype_name(dtype) == "float32" else 4096


def _max_block(dtype):
    """Upper bound on block_size from the UB budget (DESIGN.md section 4.5).

    3 active buffers (x_ub / hi_ub / y_ub) x block_size x E, x1.7 inflation:
    fp16/bf16 ~19275, fp32 ~9637.
    """
    e = _ELEM_BYTES[_dtype_name(dtype)]
    return int(_UB_CAPACITY_BYTES // (3 * e * _UB_INFLATION))


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_clamp_max(input, max):
    """Clamp-max reference (source-operator semantics, direct broadcast).

    Matches the source ``torch.clamp_max(x, hi)``: y = min(x, hi) element-wise
    with PyTorch broadcasting, output dtype = input dtype, and NaN propagation
    (a NaN in either operand yields NaN).  Independent of the NPU algorithm
    (no row/inner indexing, no ``vmin``, no persistent structure, no tail
    handling) and runs on CPU without an NPU device.
    """
    return torch.clamp_max(input.detach().cpu(), max.detach().cpu())


# Aliases so the generic ``golden_<op>`` gate name resolves for
# op_name == "_make_clamp_tensor_kernel" as well.
golden__make_clamp_tensor_kernel = golden_clamp_max
golden_clamp_max_fwd_kernel = golden_clamp_max


# ---------------------------------------------------------------------------
# Broadcast lowering (verbatim from the source operator's helper)
# ---------------------------------------------------------------------------
def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions.

    Merges adjacent dimensions that have the same broadcast behaviour
    (both real or both broadcast) to minimise the number of divmod operations
    inside the kernel body.  This is a lowering decision, not part of the op
    contract.

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
    """Build the outer-row base offset expression for the upper bound.

    ``row`` is the runtime row index; ``outer_shape`` and ``strides`` are
    Python constants, so the loop unrolls at trace time into a chain of
    ``T.floordiv`` / ``T.floormod`` ops (one divmod per outer dimension,
    executed once per task rather than once per element -- DESIGN.md R3).
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
def _make_clamp_tensor_kernel(
    N_total,
    dtype,
    has_min,
    has_max,
    coalesced_shape=None,
    hi_strides=None,
    hi_numel=None,
    block_size=_DEFAULT_BLOCK,
):
    """Build the clamp-max kernel factory (DESIGN.md section 3.3).

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(x, hi)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).  ``ClampMaxFwdOp`` uses ``has_min=False`` /
    ``has_max=True``.

    ``coalesced_shape`` / ``hi_strides`` / ``hi_numel`` describe the (possibly
    broadcast) upper bound for the in-kernel broadcast path (R3); when
    ``coalesced_shape is None`` the same-shape degenerate path (R = 1,
    L = N_total, identity bound) is used, exactly as the frozen wrapper
    invocation ``_make_clamp_tensor_kernel(N, dtype, has_min, has_max)``.
    """
    if not has_max:
        raise ValueError("_make_clamp_tensor_kernel (clamp_max) requires has_max=True")
    if has_min:
        raise ValueError(
            "this NPU kernel implements the has_max-only (clamp_max) branch; "
            "has_min is not supported"
        )
    dtype = _dtype_name(dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {_SUPPORTED_DTYPES}"
        )
    out_dtype = dtype  # same_as(input) (DESIGN.md section 0.2)
    N_total = int(N_total)
    if N_total <= 0:
        raise ValueError(f"N_total must be a positive int, got {N_total}")

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op).  Re-queried per factory call so other deployments adapt.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    same_shape = coalesced_shape is None
    if same_shape:
        L = N_total
        outer = []
        hi_outer_strides = []
        hi_last_stride = 1
        hi_numel_eff = N_total
    else:
        coalesced_shape = tuple(int(x) for x in coalesced_shape)
        hi_strides = tuple(int(x) for x in hi_strides)
        if len(coalesced_shape) == 0 or len(hi_strides) != len(coalesced_shape):
            raise ValueError("coalesced_shape / hi_strides rank mismatch")
        L = coalesced_shape[-1]
        outer = list(coalesced_shape[:-1])
        hi_outer_strides = list(hi_strides[:-1])
        hi_last_stride = hi_strides[-1]
        if hi_last_stride not in (0, 1):
            raise ValueError(f"innermost hi stride must be 0 or 1, got {hi_last_stride}")
        hi_numel_eff = int(hi_numel) if hi_numel is not None else N_total
        if math.prod(coalesced_shape) != N_total:
            raise ValueError(
                f"prod(coalesced_shape)={math.prod(coalesced_shape)} != N_total={N_total}"
            )
    R = math.prod(outer) if outer else 1

    # Innermost-broadcast upper bound (e.g. cnn-feat [256,1,1]): one clipped
    # scalar per row, broadcast natively by T.vmin.  Pre-combined as a plain
    # pure-name Python flag so the npuir script parser folds the branch.
    hi_broadcast_inner = (not same_shape) and hi_last_stride == 0

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        # Host-side UB budget guard (DESIGN.md section 4.5).
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        max_block = _max_block(dtype)
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {dtype} "
                f"(max {max_block} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # block_size-dependent task decomposition, evaluated here (not at
        # factory time) so the compiled kernel always matches the block_size
        # it is invoked with (TRAP-jit-blocksize-decomposition).  block_size
        # folds to a compile-time constant at trace time.
        chunks = T.ceildiv(L, block_size)
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            x: T.Tensor((N_total,), dtype),
            hi: T.Tensor((hi_numel_eff,), dtype),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # 全部 buffer 无条件声明：npuir 脚本解析器会解析所有条件分支体
                # （即使条件是 Python 常量），分支里引用的 buffer 若无条件声明
                # 之外则报 Undefined variable。未用分支由编译器 DCE 掉。
                x_ub = T.alloc_ub((block_size,), dtype)
                hi_ub = T.alloc_ub((block_size,), dtype)  # 末维非广播（同形段）
                hi1 = T.alloc_ub((1,), dtype)             # 末维广播（每行标量上界）
                y_ub = T.alloc_ub((block_size,), dtype)

                # persistent 核内串行循环：边界 num_local_tasks 为编译期静态值
                # （由 num_logical/num_kernels 常量除得），grid-stride 映射
                # task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # 行 base 标量：每 task 仅一次 unrolled divmod（非每元素），
                        # 末维仅在 stride==1 时贡献 inner_off。x 恒为完整展平输出
                        # 形状，故 x_base = row*L + inner_off。
                        tail = T.min(block_size, L - inner_off)
                        x_base = row * L + inner_off
                        T.copy(x[x_base : x_base + tail], x_ub[0:tail])
                        if hi_broadcast_inner:
                            # 末维广播（cnn-feat [256,1,1]）：每行一个标量上界，
                            # 单元素搬入后用 T.vmin 原生一维广播 (1,)->(block_size,)；
                            # 兜底可 T.vbrc(hi1, hi_ub) 后同形 T.vmin。
                            hi_base = _row_base(row, outer, hi_outer_strides)
                            T.copy(hi[hi_base : hi_base + 1], hi1[0:1])
                            T.vmin(x_ub, hi1, y_ub)
                        else:
                            # 末维非广播（同形 / row-broadcast [1,4096]）：上界连续
                            # 段搬入，同形 T.vmin。row-broadcast 时 hi_base 恒 0。
                            hi_base = _row_base(row, outer, hi_outer_strides) + inner_off
                            T.copy(hi[hi_base : hi_base + tail], hi_ub[0:tail])
                            # 单条向量 vmin：原生 dtype，无 fp32 中转、无 isnan 守卫
                            # （NPU VMinOp 已传播 NaN，与 torch.clamp_max 逐点一致）。
                            # 尾块 [tail:] 的陈旧数据一并参与计算但不写回 GM。
                            T.vmin(x_ub, hi_ub, y_ub)
                        T.copy(y_ub[0:tail], y[x_base : x_base + tail])

        return main

    return kernel


# DESIGN.md section 3.3 factory name (alias).
clamp_max_fwd_kernel = _make_clamp_tensor_kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_FACTORY_CACHE = {}

# Per public-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(N_total, dtype, cshape, hi_strides, hi_numel):
    key = (N_total, dtype, cshape, hi_strides, hi_numel)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _make_clamp_tensor_kernel(
            N_total,
            dtype,
            False,
            True,
            coalesced_shape=cshape,
            hi_strides=hi_strides,
            hi_numel=hi_numel,
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str, scale=4.0):
    shape = tuple(shape)
    return (torch.randn(shape, dtype=torch.float32) * scale).to(_TORCH_DTYPE[dtype_str])


def _compare(out, ref, dtype_str):
    """NaN-pattern-aware comparison; returns max_diff over finite positions."""
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(
            f"shape mismatch: out {tuple(out.shape)} != ref {tuple(ref.shape)}"
        )
    if out.dtype != ref.dtype:
        raise AssertionError(f"dtype mismatch: out {out.dtype} != ref {ref.dtype}")
    if not torch.equal(torch.isnan(out), torch.isnan(ref)):
        raise AssertionError("NaN pattern mismatch between kernel and golden")
    fin = ~torch.isnan(out) & ~torch.isnan(ref)
    atol, rtol = _ATOL_RTOL[dtype_str]
    if bool(fin.any()):
        # assert_close handles +-Inf correctly (inf == inf, inf != -inf).
        torch.testing.assert_close(
            out[fin].to(torch.float64), ref[fin].to(torch.float64),
            rtol=rtol, atol=atol, equal_nan=True,
        )
    # max_diff is measured only over positions that are finite in BOTH (so an
    # equal +Inf pair does not produce an inf-inf = NaN artifact).
    both_finite = torch.isfinite(out) & torch.isfinite(ref)
    max_diff = 0.0
    if bool(both_finite.any()):
        o = out[both_finite].to(torch.float64)
        r = ref[both_finite].to(torch.float64)
        max_diff = float((o - r).abs().max().item())
    return max_diff


def _run_case(
    x_shape,
    hi_shape,
    dtype_str,
    tag,
    block_size=None,
    coalesce=True,
    x_cpu=None,
    hi_cpu=None,
):
    """Run one clamp_max case and compare against the golden.

    ``x_shape`` is the (full) output shape -- the Op layer hands the kernel a
    flattened output-shaped ``x``.  ``hi_shape`` may be smaller (broadcast).
    ``coalesce=False`` exercises the frozen same-shape wrapper path
    (``coalesced_shape=None``); ``coalesce=True`` exercises the R3 in-kernel
    broadcast path.
    """
    dtype_str = _dtype_name(dtype_str)
    if block_size is None:
        block_size = _default_block_size(dtype_str)
    if len(hi_shape) == 0:
        hi_shape = (1,)
    hi_shape = tuple(hi_shape)
    x_shape = tuple(x_shape)

    out_shape, cshape, _a_strides, hi_strides = coalesce_broadcast_dims(x_shape, hi_shape)
    N_total = math.prod(out_shape)
    hi_numel = math.prod(hi_shape)

    if x_cpu is None:
        x_cpu = _rand_cpu(x_shape, dtype_str)
    if hi_cpu is None:
        hi_cpu = _rand_cpu(hi_shape, dtype_str)
    x_cpu = x_cpu.to(_TORCH_DTYPE[dtype_str])
    hi_cpu = hi_cpu.to(_TORCH_DTYPE[dtype_str])

    ref = golden_clamp_max(x_cpu, hi_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )

    # The Op layer guarantees x at the output shape (manifest has no
    # x-broadcast workload); materialise only if a test hands a smaller x.
    x_full = x_cpu
    if tuple(x_cpu.shape) != tuple(out_shape):
        x_full = x_cpu.expand(out_shape).contiguous()
    x_npu = x_full.reshape(-1).npu()
    hi_npu = hi_cpu.reshape(-1).npu()

    if coalesce:
        fn = _get_factory(N_total, dtype_str, cshape, hi_strides, hi_numel)
    else:
        fn = _get_factory(N_total, dtype_str, None, None, None)
    y = fn(block_size)(x_npu, hi_npu)
    out = y.cpu().reshape(out_shape)

    max_diff = _compare(out, ref, dtype_str)
    if max_diff > _STATS_MAX_DIFF.get(dtype_str, -1):
        _STATS_MAX_DIFF[dtype_str] = max_diff
    print(
        f"[{tag}] PASS: x={x_shape} hi={hi_shape} dtype={dtype_str} "
        f"block={block_size} {'coalesced' if coalesce else 'same-shape'} "
        f"max_diff={max_diff}"
    )
    return max_diff


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


def _corner_pair(N, dtype_str):
    """Build an fp pair covering the clamp_max IEEE corner semantics.

    NaN in x / NaN in hi (must propagate to NaN), +-Inf, +-0 (selection is
    sign-preserving: min(+0,-0) follows IEEE minimum), dtype max, smallest
    normal and smallest positive subnormal (DESIGN.md risk R6 / L0-4).
    """
    td = _TORCH_DTYPE[dtype_str]
    info = torch.finfo(td)
    nan = float("nan")
    inf = float("inf")
    sub = float(info.tiny) * float(info.eps)
    xs = [nan, nan, 1.0, inf, -inf, 0.0, -0.0, info.max, -info.max,
          info.tiny, -info.tiny, sub, -sub, 1.0, -2.0, 3.0]
    hs = [1.0, nan, nan, inf, -inf, -0.0, 0.0, -info.max, info.max,
          sub, -sub, info.tiny, -info.tiny, 2.0, -1.0, -3.0]
    reps = N // len(xs) + 1
    x = torch.tensor(xs * reps, dtype=torch.float64).to(td)[:N].clone()
    h = torch.tensor(hs * reps, dtype=torch.float64).to(td)[:N].clone()
    return x, h


def _nan_case(N, dtype_str):
    """x and hi with NaN / +-Inf positions (DESIGN.md L0-3)."""
    td = _TORCH_DTYPE[dtype_str]
    x = _rand_cpu((N,), dtype_str)
    h = _rand_cpu((N,), dtype_str)
    x[0] = float("nan")
    h[1] = float("nan")
    x[2] = float("inf")
    h[3] = float("-inf")
    x[4] = float("-inf")
    h[5] = float("inf")
    x[6:8] = float("nan")
    h[6:8] = float("nan")
    return x.to(td), h.to(td)


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, eight items)."""
    failures = []

    # L0-1: same-shape (frozen wrapper path) + smoke.  bf16 is MANDATORY
    # (T.vmin doc marks bf16 "x" while the probe says otherwise -> R1).
    _try_case(failures, "L0-1/smoke-fp32", x_shape=(1024,), hi_shape=(1024,),
              dtype_str="float32", tag="L0-1", coalesce=False)
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-1/same/{dtype_str}", x_shape=(4096, 4096),
                  hi_shape=(4096, 4096), dtype_str=dtype_str, tag="L0-1",
                  coalesce=False)

    # L0-2: broadcast manifest workloads + inner-dim broadcast + 0-dim bound.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for x_shape, hi_shape, label in (
            ((16, 256, 56, 56), (256, 1, 1), "cnn-feat"),
            ((4096, 4096), (1, 4096), "row-broadcast"),
        ):
            _try_case(failures, f"L0-2/{dtype_str}/{label}", x_shape=x_shape,
                      hi_shape=hi_shape, dtype_str=dtype_str, tag="L0-2")
    _try_case(failures, "L0-2/fp32/col-brc", x_shape=(4, 8), hi_shape=(4, 1),
              dtype_str="float32", tag="L0-2")

    # L0-3: NaN propagation (R2) -- same-shape and broadcast.
    for dtype_str in ("float16", "bfloat16", "float32"):
        x, h = _nan_case(4096, dtype_str)
        _try_case(failures, f"L0-3/nan/{dtype_str}", x_shape=(4096,),
                  hi_shape=(4096,), dtype_str=dtype_str, tag="L0-3",
                  coalesce=False, x_cpu=x, hi_cpu=h)
    # NaN in the broadcast bound (every row shares the same NaN bound).
    xn = _rand_cpu((64, 128), "float32")
    hn = _rand_cpu((128,), "float32")
    hn[3] = float("nan")
    hn[5] = float("inf")
    _try_case(failures, "L0-3/nan-broadcast", x_shape=(64, 128), hi_shape=(128,),
              dtype_str="float32", tag="L0-3", x_cpu=xn, hi_cpu=hn)

    # L0-4: IEEE corners (NaN / +-Inf / +-0 / max / subnormal).
    for dtype_str in ("float16", "bfloat16", "float32"):
        x, h = _corner_pair(4096, dtype_str)
        _try_case(failures, f"L0-4/corners/{dtype_str}", x_shape=(4096,),
                  hi_shape=(4096,), dtype_str=dtype_str, tag="L0-4",
                  coalesce=False, x_cpu=x, hi_cpu=h)

    # L0-5: non-divisible tails (single partial tile / unaligned tail).
    for N in (1000, 3136, 4097):
        _try_case(failures, f"L0-5/N={N}", x_shape=(N,), hi_shape=(N,),
                  dtype_str="float32", tag="L0-5", coalesce=False)

    # L0-6: scalar bound path -- 0-dim bound, full scalar, (1,) bound.
    _try_case(failures, "L0-6/scalar-dim0", x_shape=(4, 8), hi_shape=(),
              dtype_str="float32", tag="L0-6")
    _try_case(failures, "L0-6/scalar-b1", x_shape=(4, 8), hi_shape=(1, 1),
              dtype_str="float16", tag="L0-6")
    _try_case(failures, "L0-6/full-scalar", x_shape=(1024,), hi_shape=(1,),
              dtype_str="bfloat16", tag="L0-6")

    # L0-7: OP-layer I/O contract -- output shape == broadcast_shapes and
    # dtype == input dtype (asserted inside _run_case / _compare).
    _try_case(failures, "L0-7/shape-contract", x_shape=(2, 3, 4), hi_shape=(1, 3, 1),
              dtype_str="float32", tag="L0-7")

    # L0-8: boundary of the persistent split (num_logical 39/40/41).
    bs = _DEFAULT_BLOCK
    for num_logical in (39, 40, 41):
        N = num_logical * bs
        _try_case(failures, f"L0-8/num_logical={num_logical}", x_shape=(N,),
                  hi_shape=(N,), dtype_str="float32", tag="L0-8", coalesce=False)

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (eight items)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size)."""
    failures = []

    # dtype x {exact multiple, non-32B-aligned tail, multi-chunk}.
    for dtype_str in _SUPPORTED_DTYPES:
        for N in (8192, 1000, 4097):
            _try_case(failures, f"L1/{dtype_str}/N={N}", x_shape=(N,), hi_shape=(N,),
                      dtype_str=dtype_str, tag="L1", coalesce=False)

    # Broadcast variants across dtypes and shapes.
    for dtype_str in ("float16", "float32", "bfloat16"):
        for x_shape, hi_shape in (
            ((4, 8, 16), (1, 8, 1)),
            ((3, 5), (1, 1)),
            ((2, 3, 4), (2, 3, 4)),
            ((8, 1, 256), (1, 1, 256)),
        ):
            _try_case(failures, f"L1/{dtype_str}/{x_shape}", x_shape=x_shape,
                      hi_shape=hi_shape, dtype_str=dtype_str, tag="L1")

    # Non-default block sizes within the UB budget.
    _try_case(failures, "L1/fp16/N=2^20/bs=8192", x_shape=(2**20,), hi_shape=(2**20,),
              dtype_str="float16", tag="L1", block_size=8192, coalesce=False)
    _try_case(failures, "L1/fp32/N=2^20/bs=2048", x_shape=(2**20,), hi_shape=(2**20,),
              dtype_str="float32", tag="L1", block_size=2048, coalesce=False)
    _try_case(failures, "L1/bf16/N=2^18/bs=1024", x_shape=(2**18,), hi_shape=(2**18,),
              dtype_str="bfloat16", tag="L1", block_size=1024, coalesce=False)

    # E2 factory contract: one factory callable serving two block_sizes.
    try:
        dtype_str = "float32"
        x_shape = (8192,)
        _, cshape, _a, hi_strides = coalesce_broadcast_dims(x_shape, x_shape)
        x = _rand_cpu(x_shape, dtype_str)
        h = _rand_cpu(x_shape, dtype_str)
        x_npu, h_npu = x.npu(), h.npu()
        ref = golden_clamp_max(x, h)
        fn = _get_factory(8192, dtype_str, None, None, None)
        for bs in (4096, 2048):
            out = fn(bs)(x_npu, h_npu).cpu()
            _compare(out, ref, dtype_str)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard + dtype rejection.
    for _, bs in (("float16", 32768), ("float32", 16384), ("bfloat16", 65536)):
        try:
            _make_clamp_tensor_kernel(4096, _, False, True)(bs)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {_} block_size={bs} ({exc})")
        else:
            failures.append((f"L1/UB-guard-{_}-{bs}", AssertionError(f"{_} {bs} not rejected")))
    for bad in ("float64", "float8_e4m3fn", "int32"):
        try:
            _make_clamp_tensor_kernel(4096, bad, False, True)
        except ValueError:
            print(f"[L1] PASS: dtype {bad!r} rejected")
        else:
            failures.append((f"L1/dtype-reject-{bad}", AssertionError(f"{bad} not rejected")))

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme sizes (warn only, non-blocking)."""
    for N, dtype_str in ((1, "float32"), (2, "float16"), (3, "bfloat16"),
                         (2**22, "float16"), (2**22, "float32")):
        try:
            _run_case(x_shape=(N,), hi_shape=(N,), dtype_str=dtype_str, tag="L2",
                      coalesce=False)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # All-NaN x against finite hi -> all NaN.
    try:
        x = torch.full((N,), float("nan"), dtype=torch.float32)
        h = _rand_cpu((N,), "float32")
        _run_case(x_shape=(N,), hi_shape=(N,), dtype_str="float32",
                  tag="Boundary-all-nan-x", coalesce=False, x_cpu=x, hi_cpu=h)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-nan-x: {e}")

    # All +Inf x against finite hi -> hi (min picks the finite bound).
    try:
        x = torch.full((N,), float("inf"), dtype=torch.float32)
        h = _rand_cpu((N,), "float32")
        _run_case(x_shape=(N,), hi_shape=(N,), dtype_str="float32",
                  tag="Boundary-inf-x", coalesce=False, x_cpu=x, hi_cpu=h)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf-x: {e}")

    # Signed-zero selection: min(+0,-0) / min(-0,+0).
    try:
        x = torch.tensor([0.0, -0.0] * (N // 2), dtype=torch.float32)
        h = torch.tensor([-0.0, 0.0] * (N // 2), dtype=torch.float32)
        _run_case(x_shape=(N,), hi_shape=(N,), dtype_str="float32",
                  tag="Boundary-signed-zero", coalesce=False, x_cpu=x, hi_cpu=h)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): signed-zero: {e}")

    # Subnormal-heavy fp16 (tiny values, exact in native dtype).
    try:
        info = torch.finfo(torch.float16)
        sub = float(info.tiny) * float(info.eps)
        x = torch.tensor([sub, -sub, info.tiny, -info.tiny] * (N // 4),
                         dtype=torch.float32).to(torch.float16)
        h = torch.tensor([-sub, sub, info.tiny, sub] * (N // 4),
                         dtype=torch.float32).to(torch.float16)
        _run_case(x_shape=(N,), hi_shape=(N,), dtype_str="float16",
                  tag="Boundary-subnormal", coalesce=False, x_cpu=x, hi_cpu=h)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")

    # Scalar bound of NaN (0-dim bound NaN -> all NaN).
    try:
        x = _rand_cpu((N,), "float32")
        h = torch.tensor([float("nan")], dtype=torch.float32)
        _run_case(x_shape=(N,), hi_shape=(1,), dtype_str="float32",
                  tag="Boundary-scalar-nan", x_cpu=x, hi_cpu=h)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): scalar-nan: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0",
        choices=["L0", "L1", "L2", "boundary", "all"],
    )
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
