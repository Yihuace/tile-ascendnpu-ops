# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Greater-than (``gt``) forward kernel (NPU, Developer mode).

Migrated from the GPU TileOPs operator ``gt_fwd_kernel`` (the
``_make_binary_explicit`` factory specialised with ``op_func(a, b) = a > b``
and ``OUTPUT_DTYPE = torch.bool``) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = (a[off_a(i)] > b[off_b(i)]) in {0, 1}
on two already-flattened contiguous ``(a_numel,)`` / ``(b_numel,)`` tensors
of the same dtype; the public result is ``broadcast_shapes(a_shape, b_shape)``
with dtype ``bool``.  ``off_a`` / ``off_b`` are the broadcast index maps; for
same-shape inputs they are the identity.  Float comparisons follow IEEE
STRICT ordering: any NaN operand -> False, and equal operands -- including
``+0.0`` vs ``-0.0``, ``+Inf`` vs itself and ``a > a`` -- are ALWAYS False
(this is exactly where ``gt`` differs from the weak ordering ``ge``).
Integers compare by value, and for ``uint8`` by *unsigned* value.

NPU redesign (DESIGN.md section 0.6):
    R1: the source's per-element unrolled divmod chain (broadcast path only)
        is replaced by an outer-row scalar base (one unrolled mixed-radix
        divmod per task) plus innermost-dimension vectorisation.  Because
        ``T.vcmp`` accepts **no** automatic broadcast (CG-2026-0021), BOTH
        operands are explicitly materialised to ``(block_size,)`` before the
        same-shape compare: contiguous ``T.copy`` when the innermost stride
        is 1, ``T.vbrc`` when it is 0 (row-internal scalar broadcast).
    R2: one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with a
        persistent in-core ``T.serial`` grid-stride loop and an if-guard,
        replacing the CUDA grid x block two-level parallel structure.
        Element parallelism is handled by the Vector unit via ``T.vcmp``.
    R3: the public ``bool`` output is carried as **int8 0/1** -- a bool (i1)
        GM output compiles but hangs the Vector core at runtime (507034,
        CG-2026-0022).  The Op layer views the carrier as ``view(torch.bool)``
        (non-zero is true).  ``bool`` / ``uint8`` inputs are likewise carried
        as ``int8`` (CG-2026-0018).
    R4: ``T.vcmp`` does not accept int8/uint8, so those (and bool) operands
        are upcast to **float16** before the compare -- exact for all 256
        one-byte values (verify_equiv.py E3).  int16 is upcast to **float32**
        (exact for all 65536 values, E4; native int16 vcmp is scalarised,
        PL-1.11).  int32/int64 and the float family compare in their native
        domain.  The bool mask becomes the int8 carrier via
        ``vcast(bool->f16)`` + ``vcast(f16->i8, rint)``.
    R5: **uint8 unsigned-domain correction (gt-specific)**.  ``a > b`` is an
        *ordering* comparison, hence signedness-dependent: carrying a general
        ``uint8`` operand as ``int8`` silently reorders ~50% of byte pairs
        (verify_equiv.py E-NAIVE).  Before the compare the f16 operand is
        corrected to the true unsigned value via
        ``m = vcmp(s, 0, "lt")``; ``sp = s + 256``; ``u = vselect(m, sp, s)``
        (closed form ``u = s + 256*(s<0)``, verified 256x256 exhaustive, E5).
        Note ``T.vcmp`` has no scalar operand form, so the ``0`` constant is
        materialised as a ``(block_size,)`` f16 buffer via ``T.vbrc``
        (DESIGN.md section 3.3 note / REVIEW.md dimension 1 & 7).

Implementation notes (toolchain traps found during Stage 3):
    - The ``a_ub`` / ``b_ub`` staging buffers (upcast path) and the uint8
      correction buffers (``zero`` / ``m`` / ``sp`` / ``u_a`` / ``u_b``) are
      allocated **unconditionally** and only *used* under a Python-constant
      ``if``: guarding the ``T.alloc_ub`` itself is unsafe on the npuir script
      parser (TRAP-conditional-alloc-parse).  The compiler DCEs the unused
      buffers, so the native path keeps its tight UB budget.
    - The upcast path upcasts the one-byte operand to f16 *before* ``T.vbrc``:
      this both satisfies ``T.vcmp`` (no int8) and sidesteps the int8-dst
      ``T.vbrc`` runtime hang (TRAP-vbrc-int8); f16 vbrc is well-behaved and
      the values are exact.
    - The kernel body's ``block_size`` must be the same value used to derive
      the task decomposition (``chunks`` / ``num_logical`` / ``num_kernels``
      / ``num_local_tasks``).  Those are computed *inside* the jit function
      (not at factory-build time) so the returned callable stays consistent
      for every block_size it is invoked with (TRAP-jit-blocksize-
      decomposition).
    - STRICT order: the compare mode must be ``"gt"`` (not ``"ge"``), and the
      complement rewrite ``a > b == not (a <= b)`` must NOT be used -- it is
      IEEE-inequivalent for NaN (verify_equiv.py E-NAN).  ``"gt"`` directly
      yields the strict-order contract (equal -> False, NaN -> False).

Factory contract (frozen; see examples/TileOPs/tileops/kernels/elementwise/
gt/gt.py):
    gt_fwd_kernel(N_total, dtype, coalesced_shape, a_strides, b_strides,
                  a_numel, b_numel, output_dtype=None, block_size=4096,
                  unsigned=False)
        -> kernel(block_size) -> main(a, b, y)   (``out_idx=[2]``)

``dtype`` is the *public* dtype: ``"uint8"`` selects the int8 carrier **with**
the R5 unsigned correction; ``"bool"``/``"int8"`` select the int8 carrier
without correction.  ``unsigned=True`` is the explicit carrier-side marker for
the (R11) Op-layer integration when uint8 data is handed over as int8.

Run the embedded hierarchical tests:
    python gt_fwd_kernel.py --level L0
    python gt_fwd_kernel.py --level all
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
    "gt_fwd_kernel",
    "coalesce_broadcast_dims",
    "golden_gt",
]

# Programming mode actually validated by this file (TileOPs harness
# migration requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

_DEFAULT_BLOCK = 4096

# Native comparison-domain dtypes (vcmp-supported, no upcast needed).
_NATIVE_DTYPES = ("int16", "int32", "int64", "float16", "bfloat16", "float32")
# Kernel-side carrier dtypes (the public dtype after the Op-layer view: bool
# and uint8 become int8 -- DESIGN.md section 0.6 R3).
_KERNEL_DTYPES = ("int8",) + _NATIVE_DTYPES

# Public dtypes re-interpreted as int8 before reaching the kernel.
_CARRIER_VIEW = {"uint8": "int8", "bool": "int8"}

_ELEM_BYTES = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def _ub_bytes_per_elem(carrier, need_ufix=False):
    """Full-size UB bytes per element for one carrier dtype.

    Native path: a_c + b_c + cond(1) + yf(2) + y8(1) = 2E + 4.
    int16 path (upcast to f32): a_ub + b_ub (2+2) + a_c + b_c (4+4) +
    cond(1) + yf(2) + y8(1) = 16.
    int8 carrier path: a_ub + b_ub (1+1) + a_c + b_c (2+2) + cond(1) +
    yf(2) + y8(1) = 10; the uint8 unsigned-correction path additionally
    needs zero(2) + m(1) + sp(2) + u_a(2) + u_b(2) = 19.
    Matches DESIGN.md section 4.5 (uint8 row re-derived upward once the
    materialised zero operand / independent-dst vselect are accounted for;
    see ge Stage-3 retrospective -- the review-added zero buffer invalidates
    the pre-computed 13 B/elem row).
    """
    if carrier == "int8":
        return 19 if need_ufix else 10
    if carrier == "int16":
        return 16
    return 2 * _ELEM_BYTES[carrier] + 4


def _max_block(carrier, need_ufix):
    """Upper bound on block_size from the UB budget (DESIGN.md section 4.5).

    Conservative x1.7 inflation: fp16/bf16 14456 / fp32,int32 9637 /
    int64 5782 / int16 7228 / int8 11565 / uint8 6086.
    """
    per_elem = _ub_bytes_per_elem(carrier, need_ufix)
    return int(_UB_CAPACITY_BYTES // (per_elem * _UB_INFLATION))


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_gt(a, b):
    """Greater-than reference (source-operator semantics, direct broadcast).

    Matches the source-repo test benchmark (examples/TileOPs/tests/ops/
    test_gt.py: ``ref_program = torch.gt(a, b)`` compared exactly by
    ``_bool_compare``).  Inputs keep their original broadcast shapes; the
    output shape is ``broadcast_shapes(a.shape, b.shape)`` and dtype bool.
    Float follows IEEE STRICT-order semantics: any NaN operand -> False, and
    equal operands (``+0.0 > -0.0``, ``-0.0 > +0.0``, ``a > a``, ``+Inf > +Inf``)
    are all False -- exactly as ``torch.gt``.  Integers compare by native
    value, including the unsigned ``uint8`` ordering.
    Independent of the NPU algorithm (no row/inner indexing, no vcmp, no
    vcast chain, no persistent structure, no int8 carrier, no unsigned
    correction) and runs on CPU without an NPU device.
    """
    return torch.gt(a.detach().cpu(), b.detach().cpu())


# Alias so the generic ``golden_<op>`` gate name resolves for
# op_name == "gt_fwd_kernel" as well.
golden_gt_fwd_kernel = golden_gt


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
    executed once per task rather than once per element -- DESIGN.md R1,
    ~L times cheaper than the source per-element chain).
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
# dtype mapping (DESIGN.md section 0.6 R3 / R4 / R5)
# ---------------------------------------------------------------------------
def _normalize_dtype(dtype, unsigned=False):
    """Return ``(carrier_dtype, need_ufix)`` for a public dtype.

    ``uint8`` is carried as ``int8`` (npuir has no uint8 GM tensor,
    CG-2026-0018) and **requires** the R5 unsigned-domain correction because
    ``>`` is an ordering comparison.  ``bool`` is carried as ``int8``
    bit-identically (value domain {0, 1}, no correction).  ``int8`` takes the
    correction only when the caller sets the explicit ``unsigned`` marker.
    """
    if isinstance(dtype, torch.dtype):
        name = str(dtype).replace("torch.", "")
    else:
        name = str(dtype)
    if name == "uint8":
        return "int8", True
    if name == "bool":
        return "int8", False
    if name == "int8":
        return "int8", bool(unsigned)
    if name in _NATIVE_DTYPES:
        if unsigned:
            raise ValueError(
                f"unsigned=True is only valid for a 1-byte int8 carrier, got {dtype!r}"
            )
        return name, False
    raise ValueError(
        f"unsupported dtype {dtype!r}; expected one of "
        f"{('bool', 'uint8') + _KERNEL_DTYPES}"
    )

# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def gt_fwd_kernel(
    N_total,
    dtype,
    coalesced_shape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    output_dtype=None,
    block_size=_DEFAULT_BLOCK,
    unsigned=False,
):
    """Build the greater-than kernel factory.

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(a, b)`` returns ``y`` (``out_idx=[2]``,
    source-operator contract).  ``y`` is always int8 (0/1): the public bool
    output is restored by the Op layer via ``view(torch.bool)``.
    """
    carrier, need_ufix = _normalize_dtype(dtype, unsigned)
    # Output is always the 1-byte int8 carrier (bool cannot be stored to GM,
    # CG-2026-0022).  ``output_dtype`` is accepted for contract compatibility
    # but does not change the carrier.
    out_dtype = "int8"
    if output_dtype is not None:
        _normalize_dtype(output_dtype)
    # Comparison-domain mapping (DESIGN.md section 0.6 R4): int8/uint8/bool
    # upcast exactly to float16; int16 upcasts exactly to float32 (avoiding
    # the PL-1.11 native-int16 scalarisation); everything else compares
    # natively.
    if carrier == "int16":
        cmp_dtype = "float32"
    elif carrier == "int8":
        cmp_dtype = "float16"
    else:
        cmp_dtype = carrier
    upcast = cmp_dtype != carrier

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
    # adapt automatically (this machine: 20 AI cores -> 40 Vector cores).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    k = len(coalesced_shape)
    L = coalesced_shape[-1]
    outer = list(coalesced_shape[:-1])
    R = math.prod(outer) if outer else 1
    sa, sb = a_strides[-1], b_strides[-1]
    if sa not in (0, 1) or sb not in (0, 1):
        raise ValueError(f"innermost strides must be 0 or 1, got sa={sa}, sb={sb}")

    # Pre-combined Python-constant dispatch flags.  Trap: the npuir script
    # parser only folds a *plain pure-name* `if`; a **nested** `if upcast:`
    # inside `if sa == 1:` is not folded and leaks an undefined buffer
    # reference at trace time (Stage 3 finding, inherited from eq).  Flattening
    # the four operand cases per side into independent pure names keeps every
    # branch statically folded, so the upcast-only staging buffers stay absent
    # on the native path (int64 UB budget, DESIGN.md 4.5).
    a_contig_cast = (sa == 1) and upcast
    a_contig_plain = (sa == 1) and not upcast
    a_scalar_cast = (sa == 0) and upcast
    a_scalar_plain = (sa == 0) and not upcast
    b_contig_cast = (sb == 1) and upcast
    b_contig_plain = (sb == 1) and not upcast
    b_scalar_cast = (sb == 0) and upcast
    b_scalar_plain = (sb == 0) and not upcast

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size=block_size):
        # Host-side UB budget guard (DESIGN.md section 4.5).
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        max_block = _max_block(carrier, need_ufix)
        if block_size > max_block:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for {carrier} "
                f"(need_ufix={need_ufix}, max {max_block} elements, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # block_size-dependent task decomposition, evaluated here (not at
        # factory time) so the compiled kernel always matches the block_size
        # it is invoked with -- the returned callable is cached per
        # block_size (wrapper K11 contract; see TRAP-jit-blocksize-
        # decomposition).  block_size folds to a compile-time constant.
        chunks = T.ceildiv(L, block_size)
        num_logical = R * chunks
        num_kernels = min(num_logical, vector_cores)
        num_local_tasks = T.ceildiv(num_logical, num_kernels)

        @T.prim_func
        def main(
            a: T.Tensor((a_numel,), carrier),
            b: T.Tensor((b_numel,), carrier),
            y: T.Tensor((N_total,), out_dtype),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # a_c / b_c / cond 恒为 (block_size,) 同形——T.vcmp 要求两个
                # 操作数与 cond 形状完全一致（CG-2026-0021），不能像支持广播
                # 的 vadd/vand 那样留一个小形状 operand。
                a_c = T.alloc_ub((block_size,), cmp_dtype)
                b_c = T.alloc_ub((block_size,), cmp_dtype)
                cond = T.alloc_ub((block_size,), "bool")
                yf = T.alloc_ub((block_size,), "float16")
                y8 = T.alloc_ub((block_size,), "int8")
                # 标量广播源：仅当某 operand 末维 stride==0（该 operand 在行内
                # 是标量）时使用，先物化再 T.vbrc 成 (block_size,) 同形。
                a1 = T.alloc_ub((1,), carrier)
                b1 = T.alloc_ub((1,), carrier)
                a1c = T.alloc_ub((1,), cmp_dtype)
                b1c = T.alloc_ub((1,), cmp_dtype)
                # 1 字节/int16 升域的比较中转缓冲。注意：npuir 脚本解析器会
                # **解析所有条件分支体**（即使条件是 Python 常量），因此分支里
                # 引用的 buffer 必须无条件声明，否则报 "Undefined variable"
                # （TRAP-conditional-alloc-parse）。这里无条件声明 a_ub/b_ub，
                # 原生 dtype 路径上它们是死代码，由编译器 DCE 消除（实测 int64
                # @4096 可编译，说明未用缓冲被消除，未破坏 DESIGN.md 4.5 预算）。
                a_ub = T.alloc_ub((block_size,), carrier)
                b_ub = T.alloc_ub((block_size,), carrier)
                # uint8 无符号域修正链（R5）专用 buffer：zero 是与常量 0 比较
                # 所需的同形零操作数（T.vcmp 无标量形，CG-2026-0021，
                # REVIEW.md 维度 1/7 建议——必须物化，否则 trace 期报错）；
                # m 为 s<0 掩码；sp = s+256；u_a/u_b 为两条 operand 修正后的
                # 真无符号值。同样无条件声明、常量守卫使用（非 uint8 路径 DCE 掉）。
                zero = T.alloc_ub((block_size,), cmp_dtype)
                m = T.alloc_ub((block_size,), "bool")
                sp = T.alloc_ub((block_size,), cmp_dtype)
                u_a = T.alloc_ub((block_size,), cmp_dtype)
                u_b = T.alloc_ub((block_size,), cmp_dtype)

                if need_ufix:
                    # 常量 0 必须物化为同形 buffer：T.vcmp 无标量操作数形
                    # （CG-2026-0021）。每核只填一次，不随 task 变化。
                    T.vbrc(T.float16(0.0), zero)

                # persistent 核内串行循环：边界为编译期静态值（num_local_tasks
                # 由 num_logical/num_kernels 常量除得），grid-stride 映射
                # task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local_tasks):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        row = task // chunks
                        inner_off = (task % chunks) * block_size
                        # 行 base 标量：每 task 仅一次 unrolled divmod（非每元素），
                        # 末维仅在 stride==1 时贡献 inner_off。
                        a_base = _row_base(row, outer, a_strides) + (
                            inner_off if sa == 1 else 0
                        )
                        b_base = _row_base(row, outer, b_strides) + (
                            inner_off if sb == 1 else 0
                        )
                        tail = T.min(block_size, L - inner_off)

                        # --- 物化 a 为 (block_size,) 比较域（R1/R4）---
                        if a_contig_cast:
                            # 1 字节/int16 承载：连续段载入原始窄 dtype，再精确升域。
                            T.copy(a[a_base : a_base + tail], a_ub[0:tail])
                            T.vcast(a_ub, a_c, round_mode="rint")
                        if a_contig_plain:
                            T.copy(a[a_base : a_base + tail], a_c[0:tail])
                        # a 行内标量：单元素搬入 + vbrc 广播。升域路径先升域
                        # 再 vbrc（同时满足 vcmp 不支持 int8 与规避 int8-dst
                        # vbrc 运行期挂死 TRAP-vbrc-int8）。
                        if a_scalar_cast:
                            T.copy(a[a_base : a_base + 1], a1[0:1])
                            T.vcast(a1, a1c, round_mode="rint")
                            T.vbrc(a1c, a_c)
                        if a_scalar_plain:
                            T.copy(a[a_base : a_base + 1], a1[0:1])
                            T.vbrc(a1, a_c)

                        # --- a 的无符号域修正（R5，仅 uint8）---
                        # u = s + 256*(s<0)：把 int8 承载值还原成真 uint8 值
                        # （0..255）。> 排序对符号敏感，不修正则在字节 >=128
                        # 时结果错误（E-NAIVE ~50% 不一致）。全程 f16 域且精确。
                        # 注意：相等对上 > 恒 False（严格序），修正后仍如此。
                        if need_ufix:
                            T.vcmp(a_c, zero, m, "lt")
                            T.vadd(a_c, 256.0, sp)
                            T.vselect(m, sp, a_c, u_a)

                        # --- 物化 b 为 (block_size,) 比较域（同上）---
                        if b_contig_cast:
                            T.copy(b[b_base : b_base + tail], b_ub[0:tail])
                            T.vcast(b_ub, b_c, round_mode="rint")
                        if b_contig_plain:
                            T.copy(b[b_base : b_base + tail], b_c[0:tail])
                        if b_scalar_cast:
                            T.copy(b[b_base : b_base + 1], b1[0:1])
                            T.vcast(b1, b1c, round_mode="rint")
                            T.vbrc(b1c, b_c)
                        if b_scalar_plain:
                            T.copy(b[b_base : b_base + 1], b1[0:1])
                            T.vbrc(b1, b_c)

                        # --- b 的无符号域修正（R5，仅 uint8）---
                        if need_ufix:
                            T.vcmp(b_c, zero, m, "lt")
                            T.vadd(b_c, 256.0, sp)
                            T.vselect(m, sp, b_c, u_b)

                        # --- 同形向量比较 + bool→int8 落盘 & 写回 ---
                        # 比较模式必须是 "gt"（严格序）：相等操作数（含 +0/-0、
                        # +Inf 自身）恒 False、任一 NaN → False。**不得**用补集式
                        # ¬(a<=b) 替代——浮点 NaN 下不等价（E-NAN）。bool 片内
                        # 结果不可落 GM（CG-2026-0022），经 vcast(bool→f16) +
                        # vcast(f16→i8, rint) 转成 0/1 承载。尾块 [tail:] 的陈旧
                        # 数据一并参与向量计算但不写回 GM。
                        if need_ufix:
                            T.vcmp(u_a, u_b, cond, "gt")
                        else:
                            T.vcmp(a_c, b_c, cond, "gt")
                        T.vcast(cond, yf)
                        T.vcast(yf, y8, round_mode="rint")
                        flat0 = row * L + inner_off
                        T.copy(y8[0:tail], y[flat0 : flat0 + tail])

        return main

    return kernel

# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_TORCH_DTYPE = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Random-data bounds (inclusive low, exclusive high) per integer dtype; the
# full two's-complement range is sampled for int8/int16 and a wide range for
# the wider dtypes, with explicit extremes covered by L0-8/L0-9.
_RAND_BOUNDS = {
    "int8": (-128, 128),
    "int16": (-32768, 32768),
    "int32": (-2147483648, 2147483648),
    "int64": (-4611686018427387904, 4611686018427387904),
}

_EXTREMES = {
    "int8": (-128, 127),
    "int16": (-32768, 32767),
    "int32": (-2147483648, 2147483647),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint8": (0, 255),
}

_FACTORY_CACHE = {}

# Per public-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(
    N_total,
    factory_dtype,
    cshape,
    a_strides,
    b_strides,
    a_numel,
    b_numel,
    factory_unsigned=False,
):
    key = (
        N_total,
        factory_dtype,
        cshape,
        a_strides,
        b_strides,
        a_numel,
        b_numel,
        factory_unsigned,
    )
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = gt_fwd_kernel(
            N_total,
            factory_dtype,
            cshape,
            a_strides,
            b_strides,
            a_numel,
            b_numel,
            unsigned=factory_unsigned,
        )
    return _FACTORY_CACHE[key]


def _rand_cpu(shape, dtype_str):
    shape = tuple(shape)
    if dtype_str == "bool":
        return torch.randint(0, 2, shape, dtype=torch.int64).to(torch.bool)
    if dtype_str == "uint8":
        return torch.randint(0, 256, shape, dtype=torch.int64).to(torch.uint8)
    if dtype_str in _RAND_BOUNDS:
        lo, hi = _RAND_BOUNDS[dtype_str]
        return torch.randint(lo, hi, shape, dtype=torch.int64).to(_TORCH_DTYPE[dtype_str])
    return torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype_str])


def _exact_compare(out, ref):
    """Exact bool comparison; returns (max_diff, violations)."""
    if tuple(out.shape) != tuple(ref.shape):
        raise AssertionError(
            f"shape mismatch: out {tuple(out.shape)} != ref {tuple(ref.shape)}"
        )
    if out.dtype != ref.dtype:
        raise AssertionError(f"dtype mismatch: out {out.dtype} != ref {ref.dtype}")
    if torch.equal(out, ref):
        return 0, 0
    o = out.to(torch.int64)
    r = ref.to(torch.int64)
    viol = int((o != r).sum().item())
    max_diff = int((o - r).abs().max().item())
    raise AssertionError(f"exact compare failed: {viol} violation(s), max_diff={max_diff}")


def _gt_corner_pair(N, dtype_str):
    """Build an fp pair covering the gt IEEE STRICT-order corner semantics.

    Covers every case where a strict ``>`` differs from a weak ``>=`` (equal
    operands must be False here, True for ge): NaN on either/both side (must
    be False), +-0 (both orders False), +-Inf (Inf>Inf False), dtype max,
    smallest normal and smallest positive subnormal.  DESIGN.md risk R6 /
    L0-10; it is also the direct counter-check against a copied ``"ge"`` mode.
    """
    td = _TORCH_DTYPE[dtype_str]
    info = torch.finfo(td)
    nan = float("nan")
    inf = float("inf")
    sub = float(info.tiny) * float(info.eps)
    pairs = [
        (0.0, -0.0),      # +0 > -0  -> False (strict order)
        (-0.0, 0.0),      # -0 > +0  -> False (strict order)
        (nan, nan),       # NaN      -> False
        (nan, 1.0),       # NaN      -> False
        (1.0, nan),       # NaN      -> False
        (inf, inf),       # +Inf equal -> False
        (inf, -inf),      # +Inf > -Inf -> True
        (-inf, inf),      # -Inf > +Inf -> False
        (info.max, info.max),  # dtype max, equal -> False
        (info.tiny, info.tiny),  # smallest normal, equal -> False
        (sub, sub),       # smallest positive subnormal, equal -> False
        (-1.0, -2.0),     # ordered -> True
        (2.0, 2.0),       # equal -> False
        (1.0, 2.0),       # ordered -> False
    ]
    la = [p[0] for p in pairs]
    lb = [p[1] for p in pairs]
    reps = N // len(pairs) + 1
    a = torch.tensor(la * reps, dtype=td)[:N].clone()
    b = torch.tensor(lb * reps, dtype=td)[:N].clone()
    return a, b


def _run_case(
    a_shape,
    b_shape,
    dtype_str,
    tag,
    block_size=None,
    storage_view=None,
    a_cpu=None,
    b_cpu=None,
    force_equal=True,
    factory_dtype=None,
    factory_unsigned=False,
):
    """Run one kernel case and compare against the golden (exact bool).

    ``storage_view`` is None for native dtypes, or "bool"/"uint8" to exercise
    the R3 int8-carrier path: the public 1-byte tensors are viewed as int8
    before the kernel.  ``dtype_str`` is the *public* dtype handed to the
    factory ("uint8" selects the R5 unsigned correction, "bool"/"int8" do
    not).  ``factory_dtype``/``factory_unsigned`` allow the R11 carrier-side
    marker form (public data uint8, factory dtype int8 + unsigned=True).  The
    kernel always returns an int8 0/1 carrier, which is viewed back to bool
    before comparing against ``torch.gt``.
    """
    if block_size is None:
        block_size = _DEFAULT_BLOCK
    fdt = factory_dtype or dtype_str

    out_shape, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
    N_total = math.prod(out_shape)
    a_numel = math.prod(a_shape) if len(a_shape) else 1
    b_numel = math.prod(b_shape) if len(b_shape) else 1

    if a_cpu is None:
        a_cpu = _rand_cpu(a_shape, dtype_str)
        b_cpu = _rand_cpu(b_shape, dtype_str)
        # Force some positions equal so a "always true" compare cannot pass.
        # (For strict order those forced-equal lanes must resolve to False.)
        if force_equal and tuple(a_shape) == tuple(b_shape) and a_cpu.numel() > 0:
            af = a_cpu.reshape(-1)
            bf = b_cpu.reshape(-1)
            bf[::3] = af[::3]
    assert b_cpu is not None, "a_cpu and b_cpu must be supplied together"

    ref = golden_gt(a_cpu, b_cpu)
    assert tuple(ref.shape) == tuple(out_shape), (
        f"golden shape {tuple(ref.shape)} != broadcast {tuple(out_shape)}"
    )

    if storage_view:
        # bool/uint8 走 1 字节 int8 承载（npuir 不支持 uint8 GM 张量，
        # CG-2026-0018；位模式不变）。
        a_npu = a_cpu.reshape(-1).view(torch.int8).npu()
        b_npu = b_cpu.reshape(-1).view(torch.int8).npu()
    else:
        a_npu = a_cpu.reshape(-1).npu()
        b_npu = b_cpu.reshape(-1).npu()

    fn = _get_factory(
        N_total, fdt, cshape, a_strides, b_strides, a_numel, b_numel, factory_unsigned
    )
    y = fn(block_size)(a_npu, b_npu)
    # The public result is always bool: restore it from the int8 0/1 carrier
    # exactly as the Op layer does (view is non-zero-is-true).
    out = y.cpu().view(torch.bool).reshape(out_shape)
    assert out.dtype == torch.bool, f"carrier view produced dtype {out.dtype}"

    max_diff, viol = _exact_compare(out, ref)
    stat_key = storage_view if storage_view else dtype_str
    if max_diff > _STATS_MAX_DIFF.get(stat_key, -1):
        _STATS_MAX_DIFF[stat_key] = max_diff

    print(
        f"[{tag}] PASS: a={tuple(a_shape)} b={tuple(b_shape)} "
        f"dtype={('->'.join((storage_view, 'int8')) if storage_view else dtype_str)} "
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
    """L0 gate suite (DESIGN.md section 8.3, fourteen items)."""
    failures = []

    # L0-1: smoke, same-shape, float dtypes (manifest smoke-1k).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-1/{dtype_str}", a_shape=(1024,), b_shape=(1024,), dtype_str=dtype_str, tag="L0-1")

    # L0-2: large same-shape fp32/fp16 (manifest hidden-state-prefill, chunks>1).
    for dtype_str in ("float32", "float16"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            a_shape=(2048, 4096),
            b_shape=(2048, 4096),
            dtype_str=dtype_str,
            tag="L0-2",
        )

    # L0-3: broadcast three modes fp32/fp16 (manifest cnn-feat + row-broadcast
    # + bias-add).
    for dtype_str in ("float32", "float16"):
        for a_shape, b_shape, label in (
            ((16, 256, 56, 56), (256, 1, 1), "cnn-feat"),
            ((4096, 4096), (1, 4096), "row-broadcast"),
            ((2, 64, 128), (1, 1, 128), "bias-add"),
        ):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/{label}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L0-3",
            )

    # L0-4: a-side broadcast (roles swapped, exercises sa == 0).
    for a_shape, b_shape in (((1, 4096), (4096, 4096)), ((2, 1, 3), (2, 3, 3))):
        _try_case(failures, f"L0-4/{a_shape}", a_shape=a_shape, b_shape=b_shape, dtype_str="float32", tag="L0-4")

    # L0-5: scalar broadcast / full scalar (L == 1 both-vbrc degenerate,
    # avoids TRAP-vbrc-same-shape-empty-broadcast since dst > 1).
    _try_case(failures, "L0-5/scalar-broadcast", a_shape=(64, 128), b_shape=(1, 1), dtype_str="float32", tag="L0-5")
    _try_case(failures, "L0-5/full-scalar", a_shape=(1,), b_shape=(1,), dtype_str="float32", tag="L0-5")

    # L0-6: non-divisible tails (single partial tile / tail length not 32B
    # aligned): covers DESIGN.md 6.4 / risk R9.
    for N in (1000, 3136, 4097):
        _try_case(failures, f"L0-6/N={N}", a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="L0-6")

    # L0-7: bool storage path (bool -> int8 view): same-shape, and the
    # manifest per-channel broadcast (sb == 0) -- both force the int8-dst
    # T.vbrc materialisation, but via the f16 upcast (R4).
    for a_shape, b_shape in (((4096,), (4096,)), ((2, 512, 768), (1, 1, 768))):
        a = torch.randint(0, 2, a_shape).bool()
        b = torch.randint(0, 2, b_shape).bool()
        _try_case(
            failures,
            f"L0-7/bool/{a_shape}",
            a_shape=a_shape,
            b_shape=b_shape,
            dtype_str="bool",
            tag="L0-7",
            storage_view="bool",
            a_cpu=a,
            b_cpu=b,
        )

    # L0-8: integer dtype full set (int8 -> f16 upcast; int16 -> f32 upcast;
    # native int32/int64; uint8 -> int8 carrier + unsigned correction).
    for dtype_str in ("int8", "int16", "int32", "int64"):
        _try_case(failures, f"L0-8/{dtype_str}", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L0-8")
    _try_case(failures, "L0-8/uint8", a_shape=(4096,), b_shape=(4096,), dtype_str="uint8", tag="L0-8", storage_view="uint8")
    # integer two's-complement extremes (int16 -> f32 exactness at +-extrema).
    for dtype_str in ("int8", "int16", "int32", "int64"):
        lo, hi = _EXTREMES[dtype_str]
        td = _TORCH_DTYPE[dtype_str]
        pattern = [lo, hi, -1, 0, 1, lo, hi, -1]
        a = torch.tensor(pattern * 32, dtype=td)
        b = torch.tensor(pattern[::-1] * 32, dtype=td)
        _try_case(failures, f"L0-8/extremes/{dtype_str}", a_shape=(256,), b_shape=(256,), dtype_str=dtype_str, tag="L0-8", a_cpu=a, b_cpu=b)
    _try_case(
        failures,
        "L0-8/extremes/uint8",
        a_shape=(256,),
        b_shape=(256,),
        dtype_str="uint8",
        tag="L0-8",
        storage_view="uint8",
        a_cpu=torch.tensor(_EXTREMES["uint8"] * 128, dtype=torch.uint8),
        b_cpu=torch.tensor(_EXTREMES["uint8"][::-1] * 128, dtype=torch.uint8),
    )

    # L0-9: known-equal boundary.  STRICT order (gt): an all-equal pair must
    # be all-False (the ge-kernel signature would give all-True here); a
    # partially-equal pair must be False on the forced-equal lanes.
    a = _rand_cpu((4096,), "int32")
    _try_case(failures, "L0-9/all-equal-int32", a_shape=(4096,), b_shape=(4096,), dtype_str="int32", tag="L0-9", a_cpu=a, b_cpu=a.clone())
    a = _rand_cpu((4096,), "float32")
    b = a.clone()
    b[1::2] = torch.randn(b[1::2].numel(), dtype=torch.float32)
    _try_case(failures, "L0-9/partial-float32", a_shape=(4096,), b_shape=(4096,), dtype_str="float32", tag="L0-9", a_cpu=a, b_cpu=b)

    # L0-10: float IEEE STRICT-order corner semantics (NaN/+-Inf/+-0/subnormal)
    # -- risk R6.  Equal operands, including +0/-0, MUST be False; a weak
    # ">=" mode or a complement implementation would fail here.
    for dtype_str in ("float16", "float32"):
        a, b = _gt_corner_pair(4096, dtype_str)
        _try_case(failures, f"L0-10/{dtype_str}", a_shape=(4096,), b_shape=(4096,), dtype_str=dtype_str, tag="L0-10", a_cpu=a, b_cpu=b)

    # L0-11: persistent core-count switch boundary (num_logical 39/40/41).
    bs = _DEFAULT_BLOCK
    for num_logical in (39, 40, 41):
        N = num_logical * bs
        _try_case(failures, f"L0-11/num_logical={num_logical}", a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="L0-11")

    # L0-12: vcmp same-shape-constraint regression -- `[M,N]x[1,N]` and
    # `[M,N]x[M,1]` are directly expressible on GPU but require explicit
    # same-shape materialisation here (CG-2026-0021).
    for a_shape, b_shape, label in (((64, 512), (1, 512), "row-brc"), ((64, 512), (64, 1), "col-brc")):
        _try_case(failures, f"L0-12/{label}", a_shape=a_shape, b_shape=b_shape, dtype_str="float32", tag="L0-12")

    # L0-13: uint8 unsigned-order regression (gt-specific, R5).  Must include
    # bytes >= 128 (the naive int8 carrier is wrong there, E-NAIVE).
    au = torch.randint(0, 256, (4096,), dtype=torch.int64).to(torch.uint8)
    bu = torch.randint(0, 256, (4096,), dtype=torch.int64).to(torch.uint8)
    bu[::3] = au[::3]  # equal pairs (strict order -> must be False)
    # explicit signedness corner bytes
    au[:6] = torch.tensor([0, 128, 255, 127, 200, 1], dtype=torch.uint8)
    bu[:6] = torch.tensor([128, 0, 1, 255, 100, 200], dtype=torch.uint8)
    _try_case(
        failures,
        "L0-13/uint8-full-range",
        a_shape=(4096,),
        b_shape=(4096,),
        dtype_str="uint8",
        tag="L0-13",
        storage_view="uint8",
        a_cpu=au,
        b_cpu=bu,
    )

    # L0-14: uint8 broadcast + unsigned correction combined (R5 + R1 vbrc
    # materialisation), full byte range including >= 128.
    a14 = torch.randint(0, 256, (2, 64, 128), dtype=torch.int64).to(torch.uint8)
    b14 = torch.randint(0, 256, (1, 1, 128), dtype=torch.int64).to(torch.uint8)
    a14.reshape(-1)[:8] = torch.tensor([128, 255, 127, 0, 200, 1, 129, 254], dtype=torch.uint8)
    b14.reshape(-1)[:8] = torch.tensor([0, 128, 255, 127, 100, 200, 130, 253], dtype=torch.uint8)
    _try_case(
        failures,
        "L0-14/uint8-broadcast",
        a_shape=(2, 64, 128),
        b_shape=(1, 1, 128),
        dtype_str="uint8",
        tag="L0-14",
        storage_view="uint8",
        a_cpu=a14,
        b_cpu=b14,
    )

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (fourteen items)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size)."""
    failures = []

    # dtype x {exact multiple, non-32B-aligned tail, multi-chunk}.
    for dtype_str in ("int8", "int16", "int32", "int64", "float16", "float32", "bfloat16"):
        for N in (8192, 1000, 4097):
            _try_case(failures, f"L1/{dtype_str}/N={N}", a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L1")

    # Broadcast variants across dtypes and shapes.
    for dtype_str in ("int8", "int16", "int32", "float16"):
        for a_shape, b_shape in (
            ((4, 8, 16), (1, 8, 1)),
            ((3, 5), (1, 1)),
            ((2, 3, 4), (2, 3, 4)),
        ):
            _try_case(
                failures,
                f"L1/{dtype_str}/{a_shape}",
                a_shape=a_shape,
                b_shape=b_shape,
                dtype_str=dtype_str,
                tag="L1",
            )

    # int8 upcast + innermost stride-0 broadcast (the f16-vbrc path of R4).
    for a_shape, b_shape in (((4, 8, 16), (1, 8, 1)), ((32, 128), (1, 1))):
        _try_case(failures, f"L1/int8-vbrc/{a_shape}", a_shape=a_shape, b_shape=b_shape, dtype_str="int8", tag="L1")

    # uint8 unsigned-correction + broadcast (R5 with vbrc / both operands).
    for a_shape, b_shape in (((4, 8, 16), (1, 8, 1)), ((16, 256, 56, 56), (256, 1, 1)), ((32, 128), (1, 1))):
        _try_case(failures, f"L1/uint8-vbrc/{a_shape}", a_shape=a_shape, b_shape=b_shape, dtype_str="uint8", tag="L1", storage_view="uint8")

    # bool storage path with stride-0 broadcast (f16 vbrc on the carrier).
    for a_shape, b_shape in (((16, 256, 56, 56), (256, 1, 1)), ((64, 128), (1, 1))):
        a = torch.randint(0, 2, a_shape).bool()
        b = torch.randint(0, 2, b_shape).bool()
        _try_case(failures, f"L1/bool-vbrc/{a_shape}", a_shape=a_shape, b_shape=b_shape, dtype_str="bool", tag="L1", storage_view="bool", a_cpu=a, b_cpu=b)

    # Non-default block sizes within the UB budget.
    _try_case(failures, "L1/int8/N=2^20/bs=8192", a_shape=(2**20,), b_shape=(2**20,), dtype_str="int8", tag="L1", block_size=8192)
    _try_case(failures, "L1/int32/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="int32", tag="L1", block_size=2048)
    _try_case(failures, "L1/int64/N=2^18/bs=1024", a_shape=(2**18,), b_shape=(2**18,), dtype_str="int64", tag="L1", block_size=1024)
    _try_case(failures, "L1/fp32/N=2^20/bs=2048", a_shape=(2**20,), b_shape=(2**20,), dtype_str="float32", tag="L1", block_size=2048)
    _try_case(failures, "L1/uint8/N=2^18/bs=2048", a_shape=(2**18,), b_shape=(2**18,), dtype_str="uint8", tag="L1", storage_view="uint8", block_size=2048)

    # R11 carrier-side marker form: public uint8 data carried as int8 with the
    # explicit unsigned=True factory flag.
    _try_case(
        failures,
        "L1/uint8-carrier-marker",
        a_shape=(4096,),
        b_shape=(4096,),
        dtype_str="uint8",
        tag="L1",
        storage_view="uint8",
        factory_dtype="int8",
        factory_unsigned=True,
    )

    # E2 factory contract: one factory callable serving two block_sizes.
    try:
        dtype_str = "float32"
        a_shape = (8192,)
        _, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, a_shape)
        a = _rand_cpu(a_shape, dtype_str).npu()
        b = a.cpu().clone()
        b[::3] = _rand_cpu((b[::3].numel(),), dtype_str)
        b = b.npu()
        ref = golden_gt(a.cpu(), b.cpu())
        fn = _get_factory(8192, dtype_str, cshape, a_strides, b_strides, 8192, 8192)
        for bs in (4096, 2048):
            out = fn(bs)(a, b).cpu().view(torch.bool)
            _exact_compare(out, ref)
        print("[L1] PASS: E2 factory(block_size) dual-config reuse")
    except AssertionError as exc:
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/E2-reuse", exc))
        print(f"[collected] L1/E2-reuse RUNTIME-ERROR: {exc!r}")

    # UB budget guard + dtype rejection.
    for carrier, bs, uns in (
        ("int64", 8192, False),
        ("int8", 16384, False),
        ("float32", 16384, False),
        ("int8", 8192, True),
        ("int16", 8192, False),
    ):
        try:
            gt_fwd_kernel(4096, carrier, (4096,), (1,), (1,), 4096, 4096, unsigned=uns)(bs)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {carrier} unsigned={uns} block_size={bs} ({exc})")
        else:
            failures.append(
                (f"L1/UB-guard-{carrier}-{uns}", AssertionError(f"{carrier} {bs} not rejected"))
            )
    for bad in ("float64", "float8_e4m3fn", "complex64"):
        try:
            _normalize_dtype(bad)
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
    for N, dtype_str in ((1, "float32"), (2, "int8"), (3, "int64"), (2**22, "int8"), (2**22, "float32")):
        try:
            _run_case(a_shape=(N,), b_shape=(N,), dtype_str=dtype_str, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 4096

    # uint8 all-255 vs all-0 (full-range extremes of the unsigned correction).
    try:
        a = torch.full((N,), 255, dtype=torch.uint8)
        b = torch.zeros(N, dtype=torch.uint8)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="uint8", tag="Boundary-uint8-255v0", storage_view="uint8", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): uint8-255v0: {e}")

    # uint8 boundary bytes 127/128 (the signed-wrap boundary).
    try:
        a = torch.tensor([127, 128, 127, 128] * (N // 4), dtype=torch.uint8)
        b = torch.tensor([128, 127, 128, 127] * (N // 4), dtype=torch.uint8)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="uint8", tag="Boundary-uint8-127-128", storage_view="uint8", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): uint8-127-128: {e}")

    # STRICT order: x > x is False for every value (including int16 extrema);
    # explicit all-False assertion catches a copied weak-order ("ge") kernel.
    try:
        a = torch.tensor([-32768, 32767, 0, -1, 1] * (N // 5 + 1), dtype=torch.int16)[:N]
        max_diff, _ = _run_case(a_shape=(N,), b_shape=(N,), dtype_str="int16", tag="Boundary-int16-self-gt", a_cpu=a, b_cpu=a.clone())
        assert max_diff == 0, "x > x must be False for every int16 value"
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): int16-self-gt: {e}")

    # NaN-heavy float32 (all NaN on one side -> all False).
    try:
        b = _rand_cpu((N,), "float32")
        a = torch.full((N,), float("nan"), dtype=torch.float32)
        _run_case(a_shape=(N,), b_shape=(N,), dtype_str="float32", tag="Boundary-nan", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): nan: {e}")

    # bool all-True storage path: True > True is False (strict order).
    try:
        a = torch.ones((512,), dtype=torch.bool)
        b = torch.ones((512,), dtype=torch.bool)
        _run_case(a_shape=(512,), b_shape=(512,), dtype_str="bool", tag="Boundary-bool-ones", storage_view="bool", a_cpu=a, b_cpu=b)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): bool-ones: {e}")


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
