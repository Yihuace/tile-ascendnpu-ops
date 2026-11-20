# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""silu-and-mul fused gated activation forward (``_make_fused_gated_explicit``).

Migrated from the GPU TileOPs operator ``FusedGatedKernel._make_fused_gated_explicit``
(the ``explicit_parallel`` strategy builder; Op class ``SiluAndMulFwdOp``) to
TileLang ``target="npuir"`` on Ascend NPU.  Programming mode: **Developer**.

Semantics (frozen by DESIGN.md section 0.1):
    x   : packed ``(M, 2N)``  (gate = columns ``[0, N)``, value = columns ``[N, 2N)``)
    y   : ``(M, N)``, dtype ``same_as(x)``
    y[m, n] = silu(x[m, n]) * x[m, N + n],   silu(g) = g / (1 + exp(-g))
Computes in the fp32 domain and rounds back once (``rint``) to the input dtype,
matching the source reference (``F.silu(gate.float()) * value.float()`` then
single round-back).  Pure element-wise: no reduction, no cross-element
dependency, no accumulation-order concern.

NPU redesign (DESIGN.md section 0.5 / 0.6):
    R1: the source per-element scalar chain in ``T.Parallel(threads, npt)`` is
        replaced by a UB-resident v-prefix vector chain.  The selected formula
        (DESIGN.md section 1.6.1, machine-verified 15/15 EQUIV_PASS) is the
        direct-exp form
            y = gate * value / (1 + exp(-gate))
        i.e. 5 vector ops per tile (``vmul(-1)`` -> ``vexp`` -> ``vadd(+1)`` ->
        ``vdiv(silu)`` -> ``vmul(s*v)``) -- the GPU-only ``exp2(-g*log2e)``
        lowering is revoked (no benefit on NPU) and the ``vsigmoid``
        combination path (6 ops) is rejected.  sub-fp32 dtypes add 3 ``vcast``
        ops.
        Association choice ``(gate/d) * value`` rather than ``(gate*value)/d``:
        both are the same real-arithmetic formula and cost the same 5 ops, but
        when ``|gate|`` and ``|value|`` are both near the dtype max the merged
        numerator ``gate*value`` overflows to ``inf`` *and* ``1+exp(-gate)``
        overflows to ``inf`` -> ``inf/inf = NaN``, while the golden (source
        ``silu(gate)*value``) yields a finite 0.  The un-merged association
        ``(gate/d)*value`` reproduces the golden exactly on that corner
        (DESIGN.md 1.6.1 verifier E2 actually exercised this un-merged form).
    R2: the CUDA two-dimensional grid ``(col-block, row)`` becomes a
        one-dimensional ``T.Kernel(num_kernels, is_npu=True)`` with raster
        ``(row-block, chunk)`` flattening and persistent in-core
        ``T.serial(num_local)`` grid-stride scheduling.  ``num_kernels =
        min(num_logical, vector_cores)`` with ``vector_cores =
        NPUUtils.get().get_aivector_core_num()`` (pure-Vector op; this machine
        reports 20 AI cores -> 40 Vector cores).
    R3: the source ``op_func`` scalar closure cannot be vectorised; the SiLU
        math is inlined as the v-prefix chain above.  The factory signature
        keeps ``op_func`` only as a shape marker for the wrapper contract.

fp32 compute domain is mandatory (PL-1.17): ``vexp``/``vmul``/``vdiv`` have no
native bf16 support, and fp16 must match the torch fp32-opmath golden -- hence
``vcast(round_mode="rint")`` up to fp32 and a single ``rint`` round-back.

Factory contract (frozen; wrapper
``examples/TileOPs/tileops/kernels/elementwise/silu_and_mul/silu_and_mul.py``):
    _make_fused_gated_explicit(M, N, dtype, op_func, block_size=1024,
                               block_M=2, output_dtype=None)
        -> kernel(block_size) -> main(x) -> y              (``out_idx=[1]``)

Run the embedded hierarchical tests:
    python _make_fused_gated_explicit.py --level L0
    python _make_fused_gated_explicit.py --level all
"""

import os

# Developer mode: single-stage copy -> vector chain -> copy per task, no manual
# sync points (DESIGN.md section 7).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = [
    "ASCEND_MODE",
    "_make_fused_gated_explicit",
    "golden_silu_and_mul",
    "validate_packed_input",
    "compute_roofline",
]

# Programming mode actually validated by this file (TileOPs harness migration
# requires this module-level literal).
ASCEND_MODE = "Developer"

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608
# Conservative auto-multi-buffer inflation factor (CONST-capacity-910B2C).
_UB_INFLATION = 1.7

# Wrapper K11 default: block_size = GPU threads * npt = 256 * 8 / 128 * 8.
_DEFAULT_BLOCK = 1024
# Row tile height (factory constant, DESIGN.md section 5.2).
_DEFAULT_BLOCK_M = 2

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

# Accuracy standard (DESIGN.md section 8.2, manifest conformance tolerances).
_ATOL_RTOL = {
    "float16": (1e-2, 1e-2),
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


def _active_bytes_per_elem(dtype_str):
    """UB bytes per element over all declared tile buffers (DESIGN.md 4.3/4.5).

    sub-fp32 path: g_ub + v_ub + y_ub (dtype) and g32 + v32 + p32 + t32 + d32
    (fp32) -> 3*E + 5*4.  fp32 path: g_ub + v_ub + y_ub + p32 + t32 + d32 ->
    6*4.  All buffers are declared unconditionally (see kernel note), so the
    budget is charged for the full set.
    """
    e = _ELEM_BYTES[dtype_str]
    if dtype_str in ("float16", "bfloat16"):
        return 3 * e + 5 * 4
    return 6 * e


def _max_tile_elems(dtype_str):
    """Upper bound on block_M * block_size from the UB budget."""
    bpe = _active_bytes_per_elem(dtype_str)
    return int(_UB_CAPACITY_BYTES // (bpe * _UB_INFLATION))


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_silu_and_mul(x):
    """Reference silu-and-mul with packed input (source-operator semantics).

    ``x`` is ``(M, 2N)``; gate = ``x[:, :N]``, value = ``x[:, N:]``;
    returns ``F.silu(gate) * value`` rounded back to ``x.dtype``.  The golden
    explicitly calls the ``torch.nn.functional`` module API (per ``S3-GOLDEN-
    TORCH``) and computes in fp32 (op-math) with a single round-back, exactly
    like the source test's ``ref_program``.  It is independent of the NPU
    algorithm (no v-prefix chain, no persistent split, no tail handling) and
    runs on CPU without an NPU device.
    """
    half = x.shape[1] // 2
    x_f32 = x.detach().cpu().float()
    gate = x_f32[:, :half]
    value = x_f32[:, half:]
    return (torch.nn.functional.silu(gate) * value).to(x.dtype)


# Alias so the generic ``golden_<op>`` gate name resolves for
# op_name == "_make_fused_gated_explicit" as well.
golden__make_fused_gated_explicit = golden_silu_and_mul


# ---------------------------------------------------------------------------
# Op-layer input contract mirror (DESIGN.md section 0.2)
# ---------------------------------------------------------------------------
def validate_packed_input(x):
    """Mirror the Op-layer ``_validate_runtime_input`` packed-input contract.

    Requires a 2-D tensor whose last dimension is even (gate/value split).
    Returns ``N`` (the half width) on success, raises ``ValueError`` otherwise.
    """
    if x.dim() != 2:
        raise ValueError(f"silu_and_mul expects a 2-D (M, 2N) input, got rank {x.dim()}")
    last = x.shape[1]
    if last % 2 != 0:
        raise ValueError(
            f"silu_and_mul expects an even last dimension (M, 2N), got {last}"
        )
    return last // 2


# ---------------------------------------------------------------------------
# Op-layer roofline mirror (DESIGN.md section 0.2 / manifest conformance)
# ---------------------------------------------------------------------------
def compute_roofline(M, N, dtype):
    """Return the manifest roofline tuple ``(6*M*N, 3*M*N*elem_bytes)``."""
    return (6 * M * N, 3 * M * N * _ELEM_BYTES[_dtype_name(dtype)])


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------
def _make_fused_gated_explicit(
    M,
    N,
    dtype,
    op_func=None,
    block_size=_DEFAULT_BLOCK,
    block_M=_DEFAULT_BLOCK_M,
    output_dtype=None,
):
    """Build the silu-and-mul kernel factory (DESIGN.md section 3.3).

    Returns a JIT callable taking a single ``block_size`` which compiles and
    returns ``main``; ``main(x)`` returns ``y`` (``out_idx=[1]``, source
    contract).  ``op_func`` is accepted for wrapper-contract shape but the
    SiLU math is inlined (R3).
    """
    del op_func  # inlined as the v-prefix chain (R3); kept for the signature
    dtype = _dtype_name(dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {_SUPPORTED_DTYPES}")
    out_dtype = _dtype_name(output_dtype) if output_dtype is not None else dtype
    M = int(M)
    N = int(N)
    if M <= 0 or N <= 0:
        raise ValueError(f"M and N must be positive ints, got M={M} N={N}")

    sub_f32 = dtype in ("float16", "bfloat16")  # bf16 has no native v-arith; fp16 needs fp32 golden alignment (PL-1.17)
    block_M = min(int(block_M), M)

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a pure
    # Vector op).  Re-queried per factory call so other deployments adapt.
    vector_cores = NPUUtils.get().get_aivector_core_num()

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size_arg):
        block_N = block_size_arg
        if block_N <= 0:
            raise ValueError(f"block_size must be positive, got {block_N}")
        max_tile = _max_tile_elems(dtype)
        if block_M * block_N > max_tile:
            raise ValueError(
                f"block_M*block_size={block_M * block_N} exceeds the UB budget for "
                f"{dtype} (max {max_tile} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # block_size 相关的任务分解在此处（非工厂期）完成，保证编译产物始终匹配
        # 调用时的 block_size（TRAP-jit-blocksize-decomposition）；trace 期折叠为常量。
        chunks = T.ceildiv(N, block_N)
        m_blocks = T.ceildiv(M, block_M)
        num_logical = m_blocks * chunks
        num_kernels = min(num_logical, vector_cores)          # persistent 分核
        num_local = T.ceildiv(num_logical, num_kernels)       # 编译期静态边界

        @T.prim_func
        def main(x: T.Tensor((M, 2 * N), dtype), y: T.Tensor((M, N), out_dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                # 全部 buffer 无条件声明：npuir 脚本解析器会解析所有分支体（即使
                # sub_f32 是 Python 常量），未声明即引用会报 Undefined variable；
                # 未使用分支的 buffer 由编译器 DCE 掉。
                g_ub = T.alloc_ub((block_M, block_N), dtype)     # gate 段 [n0,n0+bn)
                v_ub = T.alloc_ub((block_M, block_N), dtype)     # value 段 [N+n0,N+n0+bn)
                y_ub = T.alloc_ub((block_M, block_N), out_dtype)  # 回舍结果 / 输出 staging
                g32 = T.alloc_ub((block_M, block_N), "float32")  # gate 升域
                v32 = T.alloc_ub((block_M, block_N), "float32")  # value 升域
                p32 = T.alloc_ub((block_M, block_N), "float32")  # 分子 g*v（亦承载 y32）
                t32 = T.alloc_ub((block_M, block_N), "float32")  # -g -> exp(-g) 原地链
                d32 = T.alloc_ub((block_M, block_N), "float32")  # 分母 1+exp(-g)

                # persistent 核内串行循环：num_local 为编译期静态值，
                # grid-stride 映射 task = i*num_kernels + cid，if 屏蔽越界 task。
                for i in T.serial(num_local):
                    task = i * num_kernels + cid
                    if task < num_logical:
                        mb = task // chunks       # 行块（raster 序）
                        ck = task % chunks        # 列块
                        m0 = mb * block_M
                        n0 = ck * block_N
                        # runtime 尾块：非整除 shape 时逐 task 截断（不 host padding）
                        m_tail = T.min(block_M, M - m0)
                        n_tail = T.min(block_N, N - n0)
                        # 单打包 (M,2N) 上直接取段：gate 段列 [n0,n0+bn)、
                        # value 段列 [N+n0,N+n0+bn)（行 stride 2N，跨段切片）
                        T.copy(x[m0 : m0 + m_tail, n0 : n0 + n_tail], g_ub[0:m_tail, 0:n_tail])
                        T.copy(
                            x[m0 : m0 + m_tail, N + n0 : N + n0 + n_tail],
                            v_ub[0:m_tail, 0:n_tail],
                        )
                        if sub_f32:
                            # sub-fp32：vcast(rint) 升 fp32 域计算，末端单次 rint 舍回
                            T.vcast(g_ub, g32, round_mode="rint")
                            T.vcast(v_ub, v32, round_mode="rint")
                            T.vmul(g32, -1.0, t32)       # t = -g（无 vneg 原语）
                            T.vexp(t32, t32)             # t = exp(-g)
                            T.vadd(t32, 1.0, d32)        # d = 1 + exp(-g)
                            # s = g/d（silu），再 y = s*v：结合序选 (g/d)*v 而非
                            # (g*v)/d，避免 |g|,|v| 同时接近 dtype max 时分子 g*v
                            # 溢出为 inf 且分母亦 inf → inf/inf=NaN（golden 为有限 0）
                            T.vdiv(g32, d32, p32)        # p32 = silu(g)
                            T.vmul(p32, v32, p32)        # p32 = silu(g)*v
                            T.vcast(p32, y_ub, round_mode="rint")
                        else:
                            # fp32：同构链，直接 fp32 域，无中转
                            T.vmul(g_ub, -1.0, t32)
                            T.vexp(t32, t32)
                            T.vadd(t32, 1.0, d32)
                            T.vdiv(g_ub, d32, p32)       # p32 = silu(g)
                            T.vmul(p32, v_ub, y_ub)      # y = silu(g)*v
                        # 向量 op 对整 buffer 计算，仅 [0:m_tail, 0:n_tail] 写回，垃圾不落 GM
                        T.copy(
                            y_ub[0:m_tail, 0:n_tail],
                            y[m0 : m0 + m_tail, n0 : n0 + n_tail],
                        )

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------
_FACTORY_CACHE = {}
_COMPILED_CACHE = {}

# Per public-dtype worst max_diff across the whole run (final summary).
_STATS_MAX_DIFF = {}


def _get_factory(M, N, dtype_str, block_M):
    key = (M, N, dtype_str, block_M)
    if key not in _FACTORY_CACHE:
        _FACTORY_CACHE[key] = _make_fused_gated_explicit(
            M, N, dtype_str, None, block_M=block_M
        )
    return _FACTORY_CACHE[key]


def _get_compiled(M, N, dtype_str, block_M, block_size):
    key = (M, N, dtype_str, block_M, block_size)
    if key not in _COMPILED_CACHE:
        _COMPILED_CACHE[key] = _get_factory(M, N, dtype_str, block_M)(block_size)
    return _COMPILED_CACHE[key]


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
    # fin = positions non-NaN in either operand; assert_close handles +-Inf
    # correctly (inf == inf, inf != finite, inf != -inf).
    fin = ~torch.isnan(out) & ~torch.isnan(ref)
    atol, rtol = _ATOL_RTOL[dtype_str]
    if bool(fin.any()):
        torch.testing.assert_close(
            out[fin].to(torch.float64),
            ref[fin].to(torch.float64),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    both_finite = torch.isfinite(out) & torch.isfinite(ref)
    max_diff = 0.0
    if bool(both_finite.any()):
        o = out[both_finite].to(torch.float64)
        r = ref[both_finite].to(torch.float64)
        max_diff = float((o - r).abs().max().item())
    return max_diff


def _run_case(x_shape, dtype_str, tag, block_size=None, block_M=2, x_cpu=None):
    """Run one silu-and-mul case and compare against the golden.

    ``x_shape`` is the packed ``(M, 2N)`` input shape; the output must be
    ``(M, N)`` with ``dtype == x.dtype``.
    """
    dtype_str = _dtype_name(dtype_str)
    if block_size is None:
        block_size = _DEFAULT_BLOCK
    x_shape = tuple(x_shape)
    M, twoN = x_shape
    N = twoN // 2

    if x_cpu is None:
        x_cpu = _rand_cpu(x_shape, dtype_str)
    x_cpu = x_cpu.to(_TORCH_DTYPE[dtype_str])

    ref = golden_silu_and_mul(x_cpu)
    assert tuple(ref.shape) == (M, N), (
        f"golden shape {tuple(ref.shape)} != expected {(M, N)}"
    )

    fn = _get_compiled(M, N, dtype_str, min(block_M, M), block_size)
    out = fn(x_cpu.npu()).cpu()

    max_diff = _compare(out, ref, dtype_str)
    if max_diff > _STATS_MAX_DIFF.get(dtype_str, -1):
        _STATS_MAX_DIFF[dtype_str] = max_diff
    print(
        f"[{tag}] PASS: x={x_shape} (M={M},N={N}) dtype={dtype_str} "
        f"block={block_size} block_M={min(block_M, M)} max_diff={max_diff:.3e}"
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


def _corner_packed(N, dtype_str):
    """Build a ``(1, 2N)`` packed input covering the silu_and_mul corners.

    Gate half: NaN / +-Inf / +-0 / +-1 / large negative (``exp(-g)`` overflow)
    / dtype max / tiny / subnormal.  Value half: 0 / -0 / +-1 / +-Inf / NaN and
    magnitudes that make the numerator overflow class boundaries.
    """
    td = _TORCH_DTYPE[dtype_str]
    info = torch.finfo(td)
    nan = float("nan")
    inf = float("inf")
    sub = float(info.tiny) * float(info.eps)
    g = [
        nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 8.0, -8.0, 20.0, -20.0,
        info.max, -info.max, info.tiny, -info.tiny, sub, -sub, -88.0, -100.0, 3.0,
    ]
    v = [
        1.0, 1.0, 1.0, -2.0, 2.0, -2.0, -2.0, 0.5, 0.5, 0.0, -0.0,
        info.max, -info.max, sub, -sub, info.tiny, -info.tiny, 2.0, -1.0, nan,
    ]
    reps = N // len(g) + 1
    gate = torch.tensor(g * reps, dtype=torch.float64).to(td)[:N].clone()
    val = torch.tensor(v * reps, dtype=torch.float64).to(td)[:N].clone()
    return torch.cat([gate, val], dim=0).unsqueeze(0)


def _filled_packed(N, gate_val, value_val, dtype_str):
    td = _TORCH_DTYPE[dtype_str]
    g = torch.full((N,), gate_val, dtype=torch.float64).to(td)
    v = torch.full((N,), value_val, dtype=torch.float64).to(td)
    return torch.cat([g, v], dim=0).unsqueeze(0)


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, nine items)."""
    failures = []

    # L0-1: smoke (1024, 2048) across all dtypes (bf16 exercises the fp32
    # transit path that PL-1.17 makes mandatory).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(failures, f"L0-1/smoke/{dtype_str}", x_shape=(1024, 2048),
                  dtype_str=dtype_str, tag="L0-1")

    # L0-2: prefill (2048, 28672) -- chunks=14, persistent split. bf16 forced.
    for dtype_str in ("float16", "bfloat16"):
        _try_case(failures, f"L0-2/prefill/{dtype_str}", x_shape=(2048, 28672),
                  dtype_str=dtype_str, tag="L0-2")

    # L0-3: decode (1, 28672) -- M=1 -> block_M=1, num_logical=14 <= 40.
    _try_case(failures, "L0-3/decode/bf16", x_shape=(1, 28672),
              dtype_str="bfloat16", tag="L0-3")

    # L0-4: ffn (4096, 4096) -- N=2048, chunks=2.
    for dtype_str in ("float16", "bfloat16"):
        _try_case(failures, f"L0-4/ffn/{dtype_str}", x_shape=(4096, 4096),
                  dtype_str=dtype_str, tag="L0-4")

    # L0-5: non-divisible tails (row and column partial tiles; N=1025, N=1).
    _try_case(failures, "L0-5/x=(7,2050)", x_shape=(7, 2050),
              dtype_str="float32", tag="L0-5")
    _try_case(failures, "L0-5/x=(3,2)", x_shape=(3, 2),
              dtype_str="float16", tag="L0-5")

    # L0-6: IEEE corners (equal_nan): NaN / +-Inf / +-0 / exp(-g) overflow /
    # subnormal, fp16 and fp32.
    for dtype_str in ("float16", "float32"):
        x = _corner_packed(2048, dtype_str)
        _try_case(failures, f"L0-6/corners/{dtype_str}", x_shape=(1, 4096),
                  dtype_str=dtype_str, tag="L0-6", x_cpu=x)

    # L0-7: Op-layer I/O contract -- output (M, N), dtype == x.dtype (asserted
    # inside _run_case/_compare); non-square shape to catch axis swaps.
    _try_case(failures, "L0-7/contract", x_shape=(5, 14),
              dtype_str="float32", tag="L0-7")

    # L0-8: roofline contract returns (6*M*N, 3*M*N*elem_bytes).
    try:
        r = compute_roofline(1024, 1024, "float16")
        assert r == (6 * 1024 * 1024, 3 * 1024 * 1024 * 2), f"roofline mismatch {r}"
        r32 = compute_roofline(7, 5, "float32")
        assert r32 == (6 * 7 * 5, 3 * 7 * 5 * 4), f"roofline fp32 mismatch {r32}"
        print("[L0-8] PASS: roofline (6*M*N, 3*M*N*elem_bytes)")
    except AssertionError as exc:
        failures.append(("L0-8/roofline", exc))
        print(f"[collected] L0-8/roofline FAILED: {exc}")

    # L0-9: odd last dimension is rejected by the Op-layer contract.
    try:
        validate_packed_input(torch.zeros(2, 4, dtype=torch.float32))
        try:
            validate_packed_input(torch.zeros(2, 5, dtype=torch.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("odd last dim not rejected")
        try:
            validate_packed_input(torch.zeros(4, dtype=torch.float32))
        except ValueError:
            pass
        else:
            raise AssertionError("1-D input not rejected")
        print("[L0-9] PASS: odd last dim / non-2-D rejected")
    except AssertionError as exc:
        failures.append(("L0-9/validate", exc))
        print(f"[collected] L0-9/validate FAILED: {exc}")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (nine items)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size + contracts)."""
    failures = []

    # dtype x {exact multiple, multi-chunk, non-divisible tail}.
    for dtype_str in _SUPPORTED_DTYPES:
        for x_shape in ((64, 256), (33, 258), (4, 2050)):
            _try_case(failures, f"L1/{dtype_str}/{x_shape}", x_shape=x_shape,
                      dtype_str=dtype_str, tag="L1")

    # block_size variants (within the UB budget) + block_M variants.
    _try_case(failures, "L1/fp16/bs=512", x_shape=(128, 4096),
              dtype_str="float16", tag="L1", block_size=512)
    _try_case(failures, "L1/bf16/bs=2048", x_shape=(128, 4096),
              dtype_str="bfloat16", tag="L1", block_size=2048)
    _try_case(failures, "L1/fp32/bs=2048", x_shape=(128, 4096),
              dtype_str="float32", tag="L1", block_size=2048)
    _try_case(failures, "L1/fp32/blockM=1", x_shape=(129, 200),
              dtype_str="float32", tag="L1", block_M=1)
    _try_case(failures, "L1/fp16/blockM=4", x_shape=(129, 200),
              dtype_str="float16", tag="L1", block_M=4)

    # Factory(block_size) dual-config reuse on one factory instance.
    try:
        dtype_str = "float32"
        x_shape = (64, 512)
        M, twoN = x_shape
        N = twoN // 2
        x = _rand_cpu(x_shape, dtype_str)
        ref = golden_silu_and_mul(x)
        x_npu = x.npu()
        fn = _get_factory(M, N, dtype_str, min(2, M))
        for bs in (1024, 512):
            out = fn(bs)(x_npu).cpu()
            _compare(out, ref, dtype_str)
        print("[L1] PASS: factory(block_size) dual-config reuse")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/factory-reuse", exc))
        print(f"[collected] L1/factory-reuse FAILED: {exc}")

    # UB budget guard + dtype rejection.
    for dtype_str, bs in (("float16", 8192), ("float32", 4096)):
        try:
            _make_fused_gated_explicit(128, 4096, dtype_str, None, block_M=2)(bs)
        except ValueError as exc:
            print(f"[L1] PASS: UB guard rejects {dtype_str} block_size={bs} ({exc})")
        else:
            failures.append(
                (f"L1/UB-guard-{dtype_str}-{bs}", AssertionError(f"{dtype_str} {bs} not rejected"))
            )
    for bad in ("float64", "int32"):
        try:
            _make_fused_gated_explicit(64, 64, bad, None)
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
    """L2: extreme sizes / block configs (warn only, non-blocking)."""
    for x_shape, dtype_str in (
        ((1, 2), "float32"),
        ((2, 2), "float16"),
        ((16, 2), "bfloat16"),
        ((2, 2 * 1024 * 1024), "float16"),
        ((2, 2 * 1024 * 1024), "float32"),
        ((3, 60000), "bfloat16"),
    ):
        try:
            _run_case(x_shape=x_shape, dtype_str=dtype_str, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): x={x_shape} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking)."""
    N = 2048

    # All NaN.
    try:
        x = _filled_packed(N, float("nan"), 1.0, "float32")
        _run_case(x_shape=(1, 2 * N), dtype_str="float32", tag="Boundary-all-nan",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-nan: {e}")

    # All +Inf gate (silu(+inf)=+inf; finite value keeps it finite).
    try:
        x = _filled_packed(N, float("inf"), 2.0, "float32")
        _run_case(x_shape=(1, 2 * N), dtype_str="float32", tag="Boundary-inf-gate",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): inf-gate: {e}")

    # All -Inf gate: silu(-inf)=NaN (torch: -inf * sigmoid(-inf) = NaN).
    try:
        x = _filled_packed(N, float("-inf"), 2.0, "float32")
        _run_case(x_shape=(1, 2 * N), dtype_str="float32", tag="Boundary-ninf-gate",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): ninf-gate: {e}")

    # Large negative gate: exp(-g) overflows fp32 -> silu -> 0-.
    try:
        x = _filled_packed(N, -100.0, 3.0, "float32")
        _run_case(x_shape=(1, 2 * N), dtype_str="float32", tag="Boundary-neg-overflow",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): neg-overflow: {e}")

    # Signed zero gate / value.
    try:
        g = torch.tensor([0.0, -0.0] * (N // 2), dtype=torch.float32)
        v = torch.tensor([-0.0, 0.0] * (N // 2), dtype=torch.float32)
        x = torch.cat([g, v], dim=0).unsqueeze(0)
        _run_case(x_shape=(1, 2 * N), dtype_str="float32", tag="Boundary-signed-zero",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): signed-zero: {e}")

    # Subnormal-heavy fp16.
    try:
        info = torch.finfo(torch.float16)
        sub = float(info.tiny) * float(info.eps)
        g = torch.tensor([sub, -sub, info.tiny, -info.tiny] * (N // 4),
                         dtype=torch.float32).to(torch.float16)
        v = torch.tensor([-sub, sub, info.tiny, sub] * (N // 4),
                         dtype=torch.float32).to(torch.float16)
        x = torch.cat([g, v], dim=0).unsqueeze(0)
        _run_case(x_shape=(1, 2 * N), dtype_str="float16", tag="Boundary-subnormal",
                  x_cpu=x)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): subnormal: {e}")


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
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
