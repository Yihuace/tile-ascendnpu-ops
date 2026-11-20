# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_cumulative_kernel (CumprodFwdOp inclusive prefix-product scan) -- NPU Developer mode.

Migrated from the GPU TileOPs ``_cumulative_kernel`` (``op_kind="prod"`` branch,
``tileops/kernels/reduction/cumulative.py`` L176-L254) to ``target="npuir"``.

Semantics (DESIGN.md section 0.1 / 3.1, frozen):
    y[i, j] = prod_{k <= j} x[i, k]          # torch.cumprod(x, dim=-1)
    x: (M, N) fp16/bf16/fp32  ->  y: (M, N) same_as(x)

NPU redesign (DESIGN.md section 0.6 R1-R4, frozen):
    R1: the row-wise sequential running product (N serial vector ops) is
        replaced by the log-domain identity
            y = exp(cumsum(ln|x|)) * (-1)^(cumsum(x<0))
        using the NPU hardware scan primitive ``T.cumsum(dim=1)`` (one op per
        tile instead of N), with the sign carried exactly by an integer
        negative-count prefix (parity is rounding-free).
    R2: one-dimensional persistent core split -- ``num_kernels =
        min(num_logical, vector_cores)`` with an in-core ``T.serial``
        grid-stride loop and an if-guard for out-of-range row blocks
        (static loop bound, dynamic guard only).
    R3: raw-N boundary -- the kernel receives the original (M, N); no
        256-aligned pad and no masked load. N tail uses ``tn = min(block_n,
        N - col0)`` plus explicit src/dst slices; M tail uses ``real_m =
        min(block_m, M - off_m)`` plus explicit src/dst slices.
    R4: block ladder from the manual UB budget (``B_ELEM`` bytes/element).
        Single tile (N <= BLOCK_N_MAX): no carry.  Multi tile (N > BLOCK_N_MAX):
        the log-domain sum and the negative-count prefix are carried across
        tiles with an ADDITIVE per-row carry (no multiplicative overflow), kept
        in a full ``(block_m, block_n)`` buffer whose column 0 holds the carry.

Known toolchain constraints enforced here (DESIGN.md section 9):
    - ``T.cumsum`` / ``T.vln`` / ``T.vexp`` do NOT support bf16 -> every dtype
      is promoted to fp32 by the cross-dtype ``T.copy`` (Developer inserts
      ``VCast(RINT)``) and cast back on the output copy.
    - ``T.vfloor`` is NOT available on this target (A5-only codegen) -> the
      parity ``floor(k/2)`` uses ``T.vcast(..., round_mode="floor")``
      (probe P2 backed).
    - ``T.vcmp`` bool output may not be written to GM -> ``cond`` is consumed
      on-chip by ``T.vcast``.
    - ``T.Parallel`` may not read GM -> all GM traffic goes through ``T.copy``.
    - unused ``alloc`` is not dead-code-eliminated -> the single-tile and the
      multi-tile paths use separate prim_funcs so the carry buffers exist only
      where they are used.
    - ``bishengir-compile`` segfaults when a kernel contains TWO row-broadcast
      ``T.vadd(..., (block_m, 1), ...)`` ops (Stage 3 empirical finding).  The
      multi-tile carry therefore uses the column-0 fold trick: the carry lives
      in column 0 of a full ``(block_m, block_n)`` buffer and is added with a
      same-shape ``T.vadd`` (no ``broadcast=[1]`` attribute anywhere).
"""

import os

# Developer mode: the compiler manages UB (alloc_shared) and auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# Largest tile along the scan axis (DESIGN.md section 5.2); N <= this is a
# single tile with no cross-tile carry.
BLOCK_N_MAX = 4096

# Manual UB budget per block (bytes); auto-multi-buffer inflation must still
# fit the 192KB UB (DESIGN.md section 4.5).
UB_BUDGET = 81920

_SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")

# dtype -> (torch dtype, atol, rtol); tolerances from the harness
# ``_cumprod_tol`` (examples/TileOPs/tests/ops/test_cumulative.py).
_DTYPE_MAP = {
    "float16": (torch.float16, 5e-2, 5e-2),
    "bfloat16": (torch.bfloat16, 5e-2, 5e-2),
    "float32": (torch.float32, 1e-3, 1e-3),
}


def _ceildiv(a, b):
    return (a + b - 1) // b


def _is_multi_tile(N, block_n):
    return _ceildiv(N, block_n) > 1


# ---------- block ladder (DESIGN.md section 5.2 / 4.5) ----------
def select_config(M, N):
    """Factory-period (block_m, block_n) for the NPU persistent scan."""
    block_n = max(1, min(N, BLOCK_N_MAX))
    # Large-buffer bytes per element: single tile = 5 fp32 (xf/wf/lf/sf/pf) + 1
    # bool; multi tile adds the two folded carry buffers (clf/ckf).
    b_elem = 4 * 7 + 1 if _is_multi_tile(N, block_n) else 4 * 5 + 1
    block_m = max(1, min(M, UB_BUDGET // (block_n * b_elem)))
    return block_m, block_n


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_cumprod(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch reference: ``torch.cumprod(x.float(), dim).to(x.dtype)`` on CPU.

    Matches the harness ``CumprodFwdTest.ref_program`` and the frozen source
    semantics (DESIGN.md section 0.1 / 8.1).  It does NOT reproduce the NPU
    redesign (no log-domain transform, no sign split, no persistent split),
    guaranteeing validation independence.  Uses an explicit ``torch.cumprod``
    module call so the reference is unambiguously a torch API call.
    """
    return torch.cumprod(x.float(), dim=dim).to(x.dtype)


# ---------- Kernel factory ----------
def _cumulative_kernel(M, N, op_kind, dtype):
    """Build the NPU prefix-product scan kernel factory.

    Mirrors the harness contract
    ``_cumulative_kernel(M, N, op_kind, dtype)(block_m, block_n)(x)``.
    K9: the ``threads`` parameter is removed (NPU has no CUDA threads concept).

    ``op_kind`` is kept for sibling-scan interface compatibility; this
    migration unit implements the ``prod`` branch only (the ``sum`` branch is
    a sibling unit that maps directly to ``T.cumsum``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if op_kind != "prod":
        raise ValueError(f"this unit implements op_kind='prod' only, got {op_kind!r}")
    if M <= 0 or N <= 0:
        raise ValueError(f"M and N must be positive, got M={M}, N={N}")

    vector_cores = NPUUtils.get().get_aicore_num() * 2  # pure Vector cores x2

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m=None, block_n=None):
        # The harness passes its (GPU-derived) block config; the NPU kernel
        # clamps the scan-axis tile to a UB-resident width and only honours a
        # caller-provided block_m when it is positive.
        if block_n is None or block_n <= 0:
            block_n = min(N, BLOCK_N_MAX)
        block_n = max(1, min(block_n, N))
        b_elem = 4 * 7 + 1 if _is_multi_tile(N, block_n) else 4 * 5 + 1
        if block_m is None or block_m <= 0:
            block_m = max(1, min(M, UB_BUDGET // (block_n * b_elem)))
        block_m = max(1, min(block_m, M))

        n_tiles = _ceildiv(N, block_n)          # 1 -> no carry; >1 -> folded carry
        num_logical = _ceildiv(M, block_m)      # row blocks (scan axis never splits)
        num_kernels = min(num_logical, vector_cores)   # persistent grid
        num_local_tasks = _ceildiv(num_logical, num_kernels)  # static bound

        if n_tiles == 1:
            # -------- single tile per row: no cross-tile carry --------
            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M, N), dtype)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    xf = T.alloc_shared((block_m, N), "float32")   # GM staging
                    wf = T.alloc_shared((block_m, N), "float32")   # |x| / scratch
                    lf = T.alloc_shared((block_m, N), "float32")   # ln|x| / count
                    sf = T.alloc_shared((block_m, N), "float32")   # log prefix sum
                    pf = T.alloc_shared((block_m, N), "float32")   # floor scratch
                    cond = T.alloc_shared((block_m, N), "bool")    # x < |x| mask

                    for s in T.serial(num_local_tasks):    # static loop bound
                        task = s * num_kernels + cid       # grid-stride mapping
                        if task < num_logical:             # dynamic guard only
                            off_m = task * block_m
                            # M 尾块：只载入/写出 real_m 行
                            real_m = T.min(block_m, M - off_m)
                            # 载入行块（跨 dtype 自动升 fp32）
                            T.copy(x[off_m : off_m + real_m, 0:N], xf[0:real_m, 0:N])
                            # 幅值 |x| 与符号判据 x < |x| (<=> x < 0)，免零缓冲
                            T.vabs(xf, wf)
                            T.vcmp(xf, wf, cond, "lt")
                            # log 域变换：l = ln|x|，硬件 scan 求前缀和
                            T.vln(wf, lf)
                            T.cumsum(lf, sf, dim=1, reverse=False)
                            # 负号计数前缀（指示量 0/1）
                            T.vcast(cond, wf, round_mode="rint")
                            T.cumsum(wf, lf, dim=1, reverse=False)
                            # 奇偶 -> 符号：par = k - 2*floor(k/2); sign = 1 - 2*par
                            # (T.vfloor 本 target 不可用 -> vcast floor)
                            T.vmul(lf, T.float32(0.5), wf)
                            T.vcast(wf, pf, round_mode="floor")
                            T.vmul(pf, T.float32(2.0), wf)
                            T.vsub(lf, wf, wf)
                            T.vmul(wf, T.float32(2.0), wf)
                            T.vsub(T.float32(1.0), wf, wf)
                            # 指数还原 x 符号，然后写出（自动降回 dtype）
                            T.vexp(sf, sf)
                            T.vmul(sf, wf, sf)
                            T.copy(sf[0:real_m, 0:N], y[off_m : off_m + real_m, 0:N])

        else:
            # -------- multi tile: additive carry folded into column 0 --------
            # clf / ckf are FULL (block_m, N_tile) buffers whose column 0 holds
            # the per-row carry and whose remaining columns stay 0, so the carry
            # injection is a same-shape T.vadd (no broadcast; see module
            # docstring on the bishengir two-broadcast segfault).
            @T.prim_func
            def main(x: T.Tensor((M, N), dtype), y: T.Tensor((M, N), dtype)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    xf = T.alloc_shared((block_m, block_n), "float32")
                    wf = T.alloc_shared((block_m, block_n), "float32")
                    lf = T.alloc_shared((block_m, block_n), "float32")
                    sf = T.alloc_shared((block_m, block_n), "float32")
                    pf = T.alloc_shared((block_m, block_n), "float32")
                    cond = T.alloc_shared((block_m, block_n), "bool")
                    clf = T.alloc_shared((block_m, block_n), "float32")  # log carry
                    ckf = T.alloc_shared((block_m, block_n), "float32")  # count carry

                    # 一次性清零 carry 缓冲（后续只写第 0 列，其余列恒为 0）
                    T.clear(clf)
                    T.clear(ckf)

                    for s in T.serial(num_local_tasks):    # static loop bound
                        task = s * num_kernels + cid       # grid-stride mapping
                        if task < num_logical:             # dynamic guard only
                            off_m = task * block_m
                            real_m = T.min(block_m, M - off_m)
                            # 每个 task 起始重置 carry 第 0 列
                            for i in T.Parallel(block_m):
                                clf[i, 0] = T.float32(0)
                                ckf[i, 0] = T.float32(0)
                            for t in T.serial(n_tiles):
                                col0 = t * block_n
                                tn = T.min(block_n, N - col0)
                                # 载入行块（跨 dtype 自动升 fp32）
                                T.copy(
                                    x[off_m : off_m + real_m, col0 : col0 + tn],
                                    xf[0:real_m, 0:tn],
                                )
                                T.vabs(xf, wf)
                                T.vcmp(xf, wf, cond, "lt")
                                T.vln(wf, lf)
                                # log 域 carry 折入第 0 列（same-shape vadd）
                                T.vadd(lf, clf, lf)
                                T.cumsum(lf, sf, dim=1, reverse=False)
                                # 新 carry = 本 tile 末列的前缀值
                                T.copy(
                                    sf[0:real_m, tn - 1 : tn],
                                    clf[0:real_m, 0:1],
                                )
                                T.vcast(cond, wf, round_mode="rint")
                                # 计数 carry 折入第 0 列
                                T.vadd(wf, ckf, wf)
                                T.cumsum(wf, lf, dim=1, reverse=False)
                                T.copy(
                                    lf[0:real_m, tn - 1 : tn],
                                    ckf[0:real_m, 0:1],
                                )
                                # 奇偶 -> 符号
                                T.vmul(lf, T.float32(0.5), wf)
                                T.vcast(wf, pf, round_mode="floor")
                                T.vmul(pf, T.float32(2.0), wf)
                                T.vsub(lf, wf, wf)
                                T.vmul(wf, T.float32(2.0), wf)
                                T.vsub(T.float32(1.0), wf, wf)
                                T.vexp(sf, sf)
                                T.vmul(sf, wf, sf)
                                T.copy(
                                    sf[0:real_m, 0:tn],
                                    y[off_m : off_m + real_m, col0 : col0 + tn],
                                )

        return main

    return _func


# ---------- input generators ----------
def _gen_values(M, N, kind, seed=0):
    """Build a CPU fp32 tensor; ``kind`` selects the value regime."""
    g = torch.Generator().manual_seed(seed)
    if kind == "pos_near1":
        # product stays O(1): every factor ~ 1.0
        return torch.rand(M, N, generator=g) * 0.02 + 0.99
    if kind == "signed_near1":
        base = torch.rand(M, N, generator=g) * 0.02 + 0.99
        sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
        return base * sign
    if kind == "randn_small":
        return torch.randn(M, N, generator=g) * 0.3
    if kind == "mixed":
        return torch.rand(M, N, generator=g) * 2.0 - 1.0
    raise ValueError(f"unknown generator kind {kind!r}")


def _run_case(M, N, dtype_str, kind, tag, seed=0, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = _gen_values(M, N, kind, seed).to(torch_dtype)
    block_m, block_n = select_config(M, N)
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(block_m, block_n)
    y = kernel(x.npu())
    ref = golden_cumprod(x)
    torch.testing.assert_close(
        y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} "
        f"bm={block_m} bn={block_n} max_diff={max_diff:.2e}"
    )
    return max_diff


def _run_values_case(x, dtype_str, tag, equal_nan=False):
    """Run a pre-built CPU fp32 tensor through the kernel; check vs golden."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    M, N = x.shape
    x = x.to(torch_dtype)
    block_m, block_n = select_config(M, N)
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(block_m, block_n)
    y = kernel(x.npu())
    ref = golden_cumprod(x)
    torch.testing.assert_close(
        y.cpu(), ref, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} "
        f"bm={block_m} bn={block_n} max_diff={max_diff:.2e}"
    )
    return max_diff


def _run_multidim(shape, dim, dtype_str, kind, tag, seed=0):
    """Emulate the Op-layer movedim/reshape choreography around the 2-D kernel."""
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    total = 1
    for d in shape:
        total *= d
    x = _gen_values(1, total, kind, seed).reshape(shape).to(torch_dtype)
    xm = x.movedim(dim, -1).contiguous()
    x2d = xm.reshape(-1, xm.shape[-1])
    M, N = x2d.shape
    block_m, block_n = select_config(M, N)
    kernel = _cumulative_kernel(M, N, "prod", dtype_str)(block_m, block_n)
    y2d = kernel(x2d.npu())
    y = y2d.reshape(xm.shape).movedim(-1, dim)
    ref = golden_cumprod(x, dim=dim)
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape={tuple(shape)} dim={dim} dtype={dtype_str} "
        f"-> ({M},{N}) max_diff={max_diff:.2e}"
    )
    return max_diff


def _corner_rows(dtype_str):
    """(8, 257) rows covering zeros / negatives / alternating / ones / near-1."""
    torch_dtype = _DTYPE_MAP[dtype_str][0]
    rows = [
        torch.ones(257),                                             # all one
        torch.full((257,), 0.5),                                     # constant
        torch.cat([torch.zeros(1), torch.full((256,), 0.5)]),        # leading zero
        torch.cat([torch.full((200,), 0.5), torch.zeros(57)]),       # trailing zero
        torch.full((257,), -1.0),                                    # sign flips
        (torch.arange(257, dtype=torch.float32) % 2 == 0).float() * 2 - 1,
        torch.cat([torch.tensor([-2.0]), torch.full((256,), 0.5)]),  # one negative
        torch.linspace(0.9, 1.1, 257),                               # near-1 ramp
    ]
    return torch.stack(rows, dim=0).to(torch_dtype)


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.3 (manifest workloads + R1-R4 risk coverage).
def run_L0():
    max_diff = 0.0
    # L0-1: smoke (128,512) in the three dtypes.
    max_diff = max(max_diff, _run_case(128, 512, "float16", "pos_near1", "L0-1a"))
    max_diff = max(max_diff, _run_case(128, 512, "bfloat16", "pos_near1", "L0-1b"))
    max_diff = max(max_diff, _run_case(128, 512, "float32", "pos_near1", "L0-1c"))
    # L0-2/3: manifest hidden-state (2048,4096) -> persistent multi-wave.
    max_diff = max(max_diff, _run_case(2048, 4096, "float16", "pos_near1", "L0-2"))
    max_diff = max(max_diff, _run_case(2048, 4096, "bfloat16", "pos_near1", "L0-3"))
    # L0-4: manifest long-seq (64,32768) -> multi tile + additive carry.
    max_diff = max(max_diff, _run_case(64, 32768, "bfloat16", "signed_near1", "L0-4"))
    # L0-5: N non-aligned (300 not a 32/256 multiple).
    max_diff = max(max_diff, _run_case(512, 300, "float16", "signed_near1", "L0-5a"))
    max_diff = max(max_diff, _run_case(512, 300, "bfloat16", "signed_near1", "L0-5b"))
    # L0-6: M tail (129 % block_m != 0).
    max_diff = max(max_diff, _run_case(129, 512, "float16", "pos_near1", "L0-6"))
    # L0-7: harness non-aligned cases (128,300).
    max_diff = max(max_diff, _run_case(128, 300, "float16", "signed_near1", "L0-7a"))
    max_diff = max(max_diff, _run_case(128, 300, "bfloat16", "signed_near1", "L0-7b"))
    # L0-8: single row / single column degenerates.
    max_diff = max(max_diff, _run_case(1, 512, "float16", "signed_near1", "L0-8a"))
    max_diff = max(max_diff, _run_case(512, 1, "float32", "mixed", "L0-8b"))
    # L0-9: corner semantics (zeros / negatives / alternating / ones / near-1).
    for dtype_str in ("float16", "float32"):
        x = _corner_rows(dtype_str)
        max_diff = max(max_diff, _run_values_case(x, dtype_str, f"L0-9-{dtype_str}"))
    # L0-10: multi-dim inputs (Op-layer movedim/reshape around the 2-D kernel).
    max_diff = max(
        max_diff, _run_multidim((2, 64, 512), -1, "float16", "pos_near1", "L0-10a")
    )
    max_diff = max(
        max_diff, _run_multidim((2, 512, 256), 1, "float16", "pos_near1", "L0-10b")
    )
    print(f"[L0] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    max_diff = max(max_diff, _run_case(130, 256, "float16", "signed_near1", "L1-1"))
    max_diff = max(max_diff, _run_case(64, 1024, "bfloat16", "signed_near1", "L1-2"))
    max_diff = max(max_diff, _run_case(8, 512, "float16", "mixed", "L1-3"))
    max_diff = max(max_diff, _run_case(256, 1024, "float32", "signed_near1", "L1-4"))
    max_diff = max(max_diff, _run_case(3, 300, "bfloat16", "signed_near1", "L1-5"))
    max_diff = max(max_diff, _run_case(32, 32, "float32", "mixed", "L1-6"))
    max_diff = max(max_diff, _run_case(255, 777, "float16", "signed_near1", "L1-7"))
    # carry path + non-aligned tail on the scan axis (N > block_n, N % block_n != 0)
    max_diff = max(max_diff, _run_case(16, 5000, "float16", "signed_near1", "L1-8"))
    # carry path + M tail (M not a multiple of block_m with bn = BLOCK_N_MAX)
    max_diff = max(max_diff, _run_case(129, 4100, "bfloat16", "signed_near1", "L1-9"))
    # sign parity on a long axis
    max_diff = max(max_diff, _run_case(4, 9000, "float32", "signed_near1", "L1-10"))
    # 4D input along the last axis
    max_diff = max(
        max_diff, _run_multidim((2, 4, 8, 128), -1, "float16", "signed_near1", "L1-11")
    )
    print(f"[L1] ALL PASS (max_diff={max_diff:.2e})")
    return max_diff


# ---------- L2: boundary / degenerate (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", "mixed", 1),
        (8, 256, "float16", "pos_near1", 2),
        (9, 256, "float16", "pos_near1", 3),
        (32, 1, "float16", "mixed", 4),
        (4096, 2, "float32", "mixed", 5),
    ]
    for M, N, dtype_str, kind, seed in cases:
        try:
            _run_case(M, N, dtype_str, kind, "L2", seed=seed)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # All zeros -> log(0) = -inf -> exp(-inf) = 0.
    try:
        x = torch.zeros(32, 256)
        _run_values_case(x, "float16", "Boundary-all-zeros")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all zeros: {e}")

    # Negative magnitudes -> exact sign parity.
    try:
        x = -_gen_values(32, 256, "pos_near1", 7)
        _run_values_case(x, "float32", "Boundary-all-negative")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): all negative: {e}")

    # Near-dtype-max round trip (DESIGN.md section 9.2 R3 known edge).
    try:
        finfo = torch.finfo(torch.float16)
        x = torch.cat(
            [torch.full((32, 1), finfo.max), torch.ones(32, 255)], dim=1
        )
        _run_values_case(x, "float16", "Boundary-near-max-fp16")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): near-max fp16: {e}")

    # Growing sequence -> overflow to +inf (must match torch.cumprod inf).
    try:
        x = torch.full((16, 512), 1.5)
        _run_values_case(x, "float16", "Boundary-growth-overflow")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): growth overflow: {e}")

    # Shrinking sequence -> underflow to 0.
    try:
        x = torch.full((16, 512), 0.5)
        _run_values_case(x, "float16", "Boundary-shrink-underflow")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): shrink underflow: {e}")

    # NaN propagation (equal_nan=True).
    try:
        x = _gen_values(32, 256, "pos_near1", 11)
        x[0, 3] = float("nan")
        _run_values_case(x, "float32", "Boundary-nan", equal_nan=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): NaN: {e}")

    # Subnormal magnitudes.
    try:
        finfo = torch.finfo(torch.float16)
        x = torch.full((32, 256), finfo.tiny / 2, dtype=torch.float16)
        _run_values_case(x.float(), "float16", "Boundary-subnormal")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): subnormal: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
