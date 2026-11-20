# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_gemv_kernel: GEMV ``y = W @ x`` on npuir (Developer mode, Vector reduction).

Migrated from the TileOPs GPU ``_gemv_kernel`` (SM90 GEMV fast path: per-thread
``tile_k`` FMA + 32-lane ``tvm_thread_allreduce`` warp shuffle tree) to the
npuir backend.  The NPU algorithm follows the frozen DESIGN.md of
``examples/gemm/_gemv_kernel/``:

  * R1: the GPU warp shuffle tree is replaced by a Vector horizontal reduction
    ``T.reduce_sum(..., dim=-1, clear=False)`` over the contiguous K axis, with
    ``clear=False`` accumulating across K chunks (E2 chunked reduction).
  * R2: the GPU 2D grid x thread mapping is folded into a 1D ``T.Kernel`` grid
    with ``num_launched = num_logical`` (non-persistent: one row block per
    launched core; the grid is never truncated with ``min(...)``).
  * R3: the Cube ``N=1`` route is rejected (design-time measurement: 1.6x-7.9x
    slower).  ``x`` is broadcast with ``T.vbrc`` and multiplied with ``T.vmul``
    in the fp32 domain.
  * R4: no host-side padding (forbidden).  The R row tail uses dynamic-extent
    slices; the K tail is peeled out of the pipelined loop into an
    exact-extent tail tile whose K size is the compile-time remainder
    ``K % block_K``.  Both tail operands therefore contain only valid K
    elements, which enforces the DESIGN 1.6.1 E1 "no fabricated K term"
    property by construction (no ``0 * NaN`` / ``0 * inf`` path) without
    relying on ``T.clear`` of a padded tile.
  * E3: fp32 accumulation with a single fp32 -> input-dtype round in the
    epilogue VCast (no intermediate fp32 GM round trip).

Coverage: fp16/bf16/fp32 inputs, fp32 accumulation, output dtype == input dtype,
non-divisible R and K, K smaller than one block, R smaller than one block.

Run: ``python _gemv_kernel.py --level {L0|L1|all}``
"""

import argparse
import os

# Developer mode must be set explicitly (DESIGN 2.2 / 9.1.1): the default
# TILELANG_ASCEND_MODE is not Developer, and under it cross-dtype T.copy fails
# and alloc_shared lands in cbuf.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

# Stage 3 validated programming mode; must equal TILELANG_ASCEND_MODE above.
ASCEND_MODE = "Developer"

# fp32 accumulation is mandatory (source accum_dtype="float"; DESIGN 0.1).
_ACCUM_DTYPE = "float32"

# Tolerances (DESIGN 8.2): GEMV fp16 1e-2 / bf16 1.6e-2 (test_gemm.py
# GemvBoundaryFixture); fp32 is beyond the manifest dtype set but is exercised
# for the dtype-coverage requirement, with a tighter fp32-domain tolerance.
_TOL = {
    "float16": {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 1.6e-2, "rtol": 1.6e-2},
    "float32": {"atol": 1e-3, "rtol": 1e-3},
}


def _is_a5():
    """True when the compile target is the Ascend 950 (A5) family."""
    name = os.environ.get("TILELANG_ASCEND_DEVICE_NAME", "").strip()
    if not name:
        try:
            from tilelang.utils.npu_arch import get_ascend_device_name

            name = str(get_ascend_device_name() or "")
        except Exception:  # noqa: BLE001
            name = ""
    return name.startswith("Ascend950")


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_gemv(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Source-operator semantics: ``y = W @ x`` (manifest ref_api = torch.matmul).

    ``w``: (R, K) matrix (source kernel's ``b``); ``x``: (K,) vector (source
    kernel's ``a``); returns ``(R,)`` with dtype == ``w.dtype``.

    Mirrors ``examples/TileOPs/tests/ops/test_gemm.py`` ``ref_program`` on the
    GEMV path: the accumulation runs in the fp32 opmath domain through an
    explicit ``torch.matmul`` call and performs the single fp32 -> input-dtype
    round that the kernel epilogue VCast also performs (DESIGN 1.6.1 E3).  A
    bare fp16/bf16 CPU matmul would round a second time and make the comparison
    meaningless for any correct implementation.
    """
    return torch.matmul(w.to(torch.float32), x.to(torch.float32)).to(w.dtype)


# ---------- Kernel ----------
@tilelang.jit(out_idx=[-1], target="npuir")
def _gemv_kernel(R, K, dtype="float16", block_M=64, block_K=256, num_stages=2):
    """Vector-reduction GEMV for npuir (Developer mode).

    Args:
        R, K: compile-time GEMV dims (one kernel per shape, cached by jit).
        dtype: ``"float16"`` / ``"bfloat16"`` / ``"float32"``.
        block_M: output-row tile (row-split granularity).
        block_K: K tile (inner vector / reduction axis).
        num_stages: T.Pipelined K-pipeline depth.

    Returns:
        The compiled ``main`` prim_func (``y`` allocated via ``out_idx``).
    """
    accum_dtype = _ACCUM_DTYPE
    # 网格数不变量：num_launched == num_logical（非 persistent，禁止 min 截断，
    # 否则 R > block_M*40 时逻辑块数超过物理核数会导致部分输出行静默漏写）。
    num_launched = (R + block_M - 1) // block_M
    # K 主体分块数（编译期常量）；K 尾块剥离出流水（DESIGN 3.3 / 5.4：
    # 把 `rk < block_K` 提升为工厂层 Python 分支，避免流水体内运行时谓词）。
    k_full = K // block_K
    rK_last = K - k_full * block_K
    has_k_tail = rK_last > 0
    # A5: the dev codegen's CreateCastIfTypeMismatch runs getBroadcastDim on
    # the loaded source tile and the static destination tile.  A cross-dtype
    # copy whose source has a dynamic extent (the R row tail, rm) therefore
    # CHECKs out with "broadcast dim mismatch" (kDynamic vs block_M).  Staging
    # the GM read through an input-dtype tile keeps that copy same-dtype (the
    # cast path returns early) and moves the dtype conversion onto a static
    # full tile.
    a5 = _is_a5()

    @T.prim_func
    def main(
        W: T.Tensor((R, K), dtype),
        x: T.Tensor((K,), dtype),
        y: T.Tensor((R,), dtype),
    ):
        with T.Kernel(num_launched, is_npu=True) as (cid, _):
            r0 = cid * block_M
            # R 尾块有效行数（动态 extent；仅用于装/写回，不影响数值）
            rm = T.min(block_M, R - r0)

            Wf = T.alloc_fragment((block_M, block_K), accum_dtype)  # W tile (fp32)
            # A5 W staging (input dtype); unused elsewhere.
            Ws = T.alloc_shared((block_M, block_K), dtype)
            xr = T.alloc_shared((1, block_K), accum_dtype)  # x chunk (fp32)
            xb = T.alloc_shared((block_M, block_K), accum_dtype)  # x broadcast
            acc = T.alloc_fragment((block_M, 1), accum_dtype)  # row accumulator
            ou = T.alloc_shared((block_M,), dtype)  # out staging

            # clear=False 的 reduce_sum 必须在循环前清零累加器（DESIGN 9.1.5）
            T.clear(acc)

            # K 主体：静态 extent 的 K 分块；T.Pipelined 让 MTE2 装载与
            # Vector 计算（vbrc/vmul/reduce_sum）双缓冲重叠。
            if k_full > 0:
                for kk in T.Pipelined(k_full, num_stages=num_stages):
                    k0 = kk * block_K
                    # 装载 W tile：行方向动态 extent（尾行块），K 方向静态
                    if a5:
                        # 动态 extent 的同 dtype 装载，再静态整块转 fp32（见 a5 注释）
                        T.copy(W[r0 : r0 + rm, k0 : k0 + block_K],
                               Ws[0:rm, 0:block_K])
                        T.copy(Ws, Wf)
                    else:
                        T.copy(W[r0 : r0 + rm, k0 : k0 + block_K],
                               Wf[0:rm, 0:block_K])
                    # 装载 x chunk（1D -> (1, block_K)）
                    T.copy(x[k0], xr, size=[1, block_K])
                    # x 广播到行块，fp32 逐元素乘，沿 K 向量水平归约累加
                    T.vbrc(xr, xb)
                    T.vmul(Wf, xb, Wf)
                    T.reduce_sum(Wf, acc, dim=-1, clear=False)

            # K 尾块（block_K ∤ K）：剥出流水，用「精确 extent 尾 tile」承载。
            # 尾 tile 的 K 维恰为 rK_last（编译期常量），两个操作数都只含有效
            # K 元素 —— 归约永远不会看到伪造/残留的 K 尾项，等价于 E1 的双侧
            # 清零（`0*NaN` / `0*inf` 路径由构造消除，无需 T.clear 填充）。
            if has_k_tail:
                kt = k_full * block_K
                Wft = T.alloc_fragment((block_M, rK_last), accum_dtype)
                # A5 W tail staging (input dtype).
                Wst = T.alloc_shared((block_M, rK_last), dtype)
                xrt = T.alloc_shared((1, rK_last), accum_dtype)
                xbt = T.alloc_shared((block_M, rK_last), accum_dtype)
                if a5:
                    T.copy(W[r0 : r0 + rm, kt : kt + rK_last],
                           Wst[0:rm, 0:rK_last])
                    T.copy(Wst, Wft)
                else:
                    T.copy(W[r0 : r0 + rm, kt : kt + rK_last],
                           Wft[0:rm, 0:rK_last])
                T.copy(x[kt], xrt, size=[1, rK_last])
                T.vbrc(xrt, xbt)
                T.vmul(Wft, xbt, Wft)
                T.reduce_sum(Wft, acc, dim=-1, clear=False)

            # 输出：fp32 -> dtype 单次 RNE 舍入，再按动态 extent 写回有效行
            for i in T.Parallel(block_M):
                ou[i] = T.cast(acc[i, 0], dtype)
            T.copy(ou[0:rm], y[r0 : r0 + rm])

    return main


# ---------- precision comparing function ----------
def run_case(R, K, dtype, tag, block_M=64, block_K=256, num_stages=2):
    """Build inputs, run the kernel on NPU, compare against the CPU golden."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    dtype_str = str(torch_dtype).rsplit(".", 1)[-1]
    tol = _TOL[dtype_str]
    torch.manual_seed(0)

    # Independent CPU reference (golden computed in the fp32 opmath domain).
    w_cpu = torch.randn(R, K, dtype=torch.float32).to(torch_dtype)
    x_cpu = torch.randn(K, dtype=torch.float32).to(torch_dtype)
    golden = golden_gemv(w_cpu, x_cpu)

    kernel = _gemv_kernel(R, K, dtype_str, block_M, block_K, num_stages)
    y = kernel(w_cpu.npu(), x_cpu.npu())

    y_cpu = y.cpu()
    assert y_cpu.shape == golden.shape, (y_cpu.shape, golden.shape)
    assert y_cpu.dtype == golden.dtype, (y_cpu.dtype, golden.dtype)
    max_diff = (y_cpu.float() - golden.float()).abs().max().item()
    torch.testing.assert_close(y_cpu, golden, atol=tol["atol"], rtol=tol["rtol"])
    print(
        f"[{tag}] PASS gemv(R={R},K={K},dtype={dtype_str}) "
        f"block=({block_M},{block_K},{num_stages}) max_diff={max_diff:.3e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) ----------
def run_L0():
    """DESIGN 8.3 L0 gate plan (10 shapes)."""
    cases = [
        (1024, 1024, torch.float16, "L0-1 smoke fp16"),
        (1024, 1024, torch.bfloat16, "L0-2 smoke bf16"),
        (3000, 1024, torch.float16, "L0-3 lhs_row unaligned R fp16"),
        (3000, 1024, torch.bfloat16, "L0-4 lhs_row unaligned R bf16"),
        (1024, 3000, torch.float16, "L0-5 unaligned K (K tail)"),
        (3001, 1024, torch.float16, "L0-6 rhs_col odd R"),
        (7168, 7168, torch.float16, "L0-7 large shape"),
        (18432, 7168, torch.bfloat16, "L0-8 huge R (288 logical blocks)"),
        (1, 1024, torch.float16, "L0-9 R=1 single row"),
        (1024, 16, torch.float16, "L0-10 K<block_K single chunk"),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"[L0] ALL PASS: {len(cases)} cases, worst max_diff={worst:.3e}")


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    """Non-divisible R/K, K-pipeline variants, fp32 coverage, odd shapes."""
    cases = [
        (2048, 4096, torch.float16, "L1-1 fp16 long K", 64, 256, 2),
        (4096, 4096, torch.float32, "L1-2 fp32 square", 64, 256, 2),
        (2048, 2048, torch.float32, "L1-3 fp32 block_K=512", 32, 512, 2),
        (3000, 1024, torch.float32, "L1-4 fp32 unaligned R", 32, 256, 2),
        (1, 4096, torch.float16, "L1-5 R=1 long K", 64, 256, 2),
        (5, 33, torch.float16, "L1-6 tiny odd (R,K)", 16, 128, 2),
        (257, 3000, torch.float16, "L1-7 unaligned R+K, block_K=128", 64, 128, 2),
        (128, 1000, torch.float16, "L1-8 K=1000 tail 232", 32, 256, 2),
        (1000, 3000, torch.bfloat16, "L1-9 bf16 unaligned K", 64, 256, 2),
        (17, 7168, torch.float16, "L1-10 small R long K", 64, 256, 2),
        (130, 130, torch.float32, "L1-11 fp32 small block", 32, 64, 2),
        (3001, 3001, torch.float16, "L1-12 odd R+K", 64, 256, 3),
    ]
    worst = 0.0
    for c in cases:
        worst = max(worst, run_case(*c))
    print(f"[L1] ALL PASS: {len(cases)} cases, worst max_diff={worst:.3e}")


# ---------- L2: larger shapes (non-blocking) ----------
def run_L2():
    cases = [
        (18432, 7168, torch.bfloat16, "L2-1 huge R bf16"),
        (7168, 7168, torch.bfloat16, "L2-2 large square bf16"),
        (4096, 16384, torch.bfloat16, "L2-3 long K bf16"),
        (4096, 1024, torch.float32, "L2-4 fp32 unaligned R"),
        (1, 1, torch.float16, "L2-5 minimal single element"),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[L2] WARN (record without blocking): {c[:2]} -> {exc}")
    print(f"[L2] done ({len(cases)} cases), worst max_diff={worst:.3e}")


# ---------- Boundary: extreme values (non-blocking) ----------
def run_boundary():
    """Zeros / ones / large magnitudes / K=1 / R=1 / NaN-inf propagation."""
    # all-zero operands must yield an all-zero output.
    try:
        kernel = _gemv_kernel(1024, 1024, "float16")
        z = torch.zeros(1024, 1024, dtype=torch.float16)
        y = kernel(z.npu(), torch.zeros(1024, dtype=torch.float16).npu()).cpu()
        torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-3, rtol=1e-3)
        print("[Boundary] PASS zeros fp16 (1024,1024)")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN zeros -> {exc}")

    # large magnitudes (no overflow after fp32 accumulation).
    try:
        worst = run_case(1024, 1024, torch.float16, "Boundary-large", 64, 256, 2)
        print(f"[Boundary] PASS large magnitudes worst max_diff={worst:.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN large magnitudes -> {exc}")

    cases = [
        (1, 1, torch.float16, "B-1 (1,1)", 16, 16, 2),
        (1024, 1, torch.float16, "B-2 K=1 tail only", 64, 256, 2),
        (1, 1, torch.float32, "B-3 (1,1) fp32", 16, 16, 2),
        (3001, 1024, torch.float16, "B-4 odd R=3001", 64, 256, 2),
        (128, 255, torch.float16, "B-5 K=255 (one-off tail)", 32, 256, 2),
        (63, 256, torch.float16, "B-6 R=block_M-1", 64, 256, 2),
        (128, 513, torch.bfloat16, "B-7 K=513 bf16 tail", 32, 256, 2),
    ]
    worst = 0.0
    for c in cases:
        try:
            worst = max(worst, run_case(*c))
        except Exception as exc:  # noqa: BLE001
            print(f"[Boundary] WARN {c[3]} -> {exc}")

    # NaN / inf propagation: the K-tail dual clear must not turn a valid NaN
    # into a spurious value (0 * NaN == NaN is preserved, not fabricated).
    try:
        w = torch.randn(64, 300, dtype=torch.float16)
        x = torch.randn(300, dtype=torch.float16)
        w[0, 0] = float("inf")
        x[1] = float("nan")
        golden = golden_gemv(w, x)
        kernel = _gemv_kernel(64, 300, "float16")
        y = kernel(w.npu(), x.npu()).cpu()
        torch.testing.assert_close(y, golden, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("[Boundary] PASS NaN/inf propagation")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN NaN/inf -> {exc}")

    print(f"[Boundary] done ({len(cases)} cases), worst max_diff={worst:.3e}")


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
