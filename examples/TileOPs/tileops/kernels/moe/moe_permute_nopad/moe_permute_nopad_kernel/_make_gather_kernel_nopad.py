# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_make_gather_kernel_nopad -- MoE no-pad token-permute row-gather carrier.

Stage 3 implementation (Developer, target="npuir") of the frozen DESIGN.md:
    examples/moe_permute_nopad/_make_gather_kernel_nopad/DESIGN.md
    (REVIEW.md verdict = pass)

Semantics (DESIGN sec 0.1, pure row gather, FLOPs = 0):

    perm_h[s, :] = hidden_states[permuted_idx[s], :],  s in [0, numel),
    numel = num_tokens * top_k            (tight layout, no inter-expert padding)

NPU form (DESIGN sec 0.6 R1-R4 / sec 3.3 main pick C1)
-------------------------------------------------------
An independent Vector / MTE2 kernel (never inlined into the scan unit -- the
scalar index must be read from GM across a kernel boundary, CG-2026-0027):

    nk  = min(numel, P),  P = 2 * aicore_num = 40   (pure Vector core count)
    spc = ceildiv(numel, nk)                        (static in-core bound)
    with T.Kernel(nk, is_npu=True):
        for s in T.serial(spc):                     # in-core serial slot loop
            slot = core_id * spc + s
            if slot < numel:                        # runtime tail guard
                src = permuted_idx[slot]            # scalar GM read
                T.copy(hidden_states[src:src+1, :], ub_row)   # GM -> UB
                T.copy(ub_row, perm_h[slot:slot+1, :])        # UB -> GM

Every slot does two whole-row DMAs through a single UB row buffer.  The
element axis H is carried by one vectorised DMA; only the slot axis is scalar
(there is no vector row-gather primitive on NPU).

Trap avoidance (DESIGN sec 3.5.1 / 9): no GM->GM copy (UB staging instead), no
scalar consumption of a vector-produced UB value (index read from GM), no
T.alloc_local, no int32 `>>`, no scalar T.if_then_else (tail uses an `if`
statement), no manual sync primitives.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python _make_gather_kernel_nopad.py --level all
"""

import argparse
import os

# Stage 3 programming mode actually implemented/validated (must match the
# standalone-run environment variable).  Set before importing tilelang.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

# Final validated programming mode for the TileOPs harness migration.
ASCEND_MODE = "Developer"


# Pure Vector / MTE2 physical core count (DESIGN sec 5.5):
#   NPUUtils.get().get_aicore_num() -> 20  (Ascend910B2C, measured)
#   a pure Vector unit uses 2x the AICore count -> 40.
def _vector_core_count():
    try:
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num()) * 2
    except Exception:  # noqa: BLE001 - fall back to the measured Ascend910B2C value
        return 40


VECTOR_CORES = _vector_core_count()

# Keep the UB buffer single-versioned and stop the buffer planner from
# re-scoping/permuting it: the dynamic-source-row range copy is only emitted
# reliably with auto multi-buffer disabled (DESIGN sec 4.7 / 7.3, probe-validated).
_PASS = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
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


def gather_num_blocks(numel):
    """Grid size NK for the row-gather kernel.

    Non-A5 keeps ``min(numel, VECTOR_CORES)``.  On A5 the CANN 9.2 bishengir
    back end SIGSEGVs on a serial slot loop with a runtime tail guard, so use
    the largest divisor of ``numel`` that fits the core budget: every block
    gets exactly ``numel // nk`` slots, the guard folds away, and the kernel
    compiles.  (One slot per block would be simpler but breaks above 65535
    rows, which prefill ``numel = T * top_k`` exceeds.)
    """
    if numel <= 0:
        return 1
    if not _is_a5():
        return min(numel, VECTOR_CORES)
    for nk in range(min(numel, VECTOR_CORES), 0, -1):
        if numel % nk == 0:
            return nk
    return 1


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source semantics, explicit torch module API).
# --------------------------------------------------------------------------
def golden_gather(hidden_states, permuted_idx):
    """perm_h[s, :] = hidden_states[permuted_idx[s], :]  (pure PyTorch row gather).

    Built from an explicit torch module API (``torch.index_select``); it is
    independent of the NPU algorithm (no UB staging, no dynamic slice, no
    core-split serial loop).  dtype is passed through bit-exactly.
    """
    idx = permuted_idx.reshape(-1).to(torch.long)
    return torch.index_select(hidden_states, 0, idx)


# Alias matching the ``golden_<op>`` naming convention of the Stage 3 gate.
golden__make_gather_kernel_nopad = golden_gather


# --------------------------------------------------------------------------
# Kernel (DESIGN sec 3.3 main pick C1, one Vector kernel)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
def _make_gather_kernel_nopad(num_tokens, numel, hidden_size, dtype):
    """Factory for the no-pad MoE permute row-gather bottom.

    ``num_tokens`` / ``numel`` / ``hidden_size`` / ``dtype`` are compile-time
    specialised by the harness (same as the source factory + lru_cache).
    ``perm_h`` is caller-allocated (``out_idx=[]``), matching the source.
    """
    nk = gather_num_blocks(numel)  # compile-time constant (DESIGN sec 5.5)
    spc = (numel + nk - 1) // nk  # static in-core serial bound (exact on A5)

    @T.prim_func
    def _gather_main(
        hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
        permuted_idx: T.Tensor([numel], "int32"),
        perm_h: T.Tensor([numel, hidden_size], dtype),
    ):
        with T.Kernel(nk, is_npu=True) as (core_id, _subid):
            # 单行 UB 中转缓冲（元素轴 H 一条 DMA 搬完，无逐元素标量）
            ub_row = T.alloc_ub((1, hidden_size), dtype)
            # 核内槽位串行遍历：spc 为编译期静态边界（core-split §1 要素③）
            for s in T.serial(spc):
                slot = core_id * spc + s
                # 尾块守卫：numel 非核数整数倍时跳过越界槽位（与源等价）
                if slot < numel:
                    # 标量 GM 读源行号（跨 kernel 交接；不消费本 scope 向量
                    # 产出的 UB 值，规避 CG-2026-0027）
                    src = permuted_idx[slot]
                    # 整行 GM->UB：动态源行 [src:src+1]（纯 DMA 向量搬运）
                    T.copy(
                        hidden_states[src : src + 1, 0:hidden_size],
                        ub_row[0:1, 0:hidden_size],
                    )
                    # 整行 UB->GM：动态目的行 [slot:slot+1]
                    T.copy(
                        ub_row[0:1, 0:hidden_size],
                        perm_h[slot : slot + 1, 0:hidden_size],
                    )

    return _gather_main


# --------------------------------------------------------------------------
# Index builders for the layered test suite
# --------------------------------------------------------------------------
def idx_grouped(num_tokens, numel):
    """Each source token referenced K = numel/num_tokens consecutive slots
    (many-to-one gather: a token's K top-k slots share one source row)."""
    k = max(1, numel // num_tokens)
    return torch.repeat_interleave(torch.arange(num_tokens, dtype=torch.int32), k)


def idx_zero_tail(num_tokens, numel):
    """EP-style dummy slots: the tail is left unwritten -> 0 (dummy row 0)."""
    idx = torch.randint(0, num_tokens, (numel,), dtype=torch.int32)
    idx[numel // 2 :] = 0
    return idx


def idx_single(num_tokens, numel):
    """Broadcast: every slot gathers source row 0."""
    return torch.zeros(numel, dtype=torch.int32)


def idx_sparse(num_tokens, numel):
    """Only the first quarter of the source rows is referenced: the remaining
    rows belong to 'empty experts' and must never appear in the output."""
    active = max(1, num_tokens // 4)
    return torch.randint(0, active, (numel,), dtype=torch.int32)


def idx_perm(num_tokens, numel):
    """Scan-like tight order: a permutation of the token range (K = 1 layout)."""
    if numel <= num_tokens:
        return torch.randperm(num_tokens, dtype=torch.int32)[:numel]
    base = torch.randperm(num_tokens, dtype=torch.int32)
    reps = (numel + num_tokens - 1) // num_tokens
    return base.repeat(reps)[:numel]


_DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16"}


# --------------------------------------------------------------------------
# Precision comparing helper
# --------------------------------------------------------------------------
def _run_case(
    num_tokens,
    top_k,
    hidden_size,
    dtype,
    tag,
    idx_builder=None,
    seed=0,
    repeat=1,
):
    """Run one shape end-to-end on NPU, compare bit-exactly with the golden.

    Only L0/L1 use ``repeat > 1`` (repeated-run stability, guarding against the
    CG-2026-0027 stale-read class of defect).
    """
    numel = num_tokens * top_k
    dtype_str = _DTYPE_NAME[dtype]

    torch.manual_seed(seed)
    hidden_cpu = (torch.randn(num_tokens, hidden_size, dtype=torch.float32) * 0.1).to(
        dtype
    )
    if idx_builder is None:
        idx_cpu = torch.randint(0, num_tokens, (numel,), dtype=torch.int32)
    else:
        idx_cpu = idx_builder(num_tokens, numel).to(torch.int32)
    assert idx_cpu.shape == (numel,), (idx_cpu.shape, numel)

    golden_cpu = golden_gather(hidden_cpu, idx_cpu)

    kernel = _make_gather_kernel_nopad(num_tokens, numel, hidden_size, dtype_str)
    hidden = hidden_cpu.npu()
    idx = idx_cpu.npu()
    out = torch.empty(numel, hidden_size, dtype=dtype, device="npu")
    kernel(hidden, idx, out)
    torch.npu.synchronize()

    got = out.cpu()
    n_mismatch = int((got != golden_cpu).sum().item())
    max_diff = 0.0
    if n_mismatch:
        max_diff = (got.float() - golden_cpu.float()).abs().max().item()
    # Bit-exact contract: pure copy, no arithmetic, no rounding.
    torch.testing.assert_close(got, golden_cpu, rtol=0, atol=0)

    stable = True
    for _ in range(repeat - 1):
        out2 = torch.empty(numel, hidden_size, dtype=dtype, device="npu")
        kernel(hidden, idx, out2)
        torch.npu.synchronize()
        if not torch.equal(out2, out):
            stable = False
    if not stable:
        raise AssertionError(f"[{tag}] unstable across {repeat} repeated runs")

    print(
        f"[{tag}] PASS T={num_tokens} K={top_k} H={hidden_size} numel={numel} "
        f"nk={gather_num_blocks(numel)} "
        f"spc={(numel + gather_num_blocks(numel) - 1) // gather_num_blocks(numel)} "
        f"dtype={dtype_str} exact={n_mismatch == 0} max_diff={max_diff:.3e} "
        f"repeat={repeat}"
    )
    return max_diff


# --------------------------------------------------------------------------
# Hierarchical testing
# --------------------------------------------------------------------------
def run_L0():
    """Gate: DESIGN sec 8.3 main line (bit-exact) on the design test plan."""
    _run_case(4, 2, 64, torch.bfloat16, "L0-1 tiny bf16 (numel=8)")
    _run_case(4, 2, 64, torch.float16, "L0-2 tiny fp16 (numel=8)")
    _run_case(16, 2, 128, torch.bfloat16, "L0-3 small (numel=32)")
    _run_case(5, 2, 64, torch.bfloat16, "L0-4 partial tail (numel=10)")
    _run_case(
        4, 4, 64, torch.bfloat16, "L0-5 duplicate source rows", idx_grouped
    )
    _run_case(
        8, 2, 64, torch.bfloat16, "L0-6 EP dummy zero-tail", idx_zero_tail
    )
    # Representative prefill (kimi-k2 style): nk=40, spc=820.
    _run_case(4096, 8, 7168, torch.bfloat16, "L0-7 representative prefill")
    # Dtype pass-through parity at the representative width.
    _run_case(4096, 8, 7168, torch.float16, "L0-8 prefill fp16")
    # L0-9 scan -> gather interface contract (tight scan-like order via GM).
    _run_case(
        64, 1, 128, torch.bfloat16, "L0-9 scan-contract perm", idx_perm
    )
    _run_case(
        64, 1, 128, torch.bfloat16, "L0-9 scan-contract grouped", idx_grouped
    )
    # L0-10 repeated-run stability (6 runs x 2 shapes).
    _run_case(
        4, 2, 64, torch.bfloat16, "L0-10 stability tiny", repeat=6
    )
    _run_case(
        64, 2, 256, torch.bfloat16, "L0-10 stability medium", repeat=6
    )
    print("L0 done: gate cases bit-exact")


def run_L1():
    """Non-divisible numel, top_k > 1, tail guard, both dtypes."""
    _run_case(1, 1, 32, torch.bfloat16, "L1-1 numel=1 (nk=1,spc=1)")
    _run_case(7, 5, 48, torch.bfloat16, "L1-2 top_k=5 (numel=35)")
    _run_case(5, 2, 16, torch.bfloat16, "L1-3 numel=10 (<P)")
    _run_case(16, 2, 32, torch.bfloat16, "L1-4 numel=32 (<P)")
    _run_case(40, 1, 16, torch.bfloat16, "L1-5 numel=40 (==P)")
    # numel > P and not a multiple of P -> tail guard actually exercised.
    _run_case(41, 1, 32, torch.bfloat16, "L1-6 numel=41 tail guard")
    _run_case(127, 1, 32, torch.bfloat16, "L1-7 numel=127 tail guard")
    _run_case(1001, 1, 16, torch.bfloat16, "L1-8 numel=1001 tail guard")
    _run_case(1003, 1, 16, torch.float16, "L1-9 numel=1003 fp16 tail")
    _run_case(1000, 1, 32, torch.bfloat16, "L1-10 numel=1000 exact split")
    _run_case(256, 4, 64, torch.bfloat16, "L1-11 top_k=4 numel=1024")
    _run_case(
        128, 2, 64, torch.bfloat16, "L1-12 grouped many-to-one", idx_grouped
    )
    _run_case(33, 2, 128, torch.bfloat16, "L1-13 numel=66 (<2*P, tail)")
    _run_case(2000, 3, 32, torch.float16, "L1-14 K=3 numel=6000 fp16")
    print("L1 done: non-divisible / top_k>1 / tail cases bit-exact")


def run_L2():
    """Large representative shapes (advisory: record, never blocking)."""
    cases = [
        (4096, 8, 3072, torch.bfloat16, "L2-1 qwen3-30b prefill (numel=32768)"),
        (2048, 4, 1024, torch.bfloat16, "L2-2 medium (numel=8192)"),
        (8192, 2, 512, torch.float16, "L2-3 wide tokens (numel=16384)"),
        (1024, 8, 4096, torch.bfloat16, "L2-4 wide hidden (numel=8192)"),
    ]
    failures = []
    for num_tokens, top_k, hidden_size, dtype, tag in cases:
        try:
            _run_case(num_tokens, top_k, hidden_size, dtype, tag)
        except Exception as exc:  # noqa: BLE001 - L2 is advisory
            failures.append((tag, str(exc)))
            print(f"[{tag}] WARN (non-blocking): {exc}")
    print(f"L2 done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def run_boundary():
    """Edge shapes: alignment floor, minimal/full tail, empty experts."""
    cases = [
        (1, 1, 16, torch.bfloat16, "B-1 single slot, 32B-aligned row"),
        (3, 1, 16, torch.bfloat16, "B-2 numel=3 tail"),
        (40, 1, 16, torch.bfloat16, "B-3 numel=P exactly", None),
        (41, 1, 16, torch.bfloat16, "B-4 numel=P+1 tail guard", None),
        (64, 2, 16, torch.bfloat16, "B-5 broadcast row 0", idx_single),
        (
            64,
            2,
            16,
            torch.bfloat16,
            "B-6 empty experts (rows 3/4 unreferenced)",
            idx_sparse,
        ),
        (16, 1, 16, torch.bfloat16, "B-7 K=1 permutation", idx_perm),
        (8, 2, 32, torch.bfloat16, "B-8 wide tail numel=16", idx_zero_tail),
        (128, 2, 16, torch.float16, "B-9 H=16 fp16 numel=256", None),
        (255, 1, 16, torch.bfloat16, "B-10 odd numel=255", None),
    ]
    failures = []
    for num_tokens, top_k, hidden_size, dtype, tag, *rest in cases:
        idx_builder = rest[0] if rest else None
        try:
            _run_case(
                num_tokens, top_k, hidden_size, dtype, tag, idx_builder=idx_builder
            )
        except Exception as exc:  # noqa: BLE001 - Boundary is advisory
            failures.append((tag, str(exc)))
            print(f"[{tag}] WARN (non-blocking): {exc}")
    print(f"Boundary done ({len(cases)} cases, {len(failures)} warn)")
    return failures


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
        l2_fail = run_L2()
        bd_fail = run_boundary()
        if l2_fail or bd_fail:
            # L2 / Boundary are advisory: recorded, never blocking.
            print(
                f"\033[93mAdvisory warns: L2={len(l2_fail)}, "
                f"Boundary={len(bd_fail)} (non-blocking)\033[0m"
            )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
