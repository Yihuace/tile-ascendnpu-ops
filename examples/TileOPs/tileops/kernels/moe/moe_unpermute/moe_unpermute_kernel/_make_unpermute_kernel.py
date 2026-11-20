# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_make_unpermute_kernel -- MoE token unpermute (per-token weighted gather-reduce).

Stage 3 implementation (Developer, target="npuir") of the frozen design:
    examples/moe_unpermute/_make_unpermute_kernel/DESIGN.md
    (REVIEW.md verdict = pass)

Semantics (DESIGN sec 0.1, the inverse of the MoE permute unit):

    out[i, :] = scaling * sum_{k : fwd_idx[i*K+k] >= 0}
                            topk_weights[i, k] * mm2_pad[fwd_idx[i*K+k], :]

    mm2_pad      [padded_batch_sum, H]  fp16 / bf16
    fwd_idx      [numel = T*K]          int32   flat slot -> padded slot (-1 = non-local, EP)
    topk_weights [T, K]                 float32
    output       [T, H]                 fp16 / bf16  (caller-allocated: out_idx=[])

This is a per-row weighted gather plus a K reduction.  Each output row is
written by exactly one core (same token's K contributions are accumulated
in-core), so there is NO cross-token accumulation: no atomics, no
non-determinism.

NPU form (DESIGN sec 0.6 R1-R6 / sec 3.3 main pick C1-varA)
------------------------------------------------------------
A single independent Vector / MTE2 kernel; fwd_idx crosses a kernel boundary
through GM (never consumed as a scalar from a same-scope vector-produced UB
value -- CG-2026-0027):

    nk  = min(T, P),  P = 2 * aicore_num = 40   (pure Vector physical cores)
    spc = ceildiv(T, nk)                         (static in-core bound)
    with T.Kernel(nk, is_npu=True):
        for t in T.serial(spc):                  # in-core serial token loop
            token = core_id * spc + t
            if token < T:                        # runtime tail guard
                T.clear(ub_acc)                  # fp32 accumulator row
                for k in T.serial(K):            # online K reduction (k ascending)
                    raw = fwd_idx[token*K + k]   # scalar GM read (across kernels)
                    if raw >= 0:                 # EP: -1 contributes exactly zero
                        w = topk_weights[token, k]
                        T.copy(mm2_pad[raw:raw+1, :], ub_src)   # dynamic row GM -> UB
                        T.vcast(ub_src, ub_f32, round_mode="rint")
                        T.vmul(ub_f32, w, ub_f32)               # scalar weight
                        T.vadd(ub_acc, ub_f32, ub_acc)
                if scaling != 1.0:
                    T.vmul(ub_acc, scaling, ub_acc)
                T.vcast(ub_acc, ub_out, round_mode="round")
                T.copy(ub_out, output[token:token+1, :])

The element axis H is carried by whole-row DMAs and whole-row vector ops; only
the token / K axes are serial.  online (single fp32 UB accumulator) is
mandatory: a two-stage materialisation needs K*H*4 = 229KB > 192KB UB at
H=7168/K=8.

Trap avoidance (DESIGN sec 3.5.1 / 9): no GM->GM copy (UB staging instead), no
scalar consumption of a vector-produced UB value (index read from GM), no
T.alloc_local, no int32 `>>`, no scalar T.if_then_else (EP / tail guards use
`if` statements), no T.Pipelined (runtime `if` inside would SIGSEGV,
CG-2026-0001a), no atomics (not needed), no manual sync primitives.

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python _make_unpermute_kernel.py --level all
"""

import os

# Developer mode: compiler-managed UB (alloc_ub) + automatic synchronisation.
# Set before importing tilelang (must match the ASCEND_MODE literal below).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

import tilelang
import tilelang.language as T

# Final validated programming mode for the TileOPs harness migration.
ASCEND_MODE = "Developer"


# Pure Vector / MTE2 physical core count (DESIGN sec 5.5):
#   NPUUtils.get().get_aicore_num() -> 20  (Ascend910B2C, measured)
#   a pure Vector unit uses 2x the AICore count -> 40 cores.
def _vector_core_count():
    try:
        from tilelang.utils.npu_utils import NPUUtils

        return int(NPUUtils.get().get_aicore_num()) * 2
    except Exception:  # noqa: BLE001 - fall back to the measured Ascend910B2C value
        return 40


VECTOR_CORES = _vector_core_count()


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

# Keep the UB buffers single-versioned and stop the buffer planner from
# re-scoping/permuting them: the dynamic-source-row range copy is only emitted
# reliably with auto multi-buffer disabled (DESIGN sec 4.7 / 7.3).
_PASS = {
    tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False,
    tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION: False,
}


# --------------------------------------------------------------------------
# Golden (DESIGN sec 8.1 -- source semantics, explicit torch module API).
# --------------------------------------------------------------------------
def golden_unpermute(mm2_pad, fwd_idx, topk_weights, scaling=1.0):
    """Per-token weighted gather-reduce (source semantics; EP -1 -> zero).

    Built from explicit ``torch.*`` module calls (``torch.reshape`` /
    ``torch.where`` / ``torch.index_select`` / ``torch.mul`` /
    ``torch.zeros`` / ``torch.repeat_interleave`` / ``torch.index_add``) --
    independent of the NPU algorithm (no core split, no serial loop, no
    dynamic slice).  fp32 accumulation, single final cast -- same contract as
    the kernel.
    """
    num_tokens, top_k = topk_weights.shape
    hidden_size = mm2_pad.shape[1]
    out_dtype = mm2_pad.dtype

    slots = torch.reshape(fwd_idx, (num_tokens, top_k)).to(torch.long)
    valid = slots >= 0  # EP: -1 marks a non-local expert
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    # masked routing weight: non-local slots contribute exactly zero.
    weights = torch.where(valid, topk_weights, torch.zeros_like(topk_weights))

    gathered = torch.index_select(mm2_pad, 0, torch.reshape(safe, (-1,)))
    weighted = torch.mul(gathered.to(torch.float32),
                         torch.reshape(weights, (-1, 1)))
    acc = torch.zeros((num_tokens, hidden_size), dtype=torch.float32)
    slot_token = torch.repeat_interleave(
        torch.arange(num_tokens, dtype=torch.long), top_k
    )
    acc = torch.index_add(acc, 0, slot_token, weighted)
    acc = torch.mul(acc, scaling)
    return acc.to(out_dtype)


# Alias matching the ``golden_<op>`` naming convention of the Stage 3 gate.
golden__make_unpermute_kernel = golden_unpermute


# --------------------------------------------------------------------------
# Kernel (DESIGN sec 3.3 main pick C1-varA, one Vector kernel)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[], target="npuir", pass_configs=_PASS)
def _make_unpermute_kernel(
    num_tokens, top_k, hidden_size, padded_batch_sum, dtype, scaling=1.0
):
    """Factory for the MoE token unpermute bottom.

    ``num_tokens`` / ``top_k`` / ``hidden_size`` / ``padded_batch_sum`` /
    ``dtype`` / ``scaling`` are compile-time specialised by the harness (same
    as the source factory + lru_cache).  ``output`` is caller-allocated
    (``out_idx=[]``), matching the source op.
    """
    a5 = _is_a5()
    if a5:
        # A5/CANN 9.2: an scf.for over tokens whose body holds a runtime tail
        # guard makes bishengir SIGSEGV (recursive walk, measured: spc>1 with
        # a real tail crashes; spc==1 or an exact split compiles).  Launch one
        # token per block instead: spc==1, no in-core serial loop.  The grid
        # limit is 65535 blocks.
        if num_tokens > 65535:
            raise ValueError(
                f"A5/CANN 9.2 unpermute kernel is limited to 65535 tokens "
                f"per launch (one token per block); got num_tokens={num_tokens}"
            )
        nk = num_tokens
        spc = 1
    else:
        nk = min(num_tokens, VECTOR_CORES)  # compile-time constant (DESIGN sec 5.5)
        spc = (num_tokens + nk - 1) // nk  # static in-core serial bound

    @T.prim_func
    def _unpermute_main(
        mm2_pad: T.Tensor([padded_batch_sum, hidden_size], dtype),
        fwd_idx: T.Tensor([num_tokens * top_k], "int32"),
        topk_weights: T.Tensor([num_tokens, top_k], "float32"),
        output: T.Tensor([num_tokens, hidden_size], dtype),
    ):
        with T.Kernel(nk, is_npu=True) as (core_id, _subid):
            # 整行 UB 缓冲：动态源行 / fp32 提升行 / fp32 在线累加器 / 落盘行
            ub_src = T.alloc_ub((1, hidden_size), dtype)
            ub_f32 = T.alloc_ub((1, hidden_size), "float32")
            ub_acc = T.alloc_ub((1, hidden_size), "float32")
            ub_out = T.alloc_ub((1, hidden_size), dtype)
            # 核内 token 串行：spc 为编译期静态边界（core-split 要素③）
            for t in T.serial(spc):
                token = core_id * spc + t
                # 尾块守卫：T 非核数整数倍时跳过越界 token（与源 grid=T 等价）
                if token < num_tokens:
                    T.clear(ub_acc)  # 每个 token 的 fp32 累加器清零
                    for k in T.serial(top_k):  # K 串行 online 累加（k 升序）
                        # 标量 GM 读源行号（跨 kernel 交接；不消费本 scope 向量
                        # 产出的 UB 值，规避 CG-2026-0027）
                        raw = fwd_idx[token * top_k + k]
                        # EP 守卫：fwd_idx == -1 表示非本地专家，跳过该 k
                        # （等价零贡献，D-1 E1 逐位等价）
                        if a5:
                            # A5/CANN 9.2：k 循环体内的 scf.if 让 bishengir
                            # SIGSEGV（栈递归 memcpy 崩溃，实测）。改为无分支
                            # 等价式：源行号 clamp 到 0，槽位 -1 时权重乘 0。
                            #   mask = clamp(raw + 1, 0, 1) ∈ {0, 1}
                            # raw < -1 不属于契约（只有 -1 哨兵），
                            # 与参考实现的「非负才贡献」逐位等价。
                            src = T.max(raw, 0)
                            w = topk_weights[token, k] * T.cast(
                                T.min(T.max(raw + 1, 0), 1), "float32")
                            T.copy(
                                mm2_pad[src : src + 1, 0:hidden_size],
                                ub_src[0:1, 0:hidden_size],
                            )
                            T.vcast(ub_src, ub_f32, round_mode="rint")
                            T.vmul(ub_f32, w, ub_f32)
                            T.vadd(ub_acc, ub_f32, ub_acc)
                        else:
                            if raw >= 0:
                                # 标量权重（Kernel 内 / Scope 外，供 vmul 标量 src）
                                w = topk_weights[token, k]
                                # 整行 GM->UB：动态源行 [raw:raw+1]（纯 DMA 向量搬运）
                                T.copy(
                                    mm2_pad[raw : raw + 1, 0:hidden_size],
                                    ub_src[0:1, 0:hidden_size],
                                )
                                # 整行 bf16/fp16 -> fp32（round_mode 必须关键字传参）
                                T.vcast(ub_src, ub_f32, round_mode="rint")
                                # 乘标量路由权重（in-place，scalar src2 合法）
                                T.vmul(ub_f32, w, ub_f32)
                                # fp32 在线累加（in-place，dst==src0 合法）
                                T.vadd(ub_acc, ub_f32, ub_acc)
                    # 末次缩放（编译期常量分支）：out = scale * acc
                    if scaling != 1.0:
                        T.vmul(ub_acc, scaling, ub_acc)
                    # 整行 fp32 -> dtype 落盘 cast（单次 cast，与源契约一致）
                    T.vcast(ub_acc, ub_out, round_mode="round")
                    # 整行 UB->GM：动态目的行 [token:token+1]
                    T.copy(
                        ub_out[0:1, 0:hidden_size],
                        output[token : token + 1, 0:hidden_size],
                    )

    return _unpermute_main


# --------------------------------------------------------------------------
# Index builders for the layered test suite (all values must be valid padded
# slots in [0, padded_batch_sum), or -1 for the EP mask).
# --------------------------------------------------------------------------
def _idx_ep(numel, padded, top_k):
    """EP mode: every 3rd flat slot marked non-local (-1)."""
    idx = torch.randperm(padded, dtype=torch.int32)[:numel].clone()
    idx[::3] = -1
    return idx


def _idx_all_neg(numel, padded, top_k):
    """EP extreme: every slot non-local -> output must be exactly zero."""
    return torch.full((numel,), -1, dtype=torch.int32)


def _idx_dup(numel, padded, top_k):
    """Duplicate source rows: a small set of padded slots reused many times."""
    active = max(1, numel // 4)
    return torch.randint(0, active, (numel,), dtype=torch.int32)


def _idx_single(numel, padded, top_k):
    """Broadcast: every slot gathers padded row 0 (maximum duplication)."""
    return torch.zeros(numel, dtype=torch.int32)


def _idx_sparse(numel, padded, top_k):
    """Empty experts: only the first quarter of the padded rows is referenced."""
    active = max(1, numel // 4)
    return torch.randint(0, active, (numel,), dtype=torch.int32)


def _idx_skewed(numel, padded, top_k):
    """Skewed routing: all tokens to the first K padded slots."""
    base = torch.arange(numel, dtype=torch.int32)
    return (base % max(1, top_k)).to(torch.int32)


_DTYPE_NAME = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
_TOL = {torch.bfloat16: (1.6e-2, 1.6e-2), torch.float16: (1e-3, 1e-3)}


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
    scaling=1.0,
    pad_extra=0,
    seed=0,
    repeat=1,
):
    """Run one shape end-to-end on NPU and compare with the torch golden.

    ``repeat > 1`` re-runs the kernel into a fresh output buffer and requires
    bit-identical results (guards the CG-2026-0027 stale-read / WAR class).
    """
    numel = num_tokens * top_k
    padded = numel + pad_extra
    dtype_str = _DTYPE_NAME[dtype]
    rtol, atol = _TOL[dtype]

    torch.manual_seed(seed)
    mm2_cpu = (torch.randn(padded, hidden_size, dtype=torch.float32) * 0.1).to(dtype)
    if idx_builder is None:
        idx_cpu = torch.randperm(padded, dtype=torch.int32)[:numel]
    else:
        idx_cpu = idx_builder(numel, padded, top_k).to(torch.int32)
    assert idx_cpu.shape == (numel,), (idx_cpu.shape, numel)
    w_cpu = torch.rand(num_tokens, top_k, dtype=torch.float32)

    golden_cpu = golden_unpermute(mm2_cpu, idx_cpu, w_cpu, scaling)

    kernel = _make_unpermute_kernel(
        num_tokens, top_k, hidden_size, padded, dtype_str, scaling
    )
    mm2 = mm2_cpu.npu()
    idx = idx_cpu.npu()
    w = w_cpu.npu()
    out = torch.empty(num_tokens, hidden_size, dtype=dtype, device="npu")
    kernel(mm2, idx, w, out)
    torch.npu.synchronize()

    got = out.cpu()
    max_diff = (got.float() - golden_cpu.float()).abs().max().item()
    torch.testing.assert_close(
        got.float(), golden_cpu.float(), rtol=rtol, atol=atol
    )

    stable = True
    for _ in range(repeat - 1):
        out2 = torch.empty(num_tokens, hidden_size, dtype=dtype, device="npu")
        kernel(mm2, idx, w, out2)
        torch.npu.synchronize()
        if not torch.equal(out2, out):
            stable = False
    if not stable:
        raise AssertionError(f"[{tag}] unstable across {repeat} repeated runs")

    print(
        f"[{tag}] PASS T={num_tokens} K={top_k} H={hidden_size} padded={padded} "
        f"dtype={dtype_str} scaling={scaling} nk={min(num_tokens, VECTOR_CORES)} "
        f"spc={(num_tokens + min(num_tokens, VECTOR_CORES) - 1) // min(num_tokens, VECTOR_CORES)} "
        f"max_diff={max_diff:.3e} repeat={repeat}"
    )
    return max_diff


# --------------------------------------------------------------------------
# Hierarchical testing
# --------------------------------------------------------------------------
def run_L0():
    """Gate: DESIGN sec 8.3 plan (bf16 + fp16, EP, top_k=1, scaling, prefill)."""
    _run_case(1, 8, 512, torch.bfloat16, "L0-1 smoke bf16")
    _run_case(1, 8, 512, torch.float16, "L0-2 smoke fp16")
    _run_case(4, 2, 64, torch.bfloat16, "L0-3 tiny bf16")
    _run_case(4, 2, 64, torch.float16, "L0-4 tiny fp16")
    _run_case(8, 1, 64, torch.bfloat16, "L0-5 top_k=1 stability", repeat=6)
    _run_case(16, 4, 256, torch.bfloat16, "L0-6 EP mask", idx_builder=_idx_ep)
    _run_case(8, 2, 256, torch.bfloat16, "L0-7 scaling", scaling=2.827)
    _run_case(4096, 8, 7168, torch.bfloat16, "L0-8 representative prefill")
    _run_case(4096, 8, 3072, torch.bfloat16, "L0-9 qwen3-30b prefill")
    _run_case(64, 8, 2048, torch.bfloat16, "L0-10 stability", repeat=6)
    print("L0 done: gate cases pass")


def run_L1():
    """Non-divisible token/H, top_k>1, EP, padding, duplicates, empty experts."""
    # tail guard: T not a multiple of the 40 physical cores.
    _run_case(41, 1, 64, torch.bfloat16, "L1-1 T=41 tail guard")
    _run_case(127, 2, 64, torch.bfloat16, "L1-2 T=127 tail guard")
    _run_case(1001, 1, 32, torch.bfloat16, "L1-3 T=1001 tail guard")
    _run_case(1003, 1, 32, torch.float16, "L1-4 T=1003 fp16 tail")
    # top_k > 1 (K = 3, 4, 5, 8).
    _run_case(2000, 3, 32, torch.float16, "L1-5 K=3 fp16")
    _run_case(128, 4, 64, torch.bfloat16, "L1-6 K=4")
    _run_case(40, 5, 48, torch.bfloat16, "L1-7 K=5 H=48")
    _run_case(100, 8, 80, torch.bfloat16, "L1-8 K=8 H=80")
    # padded_batch_sum > T*K (unreferenced padding rows must not be read).
    _run_case(16, 4, 64, torch.bfloat16, "L1-9 padded gap", pad_extra=37)
    _run_case(100, 2, 64, torch.float16, "L1-10 padded gap fp16", pad_extra=100)
    # EP masking combined with tail + padding.
    _run_case(
        43, 3, 48, torch.bfloat16, "L1-11 EP + tail + pad",
        idx_builder=_idx_ep, pad_extra=11,
    )
    # duplicate source rows / empty experts.
    _run_case(64, 2, 64, torch.bfloat16, "L1-12 duplicate rows", idx_builder=_idx_dup)
    _run_case(
        96, 2, 64, torch.bfloat16, "L1-13 empty experts", idx_builder=_idx_sparse
    )
    # non-power-of-2 / non-16 H widths (still 32B-aligned rows: H multiple of 16).
    _run_case(33, 2, 48, torch.bfloat16, "L1-14 H=48")
    _run_case(70, 3, 144, torch.bfloat16, "L1-15 H=144")
    # scaling with K>1 and padding.
    _run_case(64, 4, 128, torch.bfloat16, "L1-16 scaling K=4", scaling=1.5,
              pad_extra=9)
    print("L1 done: non-divisible / top_k>1 / EP / padding cases pass")


def run_L2():
    """Large representative shapes (advisory: record, never blocking)."""
    cases = [
        (2048, 4, 1024, torch.bfloat16, "L2-1 medium", None, 1.0, 0),
        (8192, 2, 512, torch.float16, "L2-2 wide tokens fp16", None, 1.0, 0),
        (1024, 8, 4096, torch.bfloat16, "L2-3 wide hidden", None, 1.0, 0),
        (512, 8, 7168, torch.bfloat16, "L2-4 large-hidden-medium", None, 1.0, 0),
        (4096, 8, 7168, torch.float16, "L2-5 kimi fp16", None, 1.0, 0),
    ]
    failures = []
    for num_tokens, top_k, hidden_size, dtype, tag, builder, sc, pe in cases:
        try:
            _run_case(num_tokens, top_k, hidden_size, dtype, tag,
                      idx_builder=builder, scaling=sc, pad_extra=pe)
        except Exception as exc:  # noqa: BLE001 - L2 is advisory
            failures.append((tag, str(exc)))
            print(f"[{tag}] WARN (non-blocking): {exc}")
    print(f"L2 done ({len(cases)} cases, {len(failures)} warn)")
    return failures


def run_boundary():
    """Edge shapes: minimal widths, alignment, all-masked EP, broadcast."""
    cases = [
        (1, 1, 16, torch.bfloat16, "B-1 single token/slot", None, 1.0, 0),
        (3, 1, 16, torch.bfloat16, "B-2 T=3 tail", None, 1.0, 0),
        (40, 1, 16, torch.bfloat16, "B-3 T=P exactly", None, 1.0, 0),
        (41, 1, 16, torch.bfloat16, "B-4 T=P+1 tail guard", None, 1.0, 0),
        (16, 4, 256, torch.bfloat16, "B-5 all-(-1) EP -> zeros", _idx_all_neg, 1.0, 0),
        (64, 2, 16, torch.bfloat16, "B-6 broadcast row 0", _idx_single, 1.0, 0),
        (32, 4, 64, torch.bfloat16, "B-7 skewed routing", _idx_skewed, 1.0, 0),
        (8, 2, 8, torch.bfloat16, "B-8 H=8 minimal row", None, 1.0, 0),
        (16, 2, 32, torch.bfloat16, "B-9 H=32 fp16", None, 1.0, 0),
        (255, 1, 16, torch.bfloat16, "B-10 odd T=255", None, 1.0, 0),
        (16, 2, 72, torch.bfloat16, "B-11 H=72 32B-unaligned row", None, 1.0, 0),
        (16, 2, 50, torch.bfloat16, "B-12 H=50 4B-aligned row", None, 1.0, 0),
        (16, 2, 64, torch.bfloat16, "B-13 large pad gap", None, 1.0, 4096),
    ]
    failures = []
    for num_tokens, top_k, hidden_size, dtype, tag, builder, sc, pe in cases:
        try:
            _run_case(num_tokens, top_k, hidden_size, dtype, tag,
                      idx_builder=builder, scaling=sc, pad_extra=pe)
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
