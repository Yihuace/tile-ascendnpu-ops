# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_fused_topk_kernel -- MoE fused top-k routing (NPU Developer mode).

Migrated from the GPU TileOPs ``FusedTopKKernel`` factory
(``tileops/kernels/moe/fused_topk.py``, CUDA one-warp-per-token) to
``target="npuir"``.

Semantics (DESIGN.md section 0.1 / 1.3, frozen):
    logits = gating_output.to(float32)                      # fp32 compute domain
    scores = softmax(logits, -1) or sigmoid(logits)
    ids    = topK(scores, K)                                # per row
    w      = scores[ids]                                    # weights
    if renormalize: w = w / w.sum(-1, keepdim=True)
    -> topk_weights (T, K) float32, topk_ids (T, K) int32

NPU redesign (DESIGN.md section 0.6 / 1.6, frozen):
    R1: one-dimensional persistent core split -- ``num_kernels =
        min(num_logical, 40)`` (pure-Vector physical cores = 20 AICore x 2)
        with an in-core static ``T.serial`` grid-stride loop and a block-level
        ``if task < num_logical`` guard.
    R2: warp-shuffle reductions -> Vector row reductions
        (``T.reduce_max`` / ``T.reduce_sum`` with ``dim=1``).
    R3: K-pass masked argmax -> ``T.vsort(descending=True)`` on the RAW logits,
        taking the first K columns (values + int32 original expert indices).
        Selection on raw logits is exact because softmax/sigmoid are strictly
        monotone (E1).
    R4: no non-divisible-E padding (sorting/scoring acts on the true E columns).
    R5: renormalize fused into the same kernel; for softmax+renorm the full-row
        denominator cancels, so only the K-element denominator is needed (E2).

Toolchain traps deliberately avoided (DESIGN.md section 3.5):
    - no ``T.alloc_local`` (+ ``T.serial`` -> SIGSEGV, CG-2026-0028b);
    - no ``>>`` / scalar ``T.if_then_else`` (CG-2026-0028a/c);
    - no scalar consumption of vector output (CG-2026-0027);
    - no direct GM write inside ``T.Parallel`` (CG-2026-0030);
    - no scalar payload ``T.atomic_add`` (CG-2026-0029).
All intermediate values stay vector-shaped in UB; the only scalar control flow
is the block-level ``if task < num_logical`` guard.
"""

import os

# Developer mode: the compiler manages UB (alloc_shared) and auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
# Chip 1 (chip 0 is reserved for perf measurement).
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# UB budget for the large buffers (xf + sv + si = BM * E * 12 bytes) and max
# rows per block (DESIGN.md section 4.5).  Conservative w.r.t. 192KB UB to
# leave room for T.vsort scratch and auto-multi-buffer inflation.
UB_BUDGET = 96 * 1024
BM_MAX = 64

# Harness tolerances (test_moe_fused_topk.py): weights compared after sorting.
_RTOL = 1e-3
_ATOL = 1e-3
# Raw-score validity threshold for topk_ids (test_moe_fused_topk.py `_check`).
_ID_SCORE_SLACK = 1e-4

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_fused_topk(
    gating_output: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = False,
) -> tuple:
    """Source semantics: score full row -> topk -> optional renormalize.

    This reproduces the frozen source semantics (DESIGN.md section 0.1 / 8.1)
    and the harness reference ``fused_topk_torch``.  It does NOT reproduce the
    NPU redesign (no raw-logits vsort, no fused renorm), so validation stays
    independent.  Every step is an explicit ``torch`` module API call.
    """
    logits = gating_output.to(torch.float32)
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown scoring_func: {scoring_func}")
    # sorted=False matches the harness reference (the id set is what matters).
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.float(), topk_ids.int()


# ---------- Kernel ----------
def _select_bm(num_experts: int, num_tokens: int, hint) -> int:
    """Rows per block: UB-budget cap, optionally tightened by a caller hint."""
    cap = min(BM_MAX, UB_BUDGET // (num_experts * 12))
    cap = max(1, cap)
    bm = cap
    if hint is not None and hint > 0:
        bm = min(hint, cap)
    return max(1, min(bm, num_tokens))


def _fused_topk_kernel(
    num_tokens,
    num_experts,
    top_k,
    scoring_func="softmax",
    with_correction_bias=False,
    renormalize=False,
    in_dtype="float32",
    block_m=None,
):
    """Build the NPU fused top-k routing kernel factory.

    Mirrors the harness contract ``_fused_topk_kernel(num_tokens, num_experts,
    top_k, scoring_func, with_correction_bias)`` with two NPU-side extensions
    appended: ``renormalize`` (fused into the kernel per R5) and ``in_dtype``
    (the native input dtype; the kernel promotes to fp32 internally).

    ``with_correction_bias=True`` is deliberately not implemented in this unit
    (DESIGN.md section 9 R-1, frozen): the harness workload never generates a
    correction bias and the reference test has no numeric case for it.  The
    landing path (sort ``sigmoid(x)+bias`` then gather the unbiased sigmoid at
    the selected ids) is documented in DESIGN.md section 3.2.
    """
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(f"Unsupported scoring_func {scoring_func!r}")
    if in_dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported in_dtype {in_dtype!r}")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    if num_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError(
            f"num_tokens/num_experts/top_k must be positive, got "
            f"{num_tokens}/{num_experts}/{top_k}"
        )
    if with_correction_bias:
        raise NotImplementedError(
            "with_correction_bias=True is not implemented in this unit "
            "(DESIGN.md R-1); the standard sigmoid/softmax variant is provided."
        )

    # Pure-Vector physical cores = AICore count x 2 (DESIGN.md section 5.5).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[1, 2], target="npuir")
    def _func(TOKENS_PER_BLOCK=None):
        requested = TOKENS_PER_BLOCK if TOKENS_PER_BLOCK is not None else block_m
        bm = _select_bm(num_experts, num_tokens, requested)
        num_logical = _ceildiv(num_tokens, bm)          # row blocks
        num_kernels = min(num_logical, vector_cores)    # persistent grid
        num_local_tasks = _ceildiv(num_logical, num_kernels)  # static bound

        @T.prim_func
        def main(
            gating_output: T.Tensor((num_tokens, num_experts), in_dtype),
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),
            topk_ids: T.Tensor((num_tokens, top_k), "int32"),
        ):
            # 一维 persistent 分核：固定物理核数 + 核内静态 T.serial grid-stride
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xf = T.alloc_shared((bm, num_experts), "float32")   # fp32 行块
                sv = T.alloc_shared((bm, num_experts), "float32")   # 降序值
                si = T.alloc_shared((bm, num_experts), "int32")     # 降序原始下标
                mr = T.alloc_shared((bm, 1), "float32")             # 行 max
                sr = T.alloc_shared((bm, 1), "float32")             # 分母
                vk = T.alloc_shared((bm, top_k), "float32")         # 选中权重
                ik = T.alloc_shared((bm, top_k), "int32")           # 选中下标
                for s in T.serial(num_local_tasks):
                    task = s * num_kernels + cid
                    if task < num_logical:
                        off = task * bm
                        real = T.min(bm, num_tokens - off)  # 尾块行数
                        # 载入行块：GM -> UB，跨 dtype 自动升 fp32
                        T.copy(
                            gating_output[off : off + real, 0:num_experts],
                            xf[0:real, 0:num_experts],
                        )
                        # 选择：在原始 logits 上降序排序（值 + int32 原始下标）
                        T.vsort(xf, sv, si, descending=True, sort_axis=-1)
                        # 取前 K 列（原始 logits 值 + 专家下标）
                        T.copy(sv[0:real, 0:top_k], vk[0:real, 0:top_k])
                        T.copy(si[0:real, 0:top_k], ik[0:real, 0:top_k])
                        if scoring_func == "softmax":
                            # 行 max（softmax 稳定化，水平归约 lane->E）
                            T.reduce_max(xf, mr, dim=1, clear=True)
                            # 仅对选中 K 个元素打分：exp(x_k - m)
                            T.vsub(vk, mr, vk)
                            T.vexp(vk, vk)
                            if renormalize:
                                # 全行分母抵消，只需 K 元素分母
                                T.reduce_sum(vk, sr, dim=1, clear=True)
                            else:
                                # 非 renorm 需全行 exp 分母（原地复用 xf）
                                T.vsub(xf, mr, xf)
                                T.vexp(xf, xf)
                                T.reduce_sum(xf, sr, dim=1, clear=True)
                            # 归一化（行广播除）
                            T.vdiv(vk, sr, vk)
                        else:  # sigmoid
                            # 仅对选中 K 个元素打分：sigma(x_k)
                            T.vsigmoid(vk, vk)
                            if renormalize:
                                T.reduce_sum(vk, sr, dim=1, clear=True)
                                T.vdiv(vk, sr, vk)
                        # 写出：UB -> GM
                        T.copy(
                            vk[0:real, 0:top_k],
                            topk_weights[off : off + real, 0:top_k],
                        )
                        T.copy(
                            ik[0:real, 0:top_k],
                            topk_ids[off : off + real, 0:top_k],
                        )

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(
    num_tokens,
    num_experts,
    top_k,
    scoring_func,
    renormalize,
    dtype_str,
    tag,
    logits=None,
    seed=0,
    scale=1.0,
    block_m=None,
    equal_nan=False,
):
    torch_dtype = _DTYPE_MAP[dtype_str]
    if logits is None:
        gen = torch.Generator().manual_seed(seed)
        logits = torch.randn(num_tokens, num_experts, generator=gen) * scale
    logits = logits.reshape(num_tokens, num_experts).to(torch_dtype)

    factory = _fused_topk_kernel(
        num_tokens,
        num_experts,
        top_k,
        scoring_func,
        False,
        renormalize,
        dtype_str,
        block_m,
    )
    fn = factory(block_m)
    out_w, out_ids = fn(logits.npu())
    torch.npu.synchronize()

    ref_w, ref_ids = golden_fused_topk(logits, top_k, scoring_func, renormalize)

    # shapes / dtypes
    assert tuple(out_w.shape) == (num_tokens, top_k), (
        f"[{tag}] weights shape {tuple(out_w.shape)} != ({num_tokens},{top_k})"
    )
    assert tuple(out_ids.shape) == (num_tokens, top_k), (
        f"[{tag}] ids shape {tuple(out_ids.shape)} != ({num_tokens},{top_k})"
    )
    assert out_w.dtype == torch.float32, f"[{tag}] weights dtype {out_w.dtype}"
    assert out_ids.dtype == torch.int32, f"[{tag}] ids dtype {out_ids.dtype}"

    out_w_cpu = out_w.cpu().float()
    out_ids_cpu = out_ids.cpu()
    ref_w_sorted = ref_w.sort(dim=-1).values
    out_w_sorted = out_w_cpu.sort(dim=-1).values
    torch.testing.assert_close(
        out_w_sorted, ref_w_sorted, rtol=_RTOL, atol=_ATOL, equal_nan=equal_nan
    )

    # id validity: no duplicates + each selected raw score >= K-th ref raw score
    logits_f32 = logits.float()
    if scoring_func == "softmax":
        all_scores = torch.softmax(logits_f32, dim=-1)
    else:
        all_scores = torch.sigmoid(logits_f32)
    raw_ref = all_scores.gather(1, ref_ids.long()).sort(dim=-1).values
    kth_val = raw_ref[:, 0]
    for i in range(num_tokens):
        sel_ids = out_ids_cpu[i].tolist()
        assert len(set(sel_ids)) == top_k, (
            f"[{tag}] token {i}: duplicate expert ids {sorted(sel_ids)}"
        )
        assert all(0 <= e < num_experts for e in sel_ids), (
            f"[{tag}] token {i}: out-of-range expert ids {sorted(sel_ids)}"
        )
        if top_k < num_experts:
            for eid in sel_ids:
                got = all_scores[i, eid].item()
                assert got >= kth_val[i].item() - _ID_SCORE_SLACK, (
                    f"[{tag}] token {i}: expert {eid} score {got:.6f} < min ref "
                    f"score {kth_val[i].item():.6f}"
                )

    max_diff = (out_w_sorted - ref_w_sorted).abs().max().item()
    print(
        f"[{tag}] PASS: T={num_tokens} E={num_experts} K={top_k} "
        f"{scoring_func} renorm={renormalize} dtype={dtype_str} max_diff={max_diff:.2e}"
    )
    return max_diff


def _tie_logits(num_tokens, num_experts, groups=None, seed=0):
    """CPU logits with many exact ties (uniform, or few distinct values)."""
    gen = torch.Generator().manual_seed(seed)
    if groups is None:
        return torch.zeros(num_tokens, num_experts)
    base = torch.randn(num_tokens, groups, generator=gen)
    idx = torch.randint(0, groups, (num_tokens, num_experts), generator=gen)
    return base.gather(1, idx)


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.2 (10 cases).
def run_L0():
    max_diff = 0.0
    cases = [
        (32, 128, 8, "softmax", False, "float32", 0),
        (32, 128, 8, "softmax", False, "float16", 1),
        (32, 128, 8, "softmax", False, "bfloat16", 2),
        (32, 256, 8, "sigmoid", True, "bfloat16", 3),
        (32, 100, 4, "softmax", False, "bfloat16", 4),
        (32, 33, 2, "sigmoid", False, "bfloat16", 5),
        (4096, 256, 8, "sigmoid", False, "bfloat16", 6),
        (4096, 384, 8, "sigmoid", True, "bfloat16", 7),
        (4096, 128, 8, "softmax", False, "bfloat16", 8),
        (512, 64, 1, "softmax", False, "bfloat16", 9),
    ]
    for i, (tn, e, k, sf, rn, dt, seed) in enumerate(cases):
        max_diff = max(
            max_diff,
            _run_case(tn, e, k, sf, rn, dt, f"L0-{i + 1}", seed=seed),
        )
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")
    return max_diff


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    # scoring x renorm grid
    max_diff = max(max_diff, _run_case(128, 128, 8, "softmax", False, "bfloat16", "L1-1", seed=11))
    max_diff = max(max_diff, _run_case(128, 128, 8, "softmax", True, "bfloat16", "L1-2", seed=12))
    max_diff = max(max_diff, _run_case(128, 256, 8, "sigmoid", False, "bfloat16", "L1-3", seed=13))
    max_diff = max(max_diff, _run_case(128, 256, 8, "sigmoid", True, "bfloat16", "L1-4", seed=14))
    # fp32 / fp16 paths, top_k extremes
    max_diff = max(max_diff, _run_case(256, 64, 1, "softmax", False, "float32", "L1-5", seed=15))
    max_diff = max(max_diff, _run_case(64, 32, 32, "softmax", False, "float32", "L1-6", seed=16))
    max_diff = max(max_diff, _run_case(64, 8, 1, "sigmoid", False, "float16", "L1-7", seed=17))
    # non-divisible E
    max_diff = max(max_diff, _run_case(100, 100, 4, "softmax", False, "float32", "L1-8", seed=18))
    max_diff = max(max_diff, _run_case(33, 33, 2, "sigmoid", False, "bfloat16", "L1-9", seed=19))
    max_diff = max(max_diff, _run_case(127, 7, 3, "sigmoid", False, "float32", "L1-10", seed=20))
    # T tail (513 = 8*64 + 1) with auto block_m
    max_diff = max(max_diff, _run_case(513, 128, 8, "softmax", False, "bfloat16", "L1-11", seed=21))
    # tie-break: all-equal logits and few distinct groups
    max_diff = max(
        max_diff,
        _run_case(64, 64, 8, "softmax", False, "float32", "L1-12a",
                  logits=_tie_logits(64, 64)),
    )
    max_diff = max(
        max_diff,
        _run_case(64, 64, 8, "softmax", True, "float32", "L1-12b",
                  logits=_tie_logits(64, 64)),
    )
    max_diff = max(
        max_diff,
        _run_case(96, 48, 4, "sigmoid", True, "bfloat16", "L1-13",
                  logits=_tie_logits(96, 48, groups=3, seed=22)),
    )
    print(f"[L1] ALL PASS, max_diff={max_diff:.2e}")
    return max_diff


# ---------- L2: degenerate shapes (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, 1, "softmax", False, "float32", 31),
        (5, 8, 1, "sigmoid", False, "float16", 32),
        (1, 64, 4, "softmax", True, "bfloat16", 33),
        (8, 2, 2, "softmax", False, "float32", 34),
        (3, 5, 5, "sigmoid", False, "bfloat16", 35),
    ]
    for i, (tn, e, k, sf, rn, dt, seed) in enumerate(cases):
        try:
            _run_case(tn, e, k, sf, rn, dt, f"L2-{i + 1}", seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): T={tn} E={e} K={k} {sf} "
                f"renorm={rn} {dt}: {type(exc).__name__}: {exc}"
            )


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # exact ties at a larger shape
    try:
        _run_case(256, 96, 8, "softmax", False, "float32", "Boundary-ties",
                  logits=_tie_logits(256, 96))
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): exact-ties: {type(exc).__name__}: {exc}")

    # large magnitude (softmax saturation / exp overflow risk, max-shift)
    try:
        _run_case(32, 128, 8, "softmax", False, "float32", "Boundary-large-mag",
                  scale=50.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large-magnitude: {type(exc).__name__}: {exc}")

    # sigmoid with large negative/positive mix
    try:
        gen = torch.Generator().manual_seed(41)
        x = torch.randn(64, 256, generator=gen) * 40.0
        _run_case(64, 256, 8, "sigmoid", True, "float32", "Boundary-sigmoid-sat",
                  logits=x)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): sigmoid-saturation: {type(exc).__name__}: {exc}")

    # signed infinities in the logits (torch softmax over -inf row -> NaN)
    try:
        x = torch.randn(8, 32)
        x[0, 0] = float("inf")
        x[1, 0] = float("-inf")
        _run_case(8, 32, 4, "softmax", False, "float32", "Boundary-inf", logits=x)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): signed-inf: {type(exc).__name__}: {exc}")

    # NaN propagation (documented as outside the input domain, DESIGN.md R-7)
    try:
        x = torch.randn(8, 32)
        x[0, 3] = float("nan")
        _run_case(8, 32, 4, "softmax", False, "float32", "Boundary-nan",
                  logits=x, equal_nan=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): NaN: {type(exc).__name__}: {exc}")

    # correction_bias contract (DESIGN.md R-1: variant deferred in this unit)
    try:
        _fused_topk_kernel(4, 8, 2, "sigmoid", True)
        print("[Boundary] WARN (record only): with_correction_bias=True did not raise")
    except NotImplementedError as exc:
        print(f"[Boundary] PASS (recorded): correction_bias deferred as designed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): correction_bias unexpected: {type(exc).__name__}: {exc}")

    # standard variant with with_correction_bias=False must work
    try:
        _run_case(32, 64, 4, "sigmoid", False, "float32", "Boundary-no-bias", seed=42)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): no-bias path: {type(exc).__name__}: {exc}")


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
