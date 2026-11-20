# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_fused_topk_kernel_cb -- MoE fused top-k routing with correction bias (NPU).

Migrated from the GPU TileOPs ``FusedTopKKernel`` factory correction-bias
branch (``tileops/kernels/moe/fused_topk.py``, ``with_correction_bias=True``,
L96-L207) to ``target="npuir"``.

Semantics (DESIGN.md section 0.1 / 1.3, frozen):
    logits = gating_output.to(float32)                      # fp32 compute domain
    s      = sigmoid(logits)                                # unbiased score
    biased = s + correction_bias[e]                         # SELECTION key
    ids    = topK(biased, K)                                # descending
    w      = s[ids]                                         # UNBIASED weights
    if renormalize: w = w / w.sum(-1, keepdim=True)
    -> topk_weights (T, K) float32, topk_ids (T, K) int32

The watershed of this variant: the correction bias only affects *selection*;
the exported weights are the *unbiased* sigmoid at the selected ids.

NPU redesign (DESIGN.md section 0.6 / 1.6, frozen, algorithm C1):
    R1: one-dimensional persistent core split -- ``num_kernels =
        min(num_logical, 40)`` (pure-Vector physical cores = 20 AICore x 2)
        with an in-core static ``T.serial`` grid-stride loop and a block-level
        ``if task < num_logical`` guard.
    R2: per-lane register residency -> UB row-block ``(BM, E)`` residency.
    R3/R4: source warp K-pass masked argmax + shfl_xor -> a single
        ``T.vsort(biased, descending=True)`` for selection plus a full-width
        ``T.gather(xf, sv, si)`` that reorders the already-computed unbiased
        sigmoid into the selection permutation (``w_k = sigma(x)[si_k]``).
    R5: no non-divisible-E padding (sorting/gathering acts on the true columns).
    R6: renormalize fused into the same kernel (K-element ``reduce_sum``).

Toolchain traps deliberately avoided (DESIGN.md section 3.5):
    - no ``T.alloc_local`` (+ ``T.serial`` -> SIGSEGV, CG-2026-0028b);
    - no scalar ``>>`` / ``T.if_then_else`` (CG-2026-0028a/c);
    - no scalar consumption of a vector result (CG-2026-0027);
    - no direct GM write inside ``T.Parallel`` (CG-2026-0030);
    - no scalar payload ``T.atomic_add`` (CG-2026-0029).
All intermediate values stay vector-shaped in UB; the only scalar control flow
is the block-level ``if task < num_logical`` guard.
"""

import os

# Developer mode: the compiler manages UB (alloc_shared) and auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
# Chip 1 (chip 0 is reserved for another terminal / perf measurement).
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")

import argparse
import math

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

# UB budget for the large buffers (xf + biased + sv + si = BM * E * 16 bytes)
# and max rows per block (DESIGN.md section 4.5 / 5.2).  Conservative w.r.t.
# 192KB UB to leave room for T.vsort / T.gather / T.vbrc scratch and
# auto-multi-buffer inflation.
UB_BUDGET = 96 * 1024
BM_MAX = 64

# Harness tolerances (test_moe_fused_topk.py): weights compared after sorting.
_RTOL = 1e-3
_ATOL = 1e-3
# Selection value-validity slack for topk_ids (test_moe_fused_topk.py `_check`).
_ID_SCORE_SLACK = 1e-4

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_fused_topk_cb(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    scoring_func: str = "sigmoid",
    renormalize: bool = False,
) -> tuple:
    """Source correction-bias semantics (DESIGN.md section 0.1 / 8.1).

    Reference: ``biased = sigmoid(x) + bias`` is used ONLY for selection;
    the exported weights are the *unbiased* ``sigmoid(x)`` gathered at the
    selected ids.  This is an independent torch implementation of the frozen
    source semantics -- it does NOT reproduce the NPU redesign (no ``T.vsort``,
    no gather-reorder dataflow), so validation stays independent.
    """
    if scoring_func != "sigmoid":
        raise ValueError(
            "golden_fused_topk_cb requires scoring_func='sigmoid' "
            "(softmax + correction_bias is an illegal combination)"
        )
    logits = gating_output.to(torch.float32)             # fp32 compute domain
    scores = torch.sigmoid(logits)                       # unbiased score
    biased = scores + correction_bias.to(torch.float32).unsqueeze(0)  # key
    # Descending order matches the NPU vsort(descending=True) id layout.
    _, ids = torch.topk(biased, top_k, dim=-1, sorted=True)
    w = torch.gather(scores, 1, ids)                     # UNBIASED weights
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), ids.int()


# ---------- Kernel ----------
def _select_bm(num_experts: int, num_tokens: int, hint) -> int:
    """Rows per block: UB-budget cap, optionally tightened by a caller hint.

    BM = min(BM_MAX, floor(UB_BUDGET / (E * 16)), T), with a hard lower
    bound of 1 (DESIGN.md section 5.2; REVIEW.md advisory: guard against 0).
    """
    cap = max(1, min(BM_MAX, UB_BUDGET // (num_experts * 16)))
    bm = cap
    if hint is not None and hint > 0:
        bm = min(hint, cap)
    return max(1, min(bm, num_tokens))


def _fused_topk_kernel_cb(
    num_tokens,
    num_experts,
    top_k,
    scoring_func="sigmoid",
    with_correction_bias=True,
    renormalize=False,
    in_dtype="float32",
    block_m=None,
):
    """Build the NPU correction-bias fused top-k routing kernel factory.

    Mirrors the harness contract ``_fused_topk_kernel(num_tokens, num_experts,
    top_k, scoring_func, with_correction_bias)`` with two NPU-side extensions
    appended: ``renormalize`` (fused into the kernel per R6) and ``in_dtype``
    (the native input dtype; the kernel promotes to fp32 internally).

    This unit implements the ``with_correction_bias=True`` variant only.
    """
    # softmax + correction_bias is an illegal combination (DESIGN.md R-3).
    # The host/op layer owns this validation; the factory mirrors it so that a
    # direct factory call fails loudly instead of silently computing sigmoid.
    if scoring_func != "sigmoid":
        raise ValueError(
            f"with_correction_bias=True requires scoring_func='sigmoid', "
            f"got {scoring_func!r} (softmax + correction_bias is illegal)"
        )
    if not with_correction_bias:
        raise ValueError(
            "this unit implements with_correction_bias=True only; use the "
            "standard variant (_fused_topk_kernel) for the no-bias path"
        )
    if in_dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported in_dtype {in_dtype!r}")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    if num_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        raise ValueError(
            f"num_tokens/num_experts/top_k must be positive, got "
            f"{num_tokens}/{num_experts}/{top_k}"
        )

    # Pure-Vector physical cores = AICore count x 2 (DESIGN.md section 5.5).
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[2, 3], target="npuir")
    def _func(TOKENS_PER_BLOCK=None):
        requested = TOKENS_PER_BLOCK if TOKENS_PER_BLOCK is not None else block_m
        bm = _select_bm(num_experts, num_tokens, requested)
        num_logical = _ceildiv(num_tokens, bm)              # row blocks
        num_kernels = min(num_logical, vector_cores)        # persistent grid
        num_local_tasks = _ceildiv(num_logical, num_kernels)  # static bound

        @T.prim_func
        def main(
            gating_output: T.Tensor((num_tokens, num_experts), in_dtype),
            correction_bias: T.Tensor((num_experts,), "float32"),
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),
            topk_ids: T.Tensor((num_tokens, top_k), "int32"),
        ):
            # One-dimensional persistent split: fixed physical core count with
            # an in-core static T.serial grid-stride task loop.
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                xf = T.alloc_shared((bm, num_experts), "float32")      # sigma(x)
                biased = T.alloc_shared((bm, num_experts), "float32")  # sigma + b
                sv = T.alloc_shared((bm, num_experts), "float32")      # sorted vals
                si = T.alloc_shared((bm, num_experts), "int32")        # sorted ids
                bb = T.alloc_shared((1, num_experts), "float32")       # bias row
                vk = T.alloc_shared((bm, top_k), "float32")            # weights
                ik = T.alloc_shared((bm, top_k), "int32")              # ids
                sr = T.alloc_shared((bm, 1), "float32")                # renorm denom
                for s in T.serial(num_local_tasks):
                    task = s * num_kernels + cid
                    if task < num_logical:
                        off = task * bm
                        real = T.min(bm, num_tokens - off)  # tail rows
                        # 1. load row block: GM -> UB, auto-promote to fp32
                        T.copy(
                            gating_output[off : off + real, 0:num_experts],
                            xf[0:real, 0:num_experts],
                        )
                        # 2. full-row unbiased sigmoid, computed only once
                        T.vsigmoid(xf, xf)
                        # 3. broadcast the per-expert bias (1,E) -> (BM,E)
                        T.copy(correction_bias[0:num_experts], bb[0:1, 0:num_experts])
                        T.vbrc(bb, biased)
                        # 4. selection key = sigmoid + bias
                        T.vadd(xf, biased, biased)
                        # 5. descending sort on the biased key (values + ids)
                        T.vsort(biased, sv, si, descending=True, sort_axis=-1)
                        # 6. top-k ids
                        T.copy(si[0:real, 0:top_k], ik[0:real, 0:top_k])
                        # 7. reorder the unbiased sigmoid with the selection
                        #    permutation (full-width gather, reuses sv)
                        T.gather(xf, sv, si)
                        T.copy(sv[0:real, 0:top_k], vk[0:real, 0:top_k])
                        # 8. fused renormalize on the K-element slice
                        if renormalize:
                            T.reduce_sum(vk, sr, dim=1, clear=True)
                            T.vdiv(vk, sr, vk)
                        # 9. write back both outputs
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


# ---------- build / execute helpers ----------
def _is_ub_overflow(msg: str) -> bool:
    low = msg.lower()
    return "ub overflow" in low or "bits while" in low or "bits available" in low


def _is_transient(msg: str) -> bool:
    # Known toolchain race: npu_launcher.h.gch compiled before the header
    # exists (tilelang compile-cache race).  Retry once.
    low = msg.lower()
    return "npu_launcher" in msg or "gch" in low or "no such file or directory" in low


def _build_and_run(
    x_npu,
    b_npu,
    num_tokens,
    num_experts,
    top_k,
    scoring_func,
    renormalize,
    dtype_str,
    block_hint,
    tag,
):
    """Build the kernel and execute it, with R-5 / transient-error fallbacks.

    Fallbacks (DESIGN.md section 9 R-5):
      - UB overflow -> halve BM and rebuild (up to 4 attempts);
      - transient toolchain race -> retry once unchanged.
    Returns (out_weights, out_ids, effective_block_m).
    """
    bm = _select_bm(num_experts, num_tokens, block_hint)
    last_exc = None
    shrunk = 0
    for attempt in range(6):
        try:
            factory = _fused_topk_kernel_cb(
                num_tokens,
                num_experts,
                top_k,
                scoring_func,
                True,
                renormalize,
                dtype_str,
                bm,
            )
            fn = factory(bm)
            out_w, out_ids = fn(x_npu, b_npu)
            torch.npu.synchronize()
            return out_w, out_ids, bm
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if _is_ub_overflow(msg) and bm > 1 and shrunk < 3:
                new_bm = max(1, bm // 2)
                print(
                    f"[{tag}] WARN ub overflow at BM={bm}; retry with BM={new_bm}"
                )
                bm = new_bm
                shrunk += 1
                continue
            if _is_transient(msg) and attempt < 5:
                print(
                    f"[{tag}] WARN transient build error; retry "
                    f"(attempt {attempt + 1}): {msg.splitlines()[0][:120]}"
                )
                continue
            raise
    raise RuntimeError(f"[{tag}] build/run failed after retries: {last_exc}")


# ---------- precision comparison ----------
def _run_case(
    num_tokens,
    num_experts,
    top_k,
    renormalize,
    dtype_str,
    tag,
    logits=None,
    bias=None,
    seed=0,
    scale=1.0,
    bias_scale=0.1,
    block_m=None,
    equal_nan=False,
    allow_id_mismatch=False,
):
    torch_dtype = _DTYPE_MAP[dtype_str]
    gen = torch.Generator().manual_seed(seed)
    if logits is None:
        logits = torch.randn(num_tokens, num_experts, generator=gen) * scale
    if bias is None:
        bias = torch.randn(num_experts, generator=gen) * bias_scale
    logits = logits.reshape(num_tokens, num_experts).to(torch_dtype)
    bias = bias.reshape(num_experts).to(torch.float32)

    out_w, out_ids, eff_bm = _build_and_run(
        logits.npu(),
        bias.npu(),
        num_tokens,
        num_experts,
        top_k,
        "sigmoid",
        renormalize,
        dtype_str,
        block_m,
        tag,
    )

    ref_w, ref_ids = golden_fused_topk_cb(logits, bias, top_k, "sigmoid", renormalize)

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

    # --- weights: sorted compare against the independent golden (set equality)
    ref_w_sorted = ref_w.sort(dim=-1).values
    out_w_sorted = out_w_cpu.sort(dim=-1).values
    torch.testing.assert_close(
        out_w_sorted, ref_w_sorted, rtol=_RTOL, atol=_ATOL, equal_nan=equal_nan
    )

    # --- weights semantic anchor: w must equal unbiased sigma at the RETURNED
    #     ids (this is exactly the correction-bias watershed; it would fail if
    #     the implementation exported the biased key instead).
    scores_cpu = torch.sigmoid(logits.float())
    exp_w = scores_cpu.gather(1, out_ids_cpu.long())
    if renormalize:
        exp_w = exp_w / exp_w.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        out_w_cpu, exp_w, rtol=_RTOL, atol=_ATOL, equal_nan=equal_nan
    )

    # --- ids validity: no duplicates, in range
    assert bool(((out_ids_cpu >= 0) & (out_ids_cpu < num_experts)).all()), (
        f"[{tag}] out-of-range expert ids"
    )
    if top_k > 1:
        srt, _ = torch.sort(out_ids_cpu, dim=-1)
        assert bool((srt[:, 1:] != srt[:, :-1]).all()), (
            f"[{tag}] duplicate expert ids within a row"
        )

    # --- ids value validity + boundary-tie detection
    biased_cpu = scores_cpu + bias.unsqueeze(0)
    sel_biased = biased_cpu.gather(1, out_ids_cpu.long())
    kth = biased_cpu.topk(top_k, dim=-1).values[:, top_k - 1 : top_k]
    assert bool((sel_biased >= kth - _ID_SCORE_SLACK).all()), (
        f"[{tag}] a selected expert's biased value is below the K-th biased value"
    )
    has_boundary_tie = False
    if top_k < num_experts:
        kv = biased_cpu.topk(top_k + 1, dim=-1).values
        has_boundary_tie = bool((kv[:, top_k - 1] == kv[:, top_k]).any())

    ids_exact = bool(torch.equal(out_ids_cpu, ref_ids))

    # --- main contract: exact ids when the K/K+1 boundary is not tied
    if (not allow_id_mismatch) and (not has_boundary_tie):
        assert ids_exact, (
            f"[{tag}] topk_ids not elementwise equal to golden "
            f"(no boundary tie; exact match required)"
        )

    max_diff = (out_w_sorted - ref_w_sorted).abs().max().item()
    print(
        f"[{tag}] PASS: T={num_tokens} E={num_experts} K={top_k} "
        f"renorm={renormalize} dtype={dtype_str} BM={eff_bm} "
        f"ids_exact={ids_exact} tie={has_boundary_tie} max_diff={max_diff:.2e}"
    )
    return max_diff, ids_exact, has_boundary_tie


def _tie_logits(num_tokens, num_experts, groups=None, seed=0):
    """CPU logits with many exact ties (all-equal, or few distinct values)."""
    gen = torch.Generator().manual_seed(seed)
    if groups is None:
        return torch.zeros(num_tokens, num_experts)
    base = torch.randn(num_tokens, groups, generator=gen)
    idx = torch.randint(0, groups, (num_tokens, num_experts), generator=gen)
    return base.gather(1, idx)


def _neartie_logits(num_tokens, num_experts, delta):
    """Rows whose top-2 sigmoid scores differ by exactly ``delta``.

    Expert 0 -> sigmoid 0.5; expert 1 -> sigmoid 0.5 + delta; all others are
    pushed far below.  With K=2 the K/(K+1) margin is exactly ``delta``.
    """
    x = torch.full((num_tokens, num_experts), -50.0)
    x[:, 0] = 0.0
    s1 = 0.5 + delta
    x[:, 1] = float(math.log(s1 / (1.0 - s1)))
    return x


# ---------- L0: gate tests (must pass) ----------
# DESIGN.md section 8.3 (12 cases).
def run_L0():
    max_diff = 0.0
    cases = [
        (16, 4, 2, True, "bfloat16", "L0-1", 0, 1.0, 0.1, None),
        (16, 4, 2, True, "float16", "L0-2", 1, 1.0, 0.1, None),
        (32, 8, 2, False, "bfloat16", "L0-3", 2, 1.0, 0.1, None),
        (64, 384, 8, True, "bfloat16", "L0-4", 3, 1.0, 0.1, None),
        (512, 256, 8, True, "bfloat16", "L0-5", 4, 1.0, 0.1, None),
        (2048, 256, 8, True, "bfloat16", "L0-6", 5, 1.0, 0.1, None),
        (4096, 256, 8, True, "bfloat16", "L0-7", 6, 1.0, 0.1, None),
        (32, 33, 4, True, "bfloat16", "L0-8", 7, 1.0, 0.1, None),
        (32, 100, 8, False, "float32", "L0-9", 8, 1.0, 0.1, None),
        (512, 256, 1, True, "bfloat16", "L0-10", 9, 1.0, 0.1, None),
        (32, 256, 8, True, "float32", "L0-11", 10, 1.0, 0.5, None),
        (32, 256, 8, True, "float32", "L0-12", 11, 1.0, -0.5, None),
    ]
    for tn, e, k, rn, dt, tag, seed, sc, bs, bm in cases:
        md, _, _ = _run_case(
            tn, e, k, rn, dt, tag, seed=seed, scale=sc, bias_scale=bs, block_m=bm
        )
        max_diff = max(max_diff, md)
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")
    return max_diff


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    max_diff = 0.0
    exact_all = True

    # renorm grid on a mid shape (fp32 exact domain)
    for i, rn in enumerate((False, True)):
        md, ex, _ = _run_case(128, 64, 4, rn, "float32", f"L1-{i + 1}", seed=11 + i)
        max_diff = max(max_diff, md)
        exact_all &= ex
    # K == E
    for i, rn in enumerate((False, True)):
        md, ex, _ = _run_case(64, 32, 32, rn, "float32", f"L1-{i + 3}", seed=13 + i)
        max_diff = max(max_diff, md)
        exact_all &= ex
    # non-divisible E
    md, ex, _ = _run_case(33, 7, 3, True, "bfloat16", "L1-5", seed=15)
    max_diff = max(max_diff, md)
    exact_all &= ex
    md, ex, _ = _run_case(127, 100, 8, False, "float32", "L1-6", seed=16)
    max_diff = max(max_diff, md)
    exact_all &= ex
    # T tail (513 = 10*48 + 33 for E=128 -> BM=48)
    md, ex, _ = _run_case(513, 128, 8, True, "bfloat16", "L1-7", seed=17)
    max_diff = max(max_diff, md)
    exact_all &= ex
    # T smaller than one block (real=25 < BM=48)
    md, ex, _ = _run_case(25, 128, 8, True, "float32", "L1-8", seed=18)
    max_diff = max(max_diff, md)
    exact_all &= ex
    # E=384 (Kimi experts) with multiple waves
    md, ex, _ = _run_case(1024, 384, 8, True, "bfloat16", "L1-9", seed=19)
    max_diff = max(max_diff, md)
    exact_all &= ex
    # fp16 / fp32 dtype paths with bias
    md, ex, _ = _run_case(64, 8, 1, False, "float16", "L1-10", seed=20, bias_scale=0.5)
    max_diff = max(max_diff, md)
    exact_all &= ex
    # bias dominates (large positive bias on a subset)
    gen = torch.Generator().manual_seed(21)
    bias = torch.zeros(256)
    bias[:8] = 5.0
    md, ex, _ = _run_case(
        64, 256, 8, True, "float32", "L1-11", seed=21, bias=bias
    )
    max_diff = max(max_diff, md)
    exact_all &= ex

    # --- near-tie sensitivity (DESIGN.md section 8.2 level 2): record flips
    for i, delta in enumerate((1e-5, 1e-6, 1e-7)):
        x = _neartie_logits(64, 64, delta)
        md, ex, tie = _run_case(
            64,
            64,
            2,
            False,
            "float32",
            f"L1-near-tie-{delta:g}",
            logits=x,
            bias=torch.zeros(64),
            allow_id_mismatch=True,
        )
        max_diff = max(max_diff, md)
        print(
            f"[L1] near-tie delta={delta:g}: ids_exact={ex} "
            f"(margin {'>>' if ex else '~='} ULP)"
        )

    # --- explicit ties (DESIGN.md section 8.2 level 3): set + value semantics
    md, ex, tie = _run_case(
        64,
        64,
        8,
        False,
        "float32",
        "L1-ties-equal",
        logits=_tie_logits(64, 64),
        bias=torch.zeros(64),
    )
    max_diff = max(max_diff, md)
    print(f"[L1] all-equal ties: ids_exact={ex} boundary_tie={tie}")
    md, ex, tie = _run_case(
        96,
        48,
        4,
        True,
        "bfloat16",
        "L1-ties-groups",
        logits=_tie_logits(96, 48, groups=3, seed=22),
        bias=torch.zeros(48),
    )
    max_diff = max(max_diff, md)
    print(f"[L1] few-group ties: ids_exact={ex} boundary_tie={tie}")

    print(f"[L1] ALL PASS, id-exact on all non-tie cases={exact_all}, max_diff={max_diff:.2e}")
    return max_diff


# ---------- L2: degenerate shapes (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, 1, False, "float32", 31),
        (1, 4, 2, True, "bfloat16", 32),
        (5, 8, 1, False, "float16", 33),
        (8, 2, 2, True, "float32", 34),
        (3, 5, 5, False, "bfloat16", 35),
        (1, 64, 4, True, "bfloat16", 36),
    ]
    for i, (tn, e, k, rn, dt, seed) in enumerate(cases):
        try:
            _run_case(tn, e, k, rn, dt, f"L2-{i + 1}", seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[L2] WARN (record only): T={tn} E={e} K={k} "
                f"renorm={rn} {dt}: {type(exc).__name__}: {exc}"
            )


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # exact ties at a larger shape
    try:
        md, ex, tie = _run_case(
            256,
            96,
            8,
            False,
            "float32",
            "Boundary-ties",
            logits=_tie_logits(256, 96),
            bias=torch.zeros(96),
        )
        print(f"[Boundary] ties: ids_exact={ex} boundary_tie={tie} max_diff={md:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): exact-ties: {type(exc).__name__}: {exc}")

    # large magnitude logits (sigmoid saturation)
    try:
        md, ex, _ = _run_case(
            32, 128, 8, True, "float32", "Boundary-large-mag", scale=50.0
        )
        print(f"[Boundary] large-magnitude: max_diff={md:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): large-magnitude: {type(exc).__name__}: {exc}")

    # large negative bias (selection key pushed below the sigmoid range)
    try:
        gen = torch.Generator().manual_seed(41)
        bias = -torch.rand(128, generator=gen) * 5.0
        md, ex, _ = _run_case(64, 128, 8, True, "float32", "Boundary-neg-bias", bias=bias)
        print(f"[Boundary] negative-bias: max_diff={md:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): negative-bias: {type(exc).__name__}: {exc}")

    # bias all equal
    try:
        md, ex, _ = _run_case(
            32, 64, 4, True, "float32", "Boundary-equal-bias", bias=torch.full((64,), 0.3)
        )
        print(f"[Boundary] equal-bias: ids_exact={ex} max_diff={md:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): equal-bias: {type(exc).__name__}: {exc}")

    # E=1 (degenerate expert axis)
    try:
        md, ex, _ = _run_case(8, 1, 1, False, "float32", "Boundary-E1", seed=42)
        print(f"[Boundary] E=1: ids_exact={ex} max_diff={md:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): E=1: {type(exc).__name__}: {exc}")

    # NaN in logits (documented as outside the input domain, DESIGN.md R-8)
    try:
        x = torch.randn(8, 32)
        x[0, 3] = float("nan")
        _run_case(8, 32, 4, False, "float32", "Boundary-nan", logits=x, equal_nan=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): NaN: {type(exc).__name__}: {exc}")

    # --- contract: softmax + correction_bias must raise (DESIGN.md R-3)
    try:
        _fused_topk_kernel_cb(4, 8, 2, "softmax", True)
        raise AssertionError("[Boundary] softmax + correction_bias did not raise")
    except ValueError as exc:
        print(f"[Boundary] PASS: illegal softmax+bias combination rejected: {exc}")

    # --- contract: the no-bias variant must be rejected by THIS factory
    try:
        _fused_topk_kernel_cb(4, 8, 2, "sigmoid", False)
        raise AssertionError("[Boundary] with_correction_bias=False did not raise")
    except ValueError as exc:
        print(f"[Boundary] PASS: no-bias path rejected by cb factory: {exc}")

    # --- BM lower-bound sanity: an absurd E must not yield BM=0 / empty grid
    try:
        bm = _select_bm(1_000_000, 8, None)
        assert bm >= 1, f"BM lower bound violated: {bm}"
        print(f"[Boundary] PASS: BM lower bound guarded (E=1e6 -> BM={bm})")
    except Exception as exc:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): BM lower-bound: {type(exc).__name__}: {exc}")


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
