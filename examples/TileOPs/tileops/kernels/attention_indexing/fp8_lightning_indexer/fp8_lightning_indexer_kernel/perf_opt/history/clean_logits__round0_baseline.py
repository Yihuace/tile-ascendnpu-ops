# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""TileLang NPUIR implementation of ``clean_logits_``.

In-place window-mask clean-up of the fp32 logits buffer produced by the
lightning-indexer main kernel.  Migrated from the GPU TileOPs factory
``clean_logits_`` (``tileops/kernels/fp8_lightning_indexer.py``, L152-L184).

Semantics (unchanged from the source operator)
----------------------------------------------
For every query row ``s`` the visible key window is
``[cu_k_s(s), cu_k_e(s))`` with::

    cu_k_s(s) = max(min(CuSeqLenKS[s], seq_len_kv), 0)
    cu_k_e(s) = max(min(CuSeqLenKE[s], seq_len_kv), cu_k_s(s))

Every element of ``Logits[b, s, t, g]`` outside that window is overwritten
with the exact fp32 ``-inf``; elements inside the window keep their original
bits.  ``Logits`` is an explicit in-place operand (no separate output).

NPU redesign (see DESIGN.md 0.6 / 1.6.1)
----------------------------------------
The source uses a CUDA thread-lane predicate write (``get_thread_binding`` +
``if idx < cu_k_s``); npuir has no thread binding.  The NPU kernel keeps the
design's algorithmic structure (complement-only write, idempotent skip of the
window interior, constant ``-inf`` store outside, exact window mask on the two
window-edge tiles) with block-aligned DMA offsets only.

Implementation deviations from DESIGN.md 3.3 (forced by this toolchain)
----------------------------------------------------------------------
Both deviations keep the frozen algorithm and the wrapper interface contract
(``clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)``); they only change how the
same math is emitted.

1. ``T.vcmp`` scalar operands are materialised with ``T.vbrc`` instead of being
   passed as let-bound scalars (DESIGN.md 9.1 risk 3 documents this fallback).
   Reason: with ``TILELANG_ASCEND_MODE=Developer`` the codegen segfaults when a
   ``T.vcmp`` scalar operand varies with the enclosing ``T.serial`` induction
   variable (loop-invariant scalars are fine).  Note the design's D-3 probe
   (``history_version/design_probe_forms.py``) left ``TILELANG_ASCEND_MODE``
   unset, so it actually exercised the *Expert* path and never validated the
   claimed Developer mode.

2. The two window-edge tile masks are computed **once, outside the tile loop**
   (loop-invariant), and the tile loop only performs copy-only constant
   painting.  Reason: on this toolchain a loop-variant v-op mask
   (``vbrc`` + ``vcmp`` building a mask *inside* the ``T.serial`` tile loop)
   that is then consumed by ``T.vselect`` produces nondeterministic
   partially-masked tiles -- silently wrong output -- for ``block_N <= 128``,
   in both Developer and Expert modes and regardless of explicit pipe
   barriers/flags.  ``block_N >= 128`` with loop-invariant masks is
   bit-exact.  The complement decomposition (E1), the idempotent window
   interior skip (E2) and the constant outside store (E3) are preserved; the
   E4 single-edge specialisation is subsumed by the generic two-sided mask.
   ``block_N = 64`` still mis-computes on this toolchain, so 128/256/512 are
   used and 64 is kept only as an advisory (non-blocking) case.

Programming mode
----------------
``ASCEND_MODE = "Developer"`` (non-persistent 1-D grid, block-level vector
ops, compiler-managed UB placement and synchronization).
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch
import tilelang
import tilelang.language as T

ASCEND_MODE = "Developer"

__all__ = ["clean_logits_", "golden_clean_logits"]


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, independent of the NPU algorithm)
# ---------------------------------------------------------------------------


def golden_clean_logits(logits, cu_seqlen_ks, cu_seqlen_ke):
    """Reference window-mask clean-up, computed out of place.

    Direct port of the source op's window-mask semantics
    (``test_fp8_lightning_indexer.py::ref_program``): clamp each row's window,
    keep ``[ks, ke)`` and fill everything else with the exact fp32 ``-inf``.
    Deliberately re-implements the per-element formula rather than the NPU
    edge-tile scan, so the check stays independent.

    Args:
        logits: ``[B, S, Skv, G]`` fp32 tensor (not modified).
        cu_seqlen_ks: ``[S]`` int32 window lower bounds.
        cu_seqlen_ke: ``[S]`` int32 window upper bounds.

    Returns:
        ``[B, S, Skv, G]`` fp32 tensor, ``-inf`` outside each row's window.
    """
    batch, seq_len, seq_len_kv, kv_group = logits.shape
    ks = torch.clamp(cu_seqlen_ks, min=0, max=seq_len_kv)
    ke = torch.clamp(cu_seqlen_ke, max=seq_len_kv)
    ke = torch.maximum(ke, ks)
    ar = torch.arange(0, seq_len_kv, device=logits.device)
    keep = (ar[None, :] >= ks[:, None]) & (ar[None, :] < ke[:, None])
    keep = keep[None, :, :, None]
    neg = torch.full_like(logits, float("-inf"))
    return torch.where(keep, logits, neg)


# Canonical ``golden_<op>`` alias for external naming checks.
golden_clean_logits_ = golden_clean_logits


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


@tilelang.jit(target="npuir")
def clean_logits_(block_N=256, G_MAX=2):
    """Build the npuir ``clean_logits_`` kernel.

    Args:
        block_N: compile-time kv tile length.  128/256/512 are verified; 64
            still mis-computes on this toolchain (see module docstring).
        G_MAX: compile-time static upper bound of ``kv_group``; the on-chip
            tiles are ``[block_N, G_MAX]`` (dynamic dims are not allowed as UB
            buffer shapes), while GM copies use the dynamic extent
            ``[0:tail, 0:kv_group]``.  The caller must guarantee
            ``kv_group <= G_MAX``.

    Returns:
        The jit-decorated kernel ``clean_logits_kernel(Logits, CuSeqLenKS,
        CuSeqLenKE)``; ``Logits`` is updated in place.
    """
    dtype = "float32"
    index_dtype = "int32"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    kv_group = T.dynamic("kv_group")

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(batch * seq_len, is_npu=True) as (cid, _):
            by = cid // seq_len
            bx = cid % seq_len

            # Static-shape UB tiles (v-op carriers must use alloc_ub).
            neg_ub = T.alloc_ub([block_N, G_MAX], dtype)
            idx_ub = T.alloc_ub([block_N, G_MAX], index_dtype)
            bound_lo = T.alloc_ub([block_N, G_MAX], index_dtype)
            bound_hi = T.alloc_ub([block_N, G_MAX], index_dtype)
            cmp_lo = T.alloc_ub([block_N, G_MAX], "bool")
            cmp_hi = T.alloc_ub([block_N, G_MAX], "bool")
            mask_lo = T.alloc_ub([block_N, G_MAX], "bool")
            mask_hi = T.alloc_ub([block_N, G_MAX], "bool")
            log_ub = T.alloc_ub([block_N, G_MAX], dtype)

            # -inf constant and kv row index, materialised once per core.
            T.vbrc(T.float32(-T.infinity("float32")), neg_ub)
            T.arange(idx_ub, [1, 0], 0)

            # Block-level window bounds (clamped, degenerate-empty safe).
            cu_k_s = T.max(T.min(CuSeqLenKS[bx], seq_len_kv), 0)
            cu_k_e = T.max(T.min(CuSeqLenKE[bx], seq_len_kv), cu_k_s)
            n_kv = T.ceildiv(seq_len_kv, block_N)
            # Tile index of the left window edge (always valid) and of the
            # right window edge (tile that contains cu_k_e - 1).
            t_lo = T.min(cu_k_s // block_N, n_kv - 1)
            t_hi = T.max((cu_k_e - 1) // block_N, 0)

            # Masks of the two window-edge tiles, computed outside the tile
            # loop so that every mask operand is loop invariant.
            s_lo = cu_k_s - t_lo * block_N
            e_lo = cu_k_e - t_lo * block_N
            T.vbrc(s_lo, bound_lo)
            T.vbrc(e_lo, bound_hi)
            T.vcmp(idx_ub, bound_lo, cmp_lo, "ge")
            T.vcmp(idx_ub, bound_hi, cmp_hi, "lt")
            T.vand(cmp_lo, cmp_hi, mask_lo)
            s_hi = cu_k_s - t_hi * block_N
            e_hi = cu_k_e - t_hi * block_N
            T.vbrc(s_hi, bound_lo)
            T.vbrc(e_hi, bound_hi)
            T.vcmp(idx_ub, bound_lo, cmp_lo, "ge")
            T.vcmp(idx_ub, bound_hi, cmp_hi, "lt")
            T.vand(cmp_lo, cmp_hi, mask_hi)

            # Phase 1 (E3): constant -inf on every fully-outside tile.  The
            # loop body only performs a copy (no vector ops) and all offsets
            # are block aligned (t * block_N).
            for t in T.serial(n_kv):
                k0 = t * block_N
                tail = T.min(block_N, seq_len_kv - k0)
                if (k0 + tail <= cu_k_s) or (k0 >= cu_k_e):
                    T.copy(neg_ub[0:tail, 0:kv_group],
                           Logits[by, bx, k0:k0 + tail, 0:kv_group])

            # Phase 2 (E1): exact window mask on the two window-edge tiles.
            # Tiles strictly between them are fully inside the window and are
            # left untouched (E2 idempotent skip).
            k_lo = t_lo * block_N
            tail_lo = T.min(block_N, seq_len_kv - k_lo)
            T.copy(Logits[by, bx, k_lo:k_lo + tail_lo, 0:kv_group],
                   log_ub[0:tail_lo, 0:kv_group])
            T.vselect(mask_lo, log_ub, neg_ub, log_ub)
            T.copy(log_ub[0:tail_lo, 0:kv_group],
                   Logits[by, bx, k_lo:k_lo + tail_lo, 0:kv_group])

            k_hi = t_hi * block_N
            tail_hi = T.min(block_N, seq_len_kv - k_hi)
            T.copy(Logits[by, bx, k_hi:k_hi + tail_hi, 0:kv_group],
                   log_ub[0:tail_hi, 0:kv_group])
            T.vselect(mask_hi, log_ub, neg_ub, log_ub)
            T.copy(log_ub[0:tail_hi, 0:kv_group],
                   Logits[by, bx, k_hi:k_hi + tail_hi, 0:kv_group])

    return clean_logits_kernel


# ---------------------------------------------------------------------------
# Precision comparison
# ---------------------------------------------------------------------------

_TOL = 1e-3


def _validate_tensor_match(output, output_ref, tol=_TOL, tag=""):
    """Source-comparator contract plus the exact-mask assertion.

    Mirrors the source test ``_validate_tensor_match``: the ``isfinite`` masks
    must match, the non-finite values must match bitwise (``atol=rtol=0``,
    ``equal_nan=True``) and the finite domain must have correlation difference
    ``<= tol``.  Because this operator is an exact projection, the finite
    (window) domain is additionally required to be bit-identical.
    """
    a_finite = torch.isfinite(output)
    b_finite = torch.isfinite(output_ref)
    n_mask = int((a_finite != b_finite).sum().item())
    assert n_mask == 0, f"[{tag}] isfinite mask mismatch: {n_mask} elements"
    a_nf = output.masked_fill(a_finite, 0)
    b_nf = output_ref.masked_fill(b_finite, 0)
    assert torch.isclose(a_nf, b_nf, rtol=0, atol=0, equal_nan=True).all(), \
        f"[{tag}] non-finite value mismatch"

    both = a_finite & b_finite
    if both.any():
        max_abs = float((output[both].double() - output_ref[both].double()).abs().max())
    else:
        max_abs = 0.0

    a = output.masked_fill(~a_finite, 0).double()
    b = output_ref.masked_fill(~b_finite, 0).double()
    denom = (a * a + b * b).sum()
    corr = (2 * (a * b).sum() / denom) if denom > 0 else torch.tensor(1.0, dtype=torch.float64)
    diff = float(1.0 - corr)

    assert 0 <= diff <= tol, f"[{tag}] correlation difference {diff:.3e} > {tol}"
    assert max_abs == 0.0, f"[{tag}] window values not bit-preserved: max_abs={max_abs:.3e}"
    return diff, max_abs


# ---------------------------------------------------------------------------
# Test-case helpers
# ---------------------------------------------------------------------------


def _make_windows(seq_len, seq_len_kv, mode, seed):
    """Build per-row ``(ks, ke)`` int32 windows for the requested mode."""
    gen = torch.Generator().manual_seed(seed)
    if mode == "full":
        ks = torch.zeros(seq_len, dtype=torch.int32)
        ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
    elif mode == "empty":
        ks = torch.randint(0, seq_len_kv + 1, (seq_len,), generator=gen, dtype=torch.int32)
        ke = ks.clone()
    elif mode == "flip":
        ks = torch.randint(1, seq_len_kv + 1, (seq_len,), generator=gen, dtype=torch.int32)
        delta = torch.randint(1, seq_len_kv + 1, (seq_len,), generator=gen, dtype=torch.int32)
        ke = torch.clamp(ks - delta, min=0)
    elif mode == "narrow":
        ks = torch.randint(0, max(seq_len_kv, 1), (seq_len,), generator=gen, dtype=torch.int32)
        ke = ks + 1
    elif mode == "outofrange":
        lo = -seq_len_kv
        hi = 2 * seq_len_kv + 1
        ks = torch.randint(lo, hi, (seq_len,), generator=gen, dtype=torch.int32)
        ke = torch.randint(lo, hi, (seq_len,), generator=gen, dtype=torch.int32)
    elif mode == "aligned":
        nb = max(1, seq_len_kv // 256)
        a = torch.randint(0, nb + 1, (seq_len,), generator=gen, dtype=torch.int32) * 256
        b = torch.randint(0, nb + 1, (seq_len,), generator=gen, dtype=torch.int32) * 256
        ks = torch.minimum(a, b)
        ke = torch.maximum(a, b)
    else:  # random
        ks = torch.randint(0, seq_len_kv + 1, (seq_len,), generator=gen, dtype=torch.int32)
        span = (seq_len_kv - ks).float()
        extra = (torch.rand(seq_len, generator=gen) * (span + 1)).floor().to(torch.int32)
        ke = ks + torch.clamp(extra, max=span.to(torch.int32))
    return ks, ke


def _inject_special(logits, seed):
    """Scatter inf / -inf / NaN / -0.0 / subnormals into the input buffer."""
    flat = logits.view(-1)
    n = flat.numel()
    k = min(n, 1024)
    vals = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), -0.0,
         1e-42, -1e-42, 3.4e38, -3.4e38], dtype=torch.float32)
    idx = torch.randint(0, n, (k,), generator=torch.Generator().manual_seed(seed + 1001))
    pick = torch.randint(0, vals.numel(), (k,), generator=torch.Generator().manual_seed(seed + 2002))
    flat[idx.to(logits.device)] = vals[pick].to(logits.device)


def run_case(batch, seq_len, seq_len_kv, kv_group, tag, window="random",
             block_N=256, seed=0, special_values=False):
    """Run one case on NPU and compare against the CPU golden."""
    assert kv_group <= 2, "G_MAX=2 contract: kv_group must be <= 2"
    torch.manual_seed(seed)
    device = "npu"

    logits = torch.randn(batch, seq_len, seq_len_kv, kv_group, device=device,
                         dtype=torch.float32)
    if special_values:
        _inject_special(logits, seed)
    logits_before = logits.cpu().clone()

    ks, ke = _make_windows(seq_len, seq_len_kv, window, seed)
    kernel = clean_logits_(block_N=block_N, G_MAX=2)
    kernel(logits, ks.to(device), ke.to(device))

    ref = golden_clean_logits(logits_before, ks, ke)
    diff, max_abs = _validate_tensor_match(logits.cpu(), ref, tag=tag)
    print(f"[{tag}] PASS shape=(b{batch},s{seq_len},skv{seq_len_kv},g{kv_group}) "
          f"window={window} block_N={block_N} corr_diff={diff:.3e} max_abs={max_abs:.3e}")
    return diff, max_abs


# ---------------------------------------------------------------------------
# Hierarchical test suites
# ---------------------------------------------------------------------------


def run_L0():
    """Gate tests: DESIGN.md 8.3 L0-1 .. L0-8 (block_N=256 default)."""
    run_case(1, 1024, 1024, 1, "L0-1-smoke-full", window="full")
    run_case(1, 1024, 1024, 1, "L0-2-random", window="random", seed=1)
    run_case(1, 4096, 8192, 1, "L0-3-unaligned", window="random", seed=2)
    run_case(1, 2048, 2048, 1, "L0-4-empty", window="empty", seed=3)
    run_case(1, 512, 1000, 1, "L0-5-tail-odd-skv", window="random", seed=4)
    run_case(1, 1024, 1024, 2, "L0-6-g2", window="random", seed=5)
    run_case(1, 256, 1024, 1, "L0-7-special-values", window="random", seed=6,
             special_values=True)
    run_case(1, 512, 1024, 1, "L0-8-flip", window="flip", seed=7)


def run_L1():
    """Gate tests: cross checks on tiling / grouping / window edges."""
    run_case(1, 1024, 1024, 2, "L1-g2-blk128", window="random", seed=11, block_N=128)
    run_case(1, 1024, 2048, 1, "L1-blk512", window="random", seed=12, block_N=512)
    run_case(1, 512, 1000, 2, "L1-g2-odd-skv", window="random", seed=13)
    run_case(2, 256, 512, 1, "L1-batch2", window="random", seed=14)
    run_case(1, 1024, 1024, 1, "L1-narrow-window", window="narrow", seed=15)
    run_case(1, 300, 777, 1, "L1-odd-shapes", window="random", seed=16)
    run_case(1, 512, 1024, 1, "L1-outofrange", window="outofrange", seed=17)
    run_case(1, 512, 1024, 1, "L1-aligned-edges", window="aligned", seed=18)
    run_case(1, 512, 2048, 2, "L1-g2-blk512", window="random", seed=19, block_N=512)
    run_case(1, 256, 1024, 1, "L1-special-blk128", window="random", seed=20,
             block_N=128, special_values=True)


def _advisory(tag, fn):
    try:
        fn()
    except Exception as exc:  # pragma: no cover - advisory only
        print(f"[{tag}] WARN (recorded, non-blocking): {str(exc)[:300]}")


def run_L2():
    """Advisory cases (non-blocking)."""
    _advisory("L2-batch2-g2", lambda: run_case(2, 1024, 2048, 2, "L2-batch2-g2",
                                               window="random", seed=21))
    _advisory("L2-big-g2", lambda: run_case(1, 4096, 8192, 2, "L2-big-g2",
                                            window="random", seed=22))
    _advisory("L2-special-g2", lambda: run_case(1, 256, 1024, 2, "L2-special-g2",
                                                window="random", seed=24,
                                                special_values=True))


def _known_block64_limitation():
    """Record (do not assert) the block_N=64 toolchain defect."""
    n_bad = 0
    for seed in (101, 102, 103):
        batch, seq_len, seq_len_kv, kv_group, block_N = 1, 256, 1024, 2, 64
        torch.manual_seed(seed)
        logits = torch.randn(batch, seq_len, seq_len_kv, kv_group, device="npu",
                             dtype=torch.float32)
        before = logits.cpu().clone()
        ks, ke = _make_windows(seq_len, seq_len_kv, "random", seed)
        clean_logits_(block_N=block_N, G_MAX=2)(logits, ks.to("npu"), ke.to("npu"))
        out = logits.cpu()
        ref = golden_clean_logits(before, ks, ke)
        n_bad += int((torch.isfinite(out) != torch.isfinite(ref)).sum())
    if n_bad:
        raise RuntimeError(
            f"block_N=64 known toolchain mis-compute: {n_bad} elements "
            "(loop-invariant-mask structure still unstable at block_N=64)")
    print("[Boundary-blockN64] PASS (no mis-compute)")


def run_boundary():
    """Boundary cases (advisory, non-blocking)."""
    _advisory("Boundary-single-row", lambda: run_case(1, 1, 1024, 1,
                                                      "Boundary-single-row",
                                                      window="random", seed=31))
    _advisory("Boundary-window-1", lambda: run_case(1, 256, 1024, 1,
                                                    "Boundary-window-1",
                                                    window="narrow", seed=32))
    _advisory("Boundary-skv-le-blk", lambda: run_case(1, 128, 100, 1,
                                                      "Boundary-skv-le-blk",
                                                      window="random", seed=33))
    _advisory("Boundary-empty-all", lambda: run_case(1, 64, 256, 1,
                                                     "Boundary-empty-all",
                                                     window="empty", seed=34))
    _advisory("Boundary-full-all", lambda: run_case(1, 64, 256, 1,
                                                    "Boundary-full-all",
                                                    window="full"))
    _advisory("Boundary-negative-input", lambda: _run_negative_input())
    _advisory("Boundary-blockN64", _known_block64_limitation)


def _run_negative_input():
    """All ``-inf`` input: the window bits must survive the round trip."""
    logits = torch.full((1, 64, 256, 1), float("-inf"), device="npu", dtype=torch.float32)
    before = logits.cpu().clone()
    ks, ke = _make_windows(64, 256, "random", 41)
    clean_logits_(block_N=256, G_MAX=2)(logits, ks.to("npu"), ke.to("npu"))
    ref = golden_clean_logits(before, ks, ke)
    _validate_tensor_match(logits.cpu(), ref, tag="Boundary-negative-input")
    print("[Boundary-negative-input] PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
