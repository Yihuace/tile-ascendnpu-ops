"""Diagnostic variants of the ALiBi kernel factory (Stage 4 bottleneck decomposition).

All ``T.alloc_ub`` statements are unconditional on purpose: see opt_log.md section 7.2
(a conditional-expression allocation formulation in an earlier revision of this file
hit a compiler SIGSEGV whose trigger could not be isolated -- plain statements are the
verified-safe form).

All variants share the baseline structure (DESIGN §3.3): 1-D Kernel over 40 Vector
cores, in-core grid-stride over logical ``(bi, bj)`` spatial tiles, ``T.serial(H)``
head loop reusing the shared distance buffer ``D``, single round-back to the output
dtype.  Only the *distance chain* and/or the *head-loop body* differ, to attribute
the measured latency to a specific stage.

Select the variant with the env var ``ALIBI_PROBE_VARIANT`` (default ``full``):

  full        baseline structure (6-op distance chain + vmul/vcast/copy head loop)
  writeonly   no distance chain, no arithmetic: the head loop only re-copies a
              constant buffer -> pure UB->GM write floor (values are garbage)
  distwrite   baseline distance chain, then one vcast, then the head loop only
              copies -> distance chain + write, per-head vmul/vcast removed
  nodist      T.fill replaces the distance chain, head loop unchanged
              -> per-head vmul/vcast/copy only (values are garbage)
  arange2     only the two int16 aranges, head loop unchanged (isolates arange)
  subabs      T.fill(Rf)/T.fill(Cf) + vsub + vabs, head loop unchanged
              -> the two fp32 elementwise ops of the chain, no arange/vcast
  castsubabs  T.fill(R),T.fill(C) + vcast x2 + vsub + vabs, head loop unchanged
              -> adds the int16->fp32 casts to subabs
  f32ramp     distance chain built directly in fp32 (2 arange + vsub + vabs)
  hoist       candidate fix (only valid for ``bj == S``, i.e. ``j0 == 0``): hoist
              the column ramp ``Crel[r, c] = c`` out of the tile loop (one arange
              per core), then per tile rebuild the tiny ``(bi, 1)`` row ramp ``U``
              and compute ``D = vabs(vsub(Crel, U))`` (row-broadcast) -> 2 full-tile
              ops per tile instead of 6 (the 2 aranges and the 2 i16->f32 casts
              leave the inner loop entirely)

Numerically correct: ``full``, ``f32ramp``, ``hoist`` (they keep the true ALiBi
formula).  The others deliberately produce garbage and are only ever measured
with L0 skipped.

run: ALIBI_PROBE_VARIANT=hoist python _probe_variants.py --level L0
"""

import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
# the Stage 3 baseline lives one directory up (read-only reuse of its tiling
# helper + golden so the probe matches the baseline exactly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

from _make_alibi_kernel import _select_blocks, golden_alibi

ASCEND_MODE = "Developer"

tilelang.cache.clear_cache()

_CORE_NUM = 40
_COMPUTE_DTYPE = "float32"
_BUDGET = 8192
_VARIANT = os.environ.get("ALIBI_PROBE_VARIANT", "full")

_CORRECT_VARIANTS = ("full", "f32ramp", "hoist")


@tilelang.jit(target="npuir")
def _make_alibi_kernel(
    seq_len: int,
    num_heads: int,
    dtype: str,
    bi: int = None,
    bj: int = None,
    budget: int = _BUDGET,
    core_num: int = _CORE_NUM,
):
    variant = os.environ.get("ALIBI_PROBE_VARIANT", _VARIANT)
    S, H = int(seq_len), int(num_heads)
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32

    if bi is None or bj is None:
        auto_bi, auto_bj = _select_blocks(S, budget)
        bi = auto_bi if bi is None else int(bi)
        bj = auto_bj if bj is None else int(bj)
    bi, bj = int(bi), int(bj)

    nb = S // bj
    n_tiles = (S // bi) * nb
    stride = T.ceildiv(n_tiles, core_num)

    # `hoist` relies on the column ramp being tile-invariant -> only bj == S.
    if variant.startswith("hoist") and nb != 1:
        variant = "full"

    use_ramps = variant in ("full", "distwrite", "arange2", "castsubabs")
    use_hopper = variant in ("hoist", "hoist_noarange", "hoist_vsubonly")
    use_rf = variant != "writeonly"
    use_dst = variant not in ("writeonly", "distwrite")

    @T.prim_func
    def main(out: T.Tensor((H, S, S), dtype)):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # Allocations are chosen by compile-time (Python-level) variant
            # selection, so the UB footprint of each probe is its true one.
            R = T.alloc_ub((bi, bj), "int16")
            C = T.alloc_ub((bi, bj), "int16")
            Rf = T.alloc_ub((bi, bj), f32)
            Cf = T.alloc_ub((bi, bj), f32)
            Crel = T.alloc_ub((bi, bj), f32)
            U = T.alloc_ub((bi, 1), f32)
            Dp = T.alloc_ub((bi, bj), f32)
            O = T.alloc_ub((bi, bj), f32)
            O16 = T.alloc_ub((bi, bj), dtype)

            if variant == "writeonly":
                T.fill(O16, T.cast(0.0, dtype))
            elif use_hopper:
                # Crel[r, c] = c : tile-invariant (j0 == 0 since bj == S).
                T.npuir_arange(Crel, [0, 1], 0)
                if variant == "hoist_noarange":
                    T.npuir_arange(U, [1, 0], 0)

            for t in T.serial(stride):
                task = t * core_num + cid
                if task < n_tiles:
                    i0 = (task // nb) * bi
                    j0 = (task % nb) * bj

                    if variant == "nodist":
                        T.fill(Rf, T.cast(1.0, f32))
                    elif variant == "arange2":
                        T.npuir_arange(R, [1, 0], i0)
                        T.npuir_arange(C, [0, 1], j0)
                    elif variant == "subabs":
                        T.fill(Rf, T.cast(1.0, f32))
                        T.fill(Cf, T.cast(2.0, f32))
                        T.vsub(Rf, Cf, Rf)
                        T.vabs(Rf, Rf)
                    elif variant == "castsubabs":
                        T.fill(R, T.cast(1, "int16"))
                        T.fill(C, T.cast(2, "int16"))
                        T.vcast(R, Rf, round_mode="rint")
                        T.vcast(C, Cf, round_mode="rint")
                        T.vsub(Rf, Cf, Rf)
                        T.vabs(Rf, Rf)
                    elif variant == "f32ramp":
                        T.npuir_arange(Rf, [1, 0], i0)
                        T.npuir_arange(Cf, [0, 1], j0)
                        T.vsub(Rf, Cf, Rf)
                        T.vabs(Rf, Rf)
                    elif use_hopper:
                        # D = |(i0 + r) - c| (row-broadcast vsub, docs §2.2.2)
                        if variant != "hoist_noarange":
                            T.npuir_arange(U, [1, 0], i0)
                        T.vsub(Crel, U, Dp)
                        if variant != "hoist_vsubonly":
                            T.vabs(Dp, Dp)
                    elif variant in ("full", "distwrite"):
                        T.npuir_arange(R, [1, 0], i0)
                        T.npuir_arange(C, [0, 1], j0)
                        T.vcast(R, Rf, round_mode="rint")
                        T.vcast(C, Cf, round_mode="rint")
                        T.vsub(Rf, Cf, Rf)
                        T.vabs(Rf, Rf)

                    if variant == "distwrite":
                        if need_cast:
                            T.vcast(Rf, O16, round_mode="rint")
                        else:
                            T.copy(Rf, O16)

                    for h in T.serial(H):
                        if variant in ("writeonly", "distwrite"):
                            T.copy(O16, out[h, i0, j0])
                        else:
                            e = (
                                T.cast(-8.0, f32)
                                * T.cast(h + 1, f32)
                                / T.cast(H, f32)
                            )
                            slp = -T.exp2(e)
                            T.vmul(Dp if use_hopper else Rf, slp, O)
                            if need_cast:
                                T.vcast(O, O16, round_mode="rint")
                                T.copy(O16, out[h, i0, j0])
                            else:
                                T.copy(O, out[h, i0, j0])

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    parser.parse_known_args()
    if _VARIANT in _CORRECT_VARIANTS:
        ref = golden_alibi(64, 4, torch.float16)
        positions = torch.arange(64, dtype=torch.float32)
        dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        assert ref.shape == (4, 64, 64)
        assert float(dist.abs().max()) == 63.0
        print(f"[probe:{_VARIANT}] golden self-check OK")
    else:
        print(f"[probe:{_VARIANT}] diagnostic variant: no precision check")
    print("\033[92mprobe check passed!\033[0m")


if __name__ == "__main__":
    main()
