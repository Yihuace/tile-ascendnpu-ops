"""Round-2 probe: replace the row-broadcast vsub with vbrc + same-shape ops.

Device evidence (msprof op, 2k fp16, Stage 4) motivating this probe:

  * the row-broadcast form ``T.vsub(Crel, Rrow)`` with a ``(bi, 1)`` operand
    (docs/Tilelang.language/数学操作/T.vsub.md section 2.2.2 "行广播") costs
    ~1.4 us per 8192-element call, whereas a same-shape ``vsub`` + ``vabs``
    pair costs ~0.15 us (probe `subabs`) -- i.e. the broadcast lowering is
    ~10x more expensive than the same-shape lowering;
  * ``T.npuir_arange`` with an **fp32** dst is scalarised (~2.3 ns/element:
    probe `f32ramp` = 579 us / 483 us scalar for 2 full aranges per tile),
    while an **int16** arange over the same tile is cheap (~0.08 ns/element,
    probe `arange2` = 116 us), and an ``int16 -> fp32`` ``T.vcast`` is again
    scalarised (~15.5 us per full tile, inferred from the baseline's 455 us
    scalar total).  So the only fp32 ramp we may afford is the single
    tile-invariant one built once per core.

Variants (env ``ALIBI_PROBE2_VARIANT``):

  hoist     round-1 candidate: hoisted ``Crel[r,c]=c`` + per-tile tiny
            ``(bi,1)`` fp32 row ramp + row-broadcast ``vsub`` + ``vabs``
  hoist_b   same, but the per-tile row ramp is expanded to the full tile with
            ``T.vbrc`` ((M,1) -> (M,N), docs/Tilelang.language/shape操作/T.vbrc.md
            section 2.2.2) and the distance is computed with a same-shape
            ``vsub`` + ``vabs``
  hoist_d   one fp32 ``arange(strides=[1,-1])`` builds the tile-invariant
            ``E[r,c] = r - c`` once per core; per tile a scalar-broadcast
            ``T.vbrc(i0-j0, Qfull)`` fills the scalar plane and
            ``D = vabs(vadd(E, Qfull))`` (two same-shape ops)

Correct variants: all three.

run: ALIBI_PROBE2_VARIANT=hoist_b python _probe2.py --level L0
"""

import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
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
_VARIANT = os.environ.get("ALIBI_PROBE2_VARIANT", "hoist")


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
    variant = os.environ.get("ALIBI_PROBE2_VARIANT", _VARIANT)
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

    if nb != 1:
        variant = "hoist"

    @T.prim_func
    def main(out: T.Tensor((H, S, S), dtype)):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            Crel = T.alloc_ub((bi, bj), f32)
            Rrow = T.alloc_ub((bi, 1), f32)
            Rfull = T.alloc_ub((bi, bj), f32)
            Dm = T.alloc_ub((bi, bj), f32)
            O = T.alloc_ub((bi, bj), f32)
            O16 = T.alloc_ub((bi, bj), dtype)

            if variant == "hoist_b":
                T.npuir_arange(Crel, [0, 1], 0)
            elif variant == "hoist_d":
                T.npuir_arange(Rfull, [1, -1], 0)  # E[r, c] = r - c (fp32, once)
            else:
                T.npuir_arange(Crel, [0, 1], 0)

            for t in T.serial(stride):
                task = t * core_num + cid
                if task < n_tiles:
                    i0 = (task // nb) * bi
                    j0 = (task % nb) * bj

                    if variant == "hoist_b":
                        T.npuir_arange(Rrow, [1, 0], i0 - j0)
                        T.vbrc(Rrow, Rfull)
                        T.vsub(Crel, Rfull, Dm)
                        T.vabs(Dm, Dm)
                    elif variant == "hoist_d":
                        T.vbrc(T.cast(i0 - j0, f32), Dm)
                        T.vadd(Rfull, Dm, Dm)
                        T.vabs(Dm, Dm)
                    else:
                        T.npuir_arange(Rrow, [1, 0], i0 - j0)
                        T.vsub(Crel, Rrow, Dm)
                        T.vabs(Dm, Dm)

                    for h in T.serial(H):
                        e = (
                            T.cast(-8.0, f32)
                            * T.cast(h + 1, f32)
                            / T.cast(H, f32)
                        )
                        slp = -T.exp2(e)
                        T.vmul(Dm, slp, O)
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
    ref = golden_alibi(64, 4, torch.float16)
    assert ref.shape == (4, 64, 64)
    print(f"[probe2:{_VARIANT}] golden self-check OK")
    print("\033[92mprobe2 check passed!\033[0m")


if __name__ == "__main__":
    main()
