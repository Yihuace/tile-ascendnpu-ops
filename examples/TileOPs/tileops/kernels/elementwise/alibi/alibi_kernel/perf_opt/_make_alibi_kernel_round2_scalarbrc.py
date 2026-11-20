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
_VARIANT = os.environ.get("ALIBI_PROBE2_VARIANT", "hoist_d")


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


# ---------- precision comparing func ----------
_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances from examples/TileOPs/tests/ops/test_special_elementwise.py::test_alibi
# (fp16 1e-2, fp32 1e-5) plus the repo bf16 elementwise tolerance (DESIGN.md 8.2).
_TOLERANCE = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-2, 1e-2),
    "bfloat16": (1.6e-2, 1.6e-2),
}


def _max_diff(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Max abs error evaluated in fp64 (tensors are small enough here)."""
    return (out.to(torch.float64) - ref.to(torch.float64)).abs().max().item()


def _run_case(S, H, dtype_str, tag, bi=None, bj=None, check_diag=False):
    """Build the kernel for (S, H, dtype), run it on the NPU, compare to golden."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    ref = golden_alibi(S, H, torch_dtype)
    kernel = _make_alibi_kernel(S, H, dtype_str, bi=bi, bj=bj)

    out = torch.empty(H, S, S, dtype=torch_dtype, device="npu")
    kernel(out)
    out_cpu = out.cpu()

    atol, rtol = _TOLERANCE[dtype_str]
    max_diff = _max_diff(out_cpu, ref)
    diag_ok = True
    if check_diag:
        diag = out_cpu.diagonal(dim1=1, dim2=2)
        diag_ok = bool((diag == 0).all().item())
    try:
        torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)
        if check_diag and not diag_ok:
            raise AssertionError("diagonal entries are not exactly zero")
    except AssertionError as e:  # noqa: BLE001
        raise AssertionError(
            f"[{tag}] FAIL: S={S} H={H} dtype={dtype_str} "
            f"bi={bi} bj={bj} max_diff={max_diff:.6e} diag_ok={diag_ok}\n{e}"
        ) from e
    print(
        f"[{tag}] PASS: S={S} H={H} dtype={dtype_str} bi={bi} bj={bj} "
        f"max_diff={max_diff:.6e} atol={atol:.1e} rtol={rtol:.1e} diag_ok={diag_ok}"
    )
    return max_diff


# ---------- hierarchical testing ----------
def run_L0():
    # L0-1/2: source test case (S=128, H=8) -- H=8 slopes are exact powers of 2.
    _run_case(128, 8, "float16", "L0-test-fp16", check_diag=True)
    _run_case(128, 8, "float32", "L0-test-fp32")
    # L0-3/4: manifest smoke (S=2048, H=32).
    _run_case(2048, 32, "float16", "L0-manifest-2k-fp16")
    _run_case(2048, 32, "bfloat16", "L0-manifest-2k-bf16")
    # L0-5: large S (full manifest 4k prefill shape).
    _run_case(4096, 32, "float16", "L0-manifest-4k-fp16")
    # L0-6: int16 ramp value range -- |i-j| > 2048 (beyond fp16's exact range).
    _run_case(4096, 1, "float16", "L0-int16-range", check_diag=True)
    # L0-8: small S / single tile.
    _run_case(64, 4, "float16", "L0-small")
    # L0-9: dtype coverage at S=256, H=32.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _run_case(256, 32, dtype_str, f"L0-256-{dtype_str}")
    # L0-10: non-2-power S (divisor-based tiling keeps every tile exact).
    _run_case(1000, 8, "float16", "L0-nondiv-1000")


def run_L1():
    # L1: full manifest dtype coverage + explicit bi/bj overrides + H extremes.
    _run_case(4096, 32, "bfloat16", "L1-manifest-4k-bf16")
    _run_case(4096, 32, "float32", "L1-manifest-4k-fp32")
    # Explicit narrow bj (multi-column-block path, as device-probed in Q2).
    _run_case(2048, 32, "float16", "L1-tile-64x128", bi=64, bj=128)
    _run_case(2048, 32, "bfloat16", "L1-tile-32x256", bi=32, bj=256)
    _run_case(512, 16, "float16", "L1-512x16", bi=16, bj=128)
    # H extremes and non-power-of-2 head counts.
    _run_case(256, 1, "float16", "L1-H1")
    _run_case(256, 5, "float32", "L1-H5")
    _run_case(1024, 16, "bfloat16", "L1-1024x16")
    _run_case(2048, 8, "float16", "L1-2k-H8")
    _run_case(4096, 4, "float32", "L1-4k-H4")


def run_L2():
    # L2 (non-blocking): odd / small sizes and alternate tiles.
    for S, H, dtype_str, bi, bj in (
        (128, 8, "float16", 8, 128),
        (200, 6, "float32", None, None),
        (255, 3, "float16", None, None),
        (1024, 32, "float16", 4, 256),
        (2048, 32, "float16", 8, 512),
    ):
        try:
            _run_case(S, H, dtype_str, "L2", bi=bi, bj=bj)
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): S={S} H={H} dtype={dtype_str} err={e}")


def run_boundary():
    # Boundary (non-blocking): diagonal exactness, extreme distance, bf16 tails.
    for dtype_str in ("float32", "float16", "bfloat16"):
        try:
            _run_case(128, 8, dtype_str, "Boundary-diag", check_diag=True)
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (Record without blocking): diag dtype={dtype_str} err={e}")

    for S, H, dtype_str in (
        (4096, 1, "bfloat16"),
        (4096, 1, "float32"),
        (4096, 32, "bfloat16"),
    ):
        try:
            _run_case(S, H, dtype_str, "Boundary-large")
        except Exception as e:  # noqa: BLE001
            print(
                f"[Boundary] WARN (Record without blocking): "
                f"S={S} H={H} dtype={dtype_str} err={e}"
            )


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
