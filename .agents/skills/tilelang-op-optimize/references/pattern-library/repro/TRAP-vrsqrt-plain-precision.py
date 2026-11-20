"""TRAP-vrsqrt-plain-precision: T.vrsqrt is an approximate instruction.

Trap ID: TRAP-vrsqrt-plain-precision (candidate entry, see origin_task).
Phenomenon: on npuir (Developer mode, fp32 buffers), tilelang's
``T.vrsqrt`` lowers to a plain approximate Vector instruction whose
result carries a max relative error of ~2.7e-3 across inputs in
[9e-7, 8.7e5] (measured 2026-09-10; returned values quantized to ~10-11
significant bits, e.g. 1010/1024). Any golden computed with fp32 opmath
and a tight tolerance fails: in the ada_layer_norm stage-3 migration
the propagated error violated the fp32 1e-5 gate on ~87% and the fp16
1e-3 gate on ~6% of elements (bf16 1.6e-2 masked it). The
examples/norm/layer_norm.py precedent passes only because its tolerance
is 1e-2.

Bypass (used by the delivered _ada_layer_norm_kernel): compose the same
real function from correctly rounded ops -- rstd = sqrt(v)/v via
``T.vsqrt`` + ``T.vdiv`` (measured max rel err 5.8e-8 / 7.9e-8). A
2-iteration correction-form Newton refinement on the raw vrsqrt also
reaches 6.3e-8 but costs 12 tiny extra ops.

Assertions (magnitude-based, robust to environment drift):
  1. phenomenon present: raw T.vrsqrt max rel err > 1e-3
     (a future toolchain fixing vrsqrt to full precision flips this
     assert -> trap overturned, refresh the entry's version stamp);
  2. bypass correct: T.vsqrt + T.vdiv max rel err < 1e-6.

First verified: tilelang 0.1.2+a83118285a, Ascend910B2C, CANN 8.5.0,
torch 2.9.0+cpu, 2026-09-10.
origin_task: ada_layer_norm/_ada_layer_norm_kernel stage-3 first_impl
(DESIGN.md section 3.2 mapped rstd to T.vrsqrt; stage-3 stage-isolation
probe pinned the error to the rstd stage: mean/d exact to 1e-7 while
rstd carried 1.57e-3).
Re-verification record: (append-only)
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

M = 64  # rows: one probe value per row
dtype = "float32"


@tilelang.jit(out_idx=[1, 2], target="npuir")
def _rsqrt_probe():
    @T.prim_func
    def main(
        v: T.Tensor[(M, 1), dtype],
        raw: T.Tensor[(M, 1), dtype],
        bypass: T.Tensor[(M, 1), dtype],
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            v_ub = T.alloc_shared((M, 1), "float32")
            t = T.alloc_shared((M, 1), "float32")
            T.copy(v[0:M, 0:1], v_ub[0:M, 0:1])
            # Phenomenon: raw T.vrsqrt (approximate instruction).
            T.vrsqrt(v_ub, t)
            T.copy(t[0:M, 0:1], raw[0:M, 0:1])
            # Bypass: sqrt(v)/v == 1/sqrt(v), both ops correctly rounded.
            T.vsqrt(v_ub, t)
            T.vdiv(t, v_ub, t)
            T.copy(t[0:M, 0:1], bypass[0:M, 0:1])

    return main


def main():
    # Geometric sweep of positive inputs (LN var range and beyond).
    v = torch.tensor(
        [[10 ** (-6 + 12 * i / (M - 1))] for i in range(M)],
        dtype=torch.float32,
    )
    raw, bypass = _rsqrt_probe()(v.npu())
    exact = 1.0 / v.double().sqrt()
    raw_err = ((raw.cpu().double() - exact) / exact).abs().max().item()
    byp_err = ((bypass.cpu().double() - exact) / exact).abs().max().item()
    print(f"raw T.vrsqrt     max rel err = {raw_err:.3e}")
    print(f"vsqrt+vdiv bypass max rel err = {byp_err:.3e}")

    # 1. phenomenon present (magnitude assertion).
    assert raw_err > 1e-3, (
        f"raw T.vrsqrt max rel err {raw_err:.3e} <= 1e-3: the approximate "
        "behavior is gone -- trap may be overturned on this toolchain, "
        "refresh the entry version stamp"
    )
    # 2. bypass correct (bit-safe composition).
    assert byp_err < 1e-6, (
        f"vsqrt+vdiv bypass max rel err {byp_err:.3e} >= 1e-6: bypass "
        "degraded, re-examine"
    )
    print("TRAP-vrsqrt-plain-precision: PASS (phenomenon present, bypass OK)")


if __name__ == "__main__":
    main()
