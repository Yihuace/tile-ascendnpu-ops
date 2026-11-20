# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""REPRO (documented negative): the mul+reduce staging tile is NOT capped by a
pattern-level element limit -- a (32,512)=16384-element product tile compiles
fine in isolation.

Entry: pattern-library/constants.md#CONST-ub-budget-vs-pattern-cap
First verified: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3,
task TileOPs-_engram_decode_kernel-20261030T145026Z (round 3, 2026-10-30);
re-verified: (none yet)

Mechanism (why this matters)
----------------------------
Why this probe exists
----------------------
Stage 4 round 3 of `engram_decode` first attributed "BK=32 at d_padded=512 is
rejected with `ub overflow, requires 1606656 bits while 1572864 bits
available`" to a pattern-level cap of 8192 fp32 elements per mul+reduce tile.
**This probe refutes that attribution**: the identical mul+reduce pattern with
a (32,512) = 16384-element product tile compiles.  The real cause is the UB
*budget* shared with the rest of the kernel: the same `engram_decode` kernel at
BK=32/d_padded=512 does compile once its post-GEMV tail is removed, i.e. the
GEMV staging tiles and the tail working set coexist in the same 192 KiB and the
compiler does not hand the GEMV staging bytes back to the tail.

Two further measured properties (see pattern-library/cases.md
CASE-engram-decode-stage4) are *not* reproduced here because they need the full
kernel: (i) the reported "requires" byte count was byte-identical across f32 and
f16 weight staging and across a tail-aliasing variant, so tail reclamation is
ineffective; (ii) the usable BK at d_padded=512 therefore stays 16 while at
d_padded=256 it reaches 32.

Lesson: never infer a pattern-level capacity constant from one kernel's overflow
message -- check whether the minimal pattern reproduces it.

Grade (repro/README.md ED-B): self-contained + assertion (documented negative).
Toolchain stamp: tilelang 0.1.2+01805d7a / CANN 9.0.0 / Ascend910B3.
Runtime: 3 JIT compiles, ~1 minute.

Run: python stage-tile-not-pattern-cap.py
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

D_PAD = 512
F32 = "float32"


def build(bk, d_padded):
    """row-broadcast multiply into a (bk, d_padded) tile + vreduce accumulate."""

    @tilelang.jit(out_idx=[2], target="npuir")
    def factory():
        @T.prim_func
        def main(
            W: T.Tensor((bk, d_padded), F32),
            e: T.Tensor((bk,), F32),
            acc: T.Tensor((1, d_padded), F32),
        ):
            with T.Kernel(1, is_npu=True) as (bid, _):
                w_f = T.alloc_shared((bk, d_padded), F32)
                e_n = T.alloc_shared((bk,), F32)
                prod = T.alloc_shared((bk, d_padded), F32)
                out = T.alloc_shared((1, d_padded), F32)
                T.copy(W, w_f)
                T.copy(e, e_n)
                for kk, j in T.Parallel(bk, d_padded):
                    prod[kk, j] = w_f[kk, j] * e_n[kk]
                T.reduce_sum(prod, out, dim=0, clear=True)
                T.copy(out, acc)

        return main

    return factory()


def main():
    # 8192 elements: accepted (BK=16 at d_padded=512).
    k_ok = build(8192 // D_PAD, D_PAD)
    print(f"[ok] BK={8192 // D_PAD} x {D_PAD} = 8192 elements compiles")

    assert k_ok is not None
    # 16384 elements: also accepted -> there is no pattern-level 8192 cap.
    k_idem = build(2 * (8192 // D_PAD), D_PAD)
    assert k_idem is not None
    print("[ok] BK=32 x 512 = 16384 elements ALSO compiles "
          "-> the 8192-element limit seen in engram_decode is a UB-budget "
          "interaction with the rest of the kernel, not a pattern cap")


if __name__ == "__main__":
    main()
