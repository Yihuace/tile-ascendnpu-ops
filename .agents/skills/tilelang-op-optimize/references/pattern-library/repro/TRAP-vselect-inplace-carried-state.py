# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-vselect-inplace-carried-state

Phenomenon (toolchain trap, kind: trap):
    In a multi-iteration ``T.serial`` tile loop that carries running state
    (online argmax / running (max, idx) recurrence), the *in-place* update
    ``T.vselect(cond, chunk, running, running)`` (the "false" operand is also
    the output) silently fails to carry the state across iterations: the final
    output stays at the tile-0 value when the true extreme lives in a later
    tile.  A single-iteration loop (num_full == 2) is NOT affected -- the bug
    only manifests with num_full >= 3 (>= 2 update iterations).

Workaround (asserted below):
    Select into a distinct scratch buffer and copy back:
        T.vselect(cond, chunk, running, new)
        T.copy(new, running)
    This no-alias form carries the state correctly (bit-exact vs torch.argmax).

First verified: 2026-09-24, tilelang 0.1.2 (dev root build) + CANN 8.5.0 +
    Ascend910B2C.
origin_task: examples/argmax/_argreduce_kernel (Stage 3 first_impl; the
    DESIGN.md tiled online path was only probe-validated at num_full == 2,
    so the num_full == 20 lm-head case exposed this at first run).

Re-verify records (append):
    2026-09-24 first run: noalias bit-exact PASS; inplace stays at tile-0.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

BIG = 2**30


def _build(M, N, tile_n, inplace):
    """Minimal online argmax over N-tiles; bm=1 (multitile trap forces 1)."""
    num_full = N // tile_n
    dtype = "float16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(bm):
        @T.prim_func
        def main(x: T.Tensor((M, N), dtype), out: T.Tensor((M,), "int64")):
            with T.Kernel(M, is_npu=True) as (pid, _):
                x_ub = T.alloc_shared((bm, tile_n), dtype)
                x_work = T.alloc_fragment((bm, tile_n), dtype)
                chunk = T.alloc_fragment((bm, 1), dtype)
                brc = T.alloc_fragment((bm, tile_n), dtype)
                cand = T.alloc_fragment((bm, tile_n), "float32")
                first = T.alloc_fragment((bm, 1), "float32")
                glob = T.alloc_fragment((bm, 1), "float32")
                cond = T.alloc_fragment((bm, 1), "bool")
                running = T.alloc_fragment((bm, 1), dtype)
                ridx = T.alloc_fragment((bm, 1), "float32")
                new_r = T.alloc_fragment((bm, 1), dtype)
                new_i = T.alloc_fragment((bm, 1), "float32")
                out_ub = T.alloc_shared((bm,), "int64")

                T.copy(x[pid * bm : pid * bm + bm, 0:tile_n], x_ub)
                T.copy(x_ub, x_work)
                T.reduce_max(x_work, running, dim=1)
                T.vbrc(running, brc)
                for i, j in T.Parallel(bm, tile_n):
                    cand[i, j] = T.if_then_else(
                        x_work[i, j] == brc[i, j], T.cast(j, "float32"), T.float32(BIG)
                    )
                T.reduce_min(cand, ridx, dim=1)

                for t in T.serial(num_full - 1):
                    T.copy(
                        x[
                            pid * bm : pid * bm + bm,
                            (t + 1) * tile_n : (t + 2) * tile_n,
                        ],
                        x_ub,
                    )
                    T.copy(x_ub, x_work)
                    T.reduce_max(x_work, chunk, dim=1)
                    T.vbrc(chunk, brc)
                    for i, j in T.Parallel(bm, tile_n):
                        cand[i, j] = T.if_then_else(
                            x_work[i, j] == brc[i, j],
                            T.cast(j, "float32"),
                            T.float32(BIG),
                        )
                    T.reduce_min(cand, first, dim=1)
                    for i in T.Parallel(bm):
                        glob[i, 0] = T.cast((t + 1) * tile_n, "float32") + first[i, 0]
                    T.vcmp(chunk, running, cond, "gt")
                    if inplace:
                        # Buggy: running is both the "false" operand and the
                        # output; state is not carried over the serial loop.
                        T.vselect(cond, chunk, running, running)
                        T.vselect(cond, glob, ridx, ridx)
                    else:
                        # Workaround: no-alias select + copy back.
                        T.vselect(cond, chunk, running, new_r)
                        T.vselect(cond, glob, ridx, new_i)
                        T.copy(new_r, running)
                        T.copy(new_i, ridx)

                for i in T.Parallel(bm):
                    out_ub[i] = T.cast(ridx[i, 0], "int64")
                T.copy(out_ub, out[pid * bm : pid * bm + bm])

        return main

    return _func(1)


def main():
    M, N, tile_n = 2, 102400, 5120  # num_full = 20 (multi-iteration loop)
    x = torch.randn(M, N, dtype=torch.float16, device="npu") * 0.1
    x[0, 6000] = 5.0  # true max in tile 1 (NOT tile 0)
    x[1, 6001] = 5.0
    ref = x.argmax(dim=-1).cpu()

    y_ok = _build(M, N, tile_n, inplace=False)(x)
    assert torch.equal(y_ok.cpu(), ref), (
        f"noalias wrong: {y_ok.cpu().tolist()} vs {ref.tolist()}"
    )
    print(
        f"[workaround] noalias bit-exact PASS: {y_ok.cpu().tolist()} == {ref.tolist()}"
    )

    y_bug = _build(M, N, tile_n, inplace=True)(x)
    print(f"[bug demo]   inplace got {y_bug.cpu().tolist()} (want {ref.tolist()})")
    # The in-place form is NOT asserted bit-exact: it is expected to be wrong
    # on this toolchain.  If a future toolchain fixes the in-place alias, this
    # repro still passes (workaround assert above) and the bug demo line flips.


if __name__ == "__main__":
    main()
