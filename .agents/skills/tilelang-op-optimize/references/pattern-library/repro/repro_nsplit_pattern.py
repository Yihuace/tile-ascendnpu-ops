# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.20 Expert N-split: partial, barrier, and tie-break merge in one launch.

Entry: pattern-library/elementwise.md#PL-1.20-nsplit-perrow-partial
First verified: tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 /
Ascend910B2C, task argmax-_argreduce_kernel-20260924T031005Z;
re-verified: (none yet)
"""

import os
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, N, TN = 4, 102400, 5120
NCHUNK, NUM_BLOCKS = N // TN, N // (2 * TN)


def build_fused_nsplit():
    @tilelang.jit(out_idx=[3], target="npuir")
    def _jit():
        @T.prim_func
        def argreduce_nsplit_fused(
            x: T.Tensor((M, N), "float16"),
            ws_val: T.Tensor((NCHUNK, M), "float16"),
            ws_idx: T.Tensor((NCHUNK, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(NUM_BLOCKS, is_npu=True) as (cid, subid), T.Scope("Vector"):
                # Per-chunk partial. x_r is reused as the reduce/compare source.
                x_r = T.alloc_ub((1, TN), "float16")
                max_r = T.alloc_ub((1, 1), "float16")
                max_brc = T.alloc_ub((1, TN), "float16")
                idx_src = T.alloc_ub((TN,), "float32")
                idx_r = T.alloc_ub((1, TN), "float32")
                sentinel_r = T.alloc_ub((1, TN), "float32")
                equal_r = T.alloc_ub((1, TN), "bool")
                candidate_r = T.alloc_ub((1, TN), "float32")
                first_r = T.alloc_ub((1, 1), "float32")
                partial_val = T.alloc_ub((M,), "float16")
                partial_idx = T.alloc_ub((M,), "float32")
                # Final merge. These are small: NCHUNK * M is 20 * 4 here.
                val_ub = T.alloc_ub((NCHUNK, M), "float16")
                idx_ub = T.alloc_ub((NCHUNK, M), "float32")
                global_val = T.alloc_ub((1, M), "float16")
                global_brc = T.alloc_ub((NCHUNK, M), "float16")
                global_equal = T.alloc_ub((NCHUNK, M), "bool")
                global_sentinel = T.alloc_ub((NCHUNK, M), "float32")
                global_candidate = T.alloc_ub((NCHUNK, M), "float32")
                global_first = T.alloc_ub((1, M), "float32")
                first_flat = T.alloc_ub((M,), "float32")
                out_ub = T.alloc_ub((M,), "int64")
                T.arange(idx_src, [1], 0)
                T.reshape(idx_src, idx_r)
                T.vbrc(T.float32(2**30), sentinel_r)
                vid = cid * 2 + subid
                for r in range(M):
                    T.copy(x[r : r + 1, vid * TN : (vid + 1) * TN], x_r)
                    T.reduce_max(x_r, max_r, dim=1)
                    T.vbrc(max_r, max_brc)
                    T.vcmp(x_r, max_brc, equal_r, "eq")
                    T.vselect(equal_r, idx_r, sentinel_r, candidate_r)
                    T.reduce_min(candidate_r, first_r, dim=1)
                    partial_val[r] = max_r[0, 0]
                    partial_idx[r] = T.cast(vid * TN, "float32") + first_r[0, 0]
                with T.rs("PIPE_MTE3"):
                    T.copy(partial_val, ws_val[vid, 0:M])
                    T.copy(partial_idx, ws_idx[vid, 0:M])
                    T.block_barrier(0)
                if vid == 0:
                    T.vbrc(T.float32(2**30), global_sentinel)
                    T.copy(ws_val, val_ub)
                    T.copy(ws_idx, idx_ub)
                    T.reduce_max(val_ub, global_val, dim=0)
                    T.vbrc(global_val, global_brc)
                    T.vcmp(val_ub, global_brc, global_equal, "eq")
                    T.vselect(global_equal, idx_ub, global_sentinel, global_candidate)
                    T.reduce_min(global_candidate, global_first, dim=0)
                    T.reshape(global_first, first_flat)
                    T.vcast(first_flat, out_ub, round_mode="rint")
                    T.copy(out_ub, out)

        return argreduce_nsplit_fused

    previous_mode = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = "expert"
    try:
        return _jit()
    finally:
        if previous_mode is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = previous_mode


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    x[0, TN + 7] = 100.0
    x[0, 3 * TN + 9] = 100.0
    kernel = build_fused_nsplit()
    ws_val = torch.empty((NCHUNK, M), dtype=torch.float16, device="npu")
    ws_idx = torch.empty((NCHUNK, M), dtype=torch.float32, device="npu")
    out = kernel(x, ws_val, ws_idx)
    reference = torch.argmax(x.cpu(), dim=-1)
    assert torch.equal(out.cpu(), reference), (out.cpu(), reference)
    assert int(out.cpu()[0]) == TN + 7
    print("ASSERT PASS: PL-1.20 Expert single-launch N-split reproduced")


if __name__ == "__main__":
    main()
