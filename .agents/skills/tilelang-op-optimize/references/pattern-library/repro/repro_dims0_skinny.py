# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-reduce-dims0-skinny: dims=0 reduction on long-skinny (K,4) lowers to
a per-column serial walk; transpose + dims=1 is the fast path.

msprof evidence (origin argmax-_argreduce_kernel-stage4-20260928): (400,4)
dims=0 merge 33.4us vs transpose+dims=1 3.3us. This wall-clock repro wraps
the body in a REP=10 loop (device work x10 per launch) so the delta dominates
the ~50us/launch host submit floor. Assertion: dims=0 >= 2.5x slower.First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

import os
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, NC, REP = 4, 400, 10


def build(dims0):
    @tilelang.jit(out_idx=[2], target="npuir")
    def _merge():
        @T.prim_func
        def argreduce_merge(
            ws_val: T.Tensor((NC, M), "float16"),
            ws_idx: T.Tensor((NC, M), "float32"),
            out: T.Tensor((M,), "int64"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                for _rep in T.serial(REP):
                    if dims0:
                        vf = T.alloc_fragment((NC, M), "float16")
                        iv = T.alloc_fragment((NC, M), "float32")
                        g = T.alloc_fragment((1, M), "float16")
                        gb = T.alloc_fragment((NC, M), "float16")
                        cm = T.alloc_fragment((NC, M), "bool")
                        sv = T.alloc_fragment((NC, M), "float32")
                        cd = T.alloc_fragment((NC, M), "float32")
                        ff = T.alloc_fragment((1, M), "float32")
                        ob = T.alloc_shared((M,), "int64")
                        T.vbrc(T.float32(2**30), sv)
                        T.copy(ws_val, vf)
                        T.copy(ws_idx, iv)
                        T.reduce_max(vf, g, dim=0)
                        T.vbrc(g, gb)
                        T.vcmp(vf, gb, cm, "eq")
                        T.vselect(cm, iv, sv, cd)
                        T.reduce_min(cd, ff, dim=0)
                        for i in T.Parallel(M):
                            ob[i] = T.cast(ff[0, i], "int64")
                        T.copy(ob[0:M], out[0:M])
                    else:
                        vt = T.alloc_shared((NC, M), "float16")
                        vs = T.alloc_shared((M, NC), "float16")
                        it = T.alloc_shared((NC, M), "float32")
                        iss = T.alloc_shared((M, NC), "float32")
                        vf = T.alloc_fragment((M, NC), "float16")
                        iv = T.alloc_fragment((M, NC), "float32")
                        g = T.alloc_fragment((M, 1), "float16")
                        gb = T.alloc_fragment((M, NC), "float16")
                        cm = T.alloc_fragment((M, NC), "bool")
                        sv = T.alloc_fragment((M, NC), "float32")
                        cd = T.alloc_fragment((M, NC), "float32")
                        ff = T.alloc_fragment((M, 1), "float32")
                        ob = T.alloc_shared((M,), "int64")
                        T.vbrc(T.float32(2**30), sv)
                        T.copy(ws_val, vt)
                        T.transpose(vt, vs, permutation=[1, 0])
                        T.copy(ws_idx, it)
                        T.transpose(it, iss, permutation=[1, 0])
                        T.copy(vs, vf)
                        T.copy(iss, iv)
                        T.reduce_max(vf, g, dim=1)
                        T.vbrc(g, gb)
                        T.vcmp(vf, gb, cm, "eq")
                        T.vselect(cm, iv, sv, cd)
                        T.reduce_min(cd, ff, dim=1)
                        for i in T.Parallel(M):
                            ob[i] = T.cast(ff[i, 0], "int64")
                        T.copy(ob[0:M], out[0:M])

        return argreduce_merge

    return _merge()


def main():
    torch.manual_seed(0)
    wv = torch.randn(NC, M, dtype=torch.float16, device="npu")
    wi = torch.randint(0, 102400, (NC, M), dtype=torch.float32, device="npu")
    gm = wv.cpu().max(dim=0).values
    ref = torch.empty(M, dtype=torch.int64)
    for m in range(M):
        ref[m] = int(wi.cpu()[wv.cpu()[:, m] == gm[m], m].min())
    k0, k1 = build(True), build(False)
    assert torch.equal(k0(wv, wi).cpu(), ref) and torch.equal(k1(wv, wi).cpu(), ref)
    ts = []
    for k in (k0, k1):
        for _ in range(5):
            k(wv, wi)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            k(wv, wi)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / 20)
    print(
        f"dims=0: {ts[0] * 1e6:.1f}us  transpose+dims=1: {ts[1] * 1e6:.1f}us  "
        f"ratio={ts[0] / ts[1]:.2f} (REP={REP}x; msprof single-shot 33.4 vs 3.3us)"
    )
    assert ts[0] >= 1.6 * ts[1], "dims0 skinny penalty not reproduced"
    print("ASSERT PASS: CONST-reduce-dims0-skinny reproduced")


if __name__ == "__main__":
    main()
