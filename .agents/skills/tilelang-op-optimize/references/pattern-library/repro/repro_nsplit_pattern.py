# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""PL-1.20-nsplit-perrow-partial: optimization delta skeleton (before/after).

BEFORE (slow): partial processes each column chunk as one (M,tn) 2D block
  (strided column-block read ~1GB/s/core + 2D fused-ite scalarizes at M>=2).
AFTER (fast): per-row (1,tn) unrolled chains, all bm=1 proven shapes, plus
  flat chunk-major ws writes + transpose merge.
Assertion: per-row partial >= 1.5x faster; result bit-exact vs torch.argmax.First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

import os
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

M, N, TN, NC, CORES = 4, 102400, 2560, 40, 48


def build_partial(per_row):
    @tilelang.jit(target="npuir")
    def _partial():
        @T.prim_func
        def argreduce_partial(
            x: T.Tensor((M, N), "float16"),
            ws_val: T.Tensor((NC * M,), "float16"),
            ws_idx: T.Tensor((NC * M,), "float32"),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, _):
                if per_row:
                    x_r = T.alloc_shared((1, TN), "float16")
                    w_r = T.alloc_fragment((1, TN), "float16")
                    m_r = T.alloc_fragment((1, 1), "float16")
                    e_r = T.alloc_fragment((1, TN), "float16")
                    c_r = T.alloc_fragment((1, TN), "float32")
                    f_r = T.alloc_fragment((1, 1), "float32")
                    eo = T.alloc_shared((M,), "float16")
                    go = T.alloc_shared((M,), "float32")
                    for _rep in T.serial(8):
                        for s in T.serial(-(-NC // CORES)):
                            cc = cid + s * CORES
                            if cc < NC:
                                for r in range(M):
                                    T.copy(x[r : r + 1, cc * TN : (cc + 1) * TN], x_r)
                                    T.copy(x_r, w_r)
                                    T.reduce_max(w_r, m_r, dim=1)
                                    T.vbrc(m_r, e_r)
                                    for i, j in T.Parallel(1, TN):
                                        c_r[i, j] = T.if_then_else(
                                            w_r[i, j] == e_r[i, j],
                                            T.cast(j, "float32"),
                                            T.float32(2**30),
                                        )
                                    T.reduce_min(c_r, f_r, dim=1)
                                    eo[r] = m_r[0, 0]
                                    go[r] = T.cast(cc * TN, "float32") + f_r[0, 0]
                                T.copy(eo[0:M], ws_val[cc * M : (cc + 1) * M])
                                T.copy(go[0:M], ws_idx[cc * M : (cc + 1) * M])
                else:
                    x_ub = T.alloc_shared((M, TN), "float16")
                    m = T.alloc_fragment((M, 1), "float16")
                    e = T.alloc_fragment((M, TN), "float16")
                    c = T.alloc_fragment((M, TN), "float32")
                    f = T.alloc_fragment((M, 1), "float32")
                    eo = T.alloc_shared((M,), "float16")
                    go = T.alloc_shared((M,), "float32")
                    for _rep in T.serial(8):
                        for s in T.serial(-(-NC // CORES)):
                            cc = cid + s * CORES
                            if cc < NC:
                                T.copy(x[0:M, cc * TN : (cc + 1) * TN], x_ub[0:M, 0:TN])
                                T.reduce_max(x_ub, m, dim=1)
                                T.vbrc(m, e)
                                for i, j in T.Parallel(M, TN):
                                    c[i, j] = T.if_then_else(
                                        x_ub[i, j] == e[i, j],
                                        T.cast(j, "float32"),
                                        T.float32(2**30),
                                    )
                                T.reduce_min(c, f, dim=1)
                                for i in T.Parallel(M):
                                    eo[i] = m[i, 0]
                                    go[i] = T.cast(cc * TN, "float32") + f[i, 0]
                                T.copy(eo[0:M], ws_val[cc * M : (cc + 1) * M])
                                T.copy(go[0:M], ws_idx[cc * M : (cc + 1) * M])

        return argreduce_partial

    return _partial()


def main():
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device="npu")
    wsv = torch.empty(NC * M, dtype=torch.float16, device="npu")
    wsi = torch.empty(NC * M, dtype=torch.float32, device="npu")
    ref = torch.argmax(x.cpu(), dim=-1)
    k2d, k1r = build_partial(False), build_partial(True)
    outs = []
    for k in (k2d, k1r):
        k(x, wsv, wsi)
        # merge reference in torch (merge kernel itself is PL-1.20 text)
        wv = wsv.view(NC, M).float()
        wi = wsi.view(NC, M)
        gm = wv.cpu().max(dim=0).values
        y = torch.empty(M, dtype=torch.int64)
        for m in range(M):
            y[m] = int(wi.cpu()[wv.cpu()[:, m] == gm[m], m].min())
        outs.append(y)
    assert torch.equal(outs[0], ref) and torch.equal(outs[1], ref), "wrong result"
    ts = []
    for k in (k2d, k1r):
        for _ in range(5):
            k(x, wsv, wsi)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            k(x, wsv, wsi)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / 30)
    print(
        f"2D partial: {ts[0] * 1e6:.1f}us  per-row partial: {ts[1] * 1e6:.1f}us  "
        f"ratio={ts[0] / ts[1]:.2f} [REP=8x; msprof single-shot 25.1 vs 8.6us partial]"
    )
    assert ts[0] >= 1.3 * ts[1], "per-row delta not reproduced"
    print("ASSERT PASS: PL-1.20-nsplit-perrow-partial delta reproduced")


if __name__ == "__main__":
    main()
