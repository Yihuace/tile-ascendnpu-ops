"""D-3 design probe: validate the split-kernel NPU structure on the real device.

Tests the core constructs of the `_batch_norm_bwd_split_kernel` NPU redesign:
  1. `c = t // SPLITS` / `s = t % SPLITS` (floordiv/mod) as the middle-axis index
     of a `T.copy` GM->UB strided plane gather under a 1D persistent kernel;
  2. inner S-tile accumulation with `T.reduce(..., clear=False, size=[N, real])`
     and tail truncation;
  3. a partial workspace `partial_sum[C, SPLITS]` + a second (finalize) phase
     that merges the splits (no atomics);
  4. a third (apply) phase that maps grad_x per channel.

Run: python design_probe_split_phase.py
"""
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

N, C, S = 4, 32, 250
SPLITS = 4
DT = "float32"
L = N * S
CHUNK_S = (S + SPLITS - 1) // SPLITS
BS = 32
VEC = 40


def build():
    @tilelang.jit(target="npuir")
    def _stats():
        ntasks = C * SPLITS
        KS = min(ntasks, VEC)
        nlS = (ntasks + KS - 1) // KS

        @T.prim_func
        def stats(
            grad_out: T.Tensor((N, C, S), DT),
            x: T.Tensor((N, C, S), DT),
            mean: T.Tensor((C,), DT),
            rstd: T.Tensor((C,), DT),
            partial_sum: T.Tensor((C, SPLITS), DT),
            partial_sum_xhat: T.Tensor((C, SPLITS), DT),
        ):
            with T.Kernel(min(C * SPLITS, VEC), is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    gt = T.alloc_ub((N, BS), DT)
                    xt = T.alloc_ub((N, BS), DT)
                    prod = T.alloc_ub((N, BS), DT)
                    brc = T.alloc_ub((N, BS), DT)
                    red = T.alloc_ub((N, 1), DT)
                    red2 = T.alloc_ub((N, 1), DT)
                    m1 = T.alloc_ub((1, 1), DT)
                    r1 = T.alloc_ub((1, 1), DT)
                    gb = T.alloc_ub((1, 1), DT)
                    gx = T.alloc_ub((1, 1), DT)
                    for i in T.serial(nlS):
                        t = i * KS + cid
                        if t < ntasks:
                            c = t // SPLITS
                            sp = t % SPLITS
                            s0 = sp * CHUNK_S
                            real = T.min(CHUNK_S, S - s0)
                            ntile = (real + BS - 1) // BS
                            T.copy(mean[c:c + 1], m1)
                            T.copy(rstd[c:c + 1], r1)
                            T.clear(red)
                            T.clear(red2)
                            for tt in T.serial(ntile):
                                ss = s0 + tt * BS
                                r2 = T.min(BS, s0 + real - ss)
                                T.copy(grad_out[0:N, c, ss:ss + r2], gt[0:N, 0:r2])
                                T.copy(x[0:N, c, ss:ss + r2], xt[0:N, 0:r2])
                                T.vbrc(m1, brc)
                                T.vsub(xt, brc, xt)
                                T.vbrc(r1, brc)
                                T.vmul(xt, brc, xt)
                                T.vmul(gt, xt, prod)
                                T.reduce(gt, red, dims=[1], reduce_mode="sum",
                                         clear=False, size=[N, r2])
                                T.reduce(prod, red2, dims=[1], reduce_mode="sum",
                                         clear=False, size=[N, r2])
                            T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)
                            T.reduce(red2, gx, dims=[0], reduce_mode="sum", clear=True)
                            T.copy(gb, partial_sum[c:c + 1, sp:sp + 1])
                            T.copy(gx, partial_sum_xhat[c:c + 1, sp:sp + 1])
        return stats

    @tilelang.jit(target="npuir")
    def _finalize():
        KC = min(C, VEC)
        nlC = (C + KC - 1) // KC

        @T.prim_func
        def finalize(
            partial_sum: T.Tensor((C, SPLITS), DT),
            partial_sum_xhat: T.Tensor((C, SPLITS), DT),
            weight: T.Tensor((C,), DT),
            rstd: T.Tensor((C,), DT),
            grad_weight: T.Tensor((C,), DT),
            grad_bias: T.Tensor((C,), DT),
            scale_out: T.Tensor((C,), DT),
            shift_out: T.Tensor((C,), DT),
            center_out: T.Tensor((C,), DT),
        ):
            with T.Kernel(min(C, VEC), is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    pw = T.alloc_ub((1, SPLITS), DT)
                    pwx = T.alloc_ub((1, SPLITS), DT)
                    tot = T.alloc_ub((1, 1), DT)
                    totx = T.alloc_ub((1, 1), DT)
                    w1 = T.alloc_ub((1, 1), DT)
                    r1 = T.alloc_ub((1, 1), DT)
                    sc = T.alloc_ub((1, 1), DT)
                    sh = T.alloc_ub((1, 1), DT)
                    ce = T.alloc_ub((1, 1), DT)
                    for i in T.serial(nlC):
                        c = i * KC + cid
                        if c < C:
                            T.copy(partial_sum[c:c + 1, 0:SPLITS], pw)
                            T.copy(partial_sum_xhat[c:c + 1, 0:SPLITS], pwx)
                            T.reduce(pw, tot, dims=[1], reduce_mode="sum", clear=True)
                            T.reduce(pwx, totx, dims=[1], reduce_mode="sum", clear=True)
                            T.copy(tot, grad_bias[c:c + 1])
                            T.copy(totx, grad_weight[c:c + 1])
                            T.copy(weight[c:c + 1], w1)
                            T.copy(rstd[c:c + 1], r1)
                            T.vmul(w1, r1, sc)               # scale = w*rstd
                            T.vmul(sc, 1.0 / float(L), sh)   # per = scale / L
                            T.vmul(sh, tot, sh)              # shift = per*gb
                            T.vmul(w1, r1, ce)
                            T.vmul(ce, 1.0 / float(L), ce)
                            T.vmul(ce, totx, ce)             # ce = per*gw
                            T.vmul(ce, r1, ce)               # ce = per*gw*rstd
                            T.copy(sc, scale_out[c:c + 1])
                            T.copy(sh, shift_out[c:c + 1])
                            T.copy(ce, center_out[c:c + 1])
        return finalize

    @tilelang.jit(target="npuir")
    def _apply():
        KA = min(C, VEC)
        nlA = (C + KA - 1) // KA

        @T.prim_func
        def apply(
            grad_out: T.Tensor((N, C, S), DT),
            x: T.Tensor((N, C, S), DT),
            mean: T.Tensor((C,), DT),
            scale: T.Tensor((C,), DT),
            shift: T.Tensor((C,), DT),
            center: T.Tensor((C,), DT),
            grad_x: T.Tensor((N, C, S), DT),
        ):
            with T.Kernel(min(C, VEC), is_npu=True) as (cid, _):
                with T.Scope("Vector"):
                    gt = T.alloc_ub((N, S), DT)
                    xt = T.alloc_ub((N, S), DT)
                    brc = T.alloc_ub((N, S), DT)
                    m1 = T.alloc_ub((1, 1), DT)
                    s1 = T.alloc_ub((1, 1), DT)
                    h1 = T.alloc_ub((1, 1), DT)
                    c1 = T.alloc_ub((1, 1), DT)
                    for i in T.serial(nlA):
                        c = i * KA + cid
                        if c < C:
                            T.copy(grad_out[0:N, c, 0:S], gt)
                            T.copy(x[0:N, c, 0:S], xt)
                            T.copy(mean[c:c + 1], m1)
                            T.copy(scale[c:c + 1], s1)
                            T.copy(shift[c:c + 1], h1)
                            T.copy(center[c:c + 1], c1)
                            T.vbrc(m1, brc)
                            T.vsub(xt, brc, xt)             # x - mean
                            T.vbrc(s1, brc)
                            T.vmul(gt, brc, gt)             # scale*g
                            T.vbrc(h1, brc)
                            T.vsub(gt, brc, gt)             # - shift
                            T.vbrc(c1, brc)
                            T.vmul(xt, brc, brc)            # center*(x-mean)
                            T.vsub(gt, brc, gt)             # - center*(x-mean)
                            T.copy(gt, grad_x[0:N, c, 0:S])
        return apply

    return _stats, _finalize, _apply


def golden(grad_out, x, weight, mean, rstd):
    n, c, s = grad_out.shape
    ln = n * s
    go = grad_out.float()
    xh = (x.float() - mean.view(1, c, 1)) * rstd.view(1, c, 1)
    gbi = go.sum(dim=(0, 2))
    gwi = (go * xh).sum(dim=(0, 2))
    gx = (weight.view(1, c, 1) * rstd.view(1, c, 1) / ln) * (
        ln * go - gbi.view(1, c, 1) - xh * gwi.view(1, c, 1)
    )
    return gx, gwi, gbi


if __name__ == "__main__":
    print("device:", torch.npu.get_device_name(0), flush=True)
    torch.manual_seed(0)
    grad_out = torch.randn(N, C, S)
    x = torch.randn(N, C, S)
    weight = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5
    dev = "npu"
    stats, finalize, apply_ = build()
    go_n, x_n, mean_n, rstd_n, w_n = (grad_out.npu(), x.npu(), mean.npu(),
                                      rstd.npu(), weight.npu())
    ps = torch.empty(C, SPLITS, device=dev)
    pxs = torch.empty(C, SPLITS, device=dev)
    stats()(go_n, x_n, mean_n, rstd_n, ps, pxs)
    gw = torch.empty(C, device=dev)
    gb = torch.empty(C, device=dev)
    sc = torch.empty(C, device=dev)
    sh = torch.empty(C, device=dev)
    ce = torch.empty(C, device=dev)
    finalize()(ps, pxs, w_n, rstd_n, gw, gb, sc, sh, ce)
    gx = torch.empty_like(grad_out, device=dev)
    apply_()(go_n, x_n, mean_n, sc, sh, ce, gx)
    gx, gw, gb = gx.cpu(), gw.cpu(), gb.cpu()
    rgx, rgw, rgb = golden(grad_out, x, weight, mean, rstd)
    e_gx = (gx - rgx).abs().max().item()
    e_gw = (gw - rgw).abs().max().item()
    e_gb = (gb - rgb).abs().max().item()
    print(f"[SPLIT] max_abs_err grad_x={e_gx:.3e} grad_weight={e_gw:.3e} grad_bias={e_gb:.3e}",
          flush=True)
    ok = e_gx < 1e-3 and e_gw < 1e-3 and e_gb < 1e-3
    print("SPLIT PROBE PASS" if ok else "SPLIT PROBE FAIL")
