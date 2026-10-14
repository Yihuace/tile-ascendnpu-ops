"""Scratch v10: oversized-S path (in-core serial S-tiling, no cross-core merge).

One core per channel; a plane tile (N, bS) is accumulated elementwise across
S-tiles (UB-bounded), then reduced; a second pass re-reads the tiles for
grad_x.  Proves the large-spatial manifest workload needs no cross-core merge
and no atomics (the GPU split kernel's C*splits blocks + finalize merge is not
required on NPU).
"""
import os
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

N, C, S = 4, 8, 512
BS = 128
NT = S // BS
L = N * S
DT = "float32"


@tilelang.jit(out_idx=[5, 6, 7], target="npuir")
def _f():
    @T.prim_func
    def main(
        grad_out: T.Tensor((N, C, S), DT),
        x: T.Tensor((N, C, S), DT),
        weight: T.Tensor((C,), DT),
        mean: T.Tensor((C,), DT),
        rstd: T.Tensor((C,), DT),
        grad_x: T.Tensor((N, C, S), DT),
        grad_weight: T.Tensor((C,), DT),
        grad_bias: T.Tensor((C,), DT),
    ):
        with T.Kernel(C, is_npu=True) as (cid, _):
            gt = T.alloc_shared((N, BS), DT)
            xt = T.alloc_shared((N, BS), DT)
            acc_g = T.alloc_shared((N, BS), DT)
            acc_gx = T.alloc_shared((N, BS), DT)
            prod = T.alloc_shared((N, BS), DT)
            brc = T.alloc_shared((N, BS), DT)
            red = T.alloc_shared((N, 1), DT)
            gb = T.alloc_shared((1, 1), DT)
            gw = T.alloc_shared((1, 1), DT)
            w1 = T.alloc_shared((1, 1), DT)
            m1 = T.alloc_shared((1, 1), DT)
            r1 = T.alloc_shared((1, 1), DT)
            coef = T.alloc_shared((1, 1), DT)

            T.copy(weight[cid : cid + 1], w1)
            T.copy(mean[cid : cid + 1], m1)
            T.copy(rstd[cid : cid + 1], r1)
            T.clear(acc_g)
            T.clear(acc_gx)

            # pass 1: elementwise tile accumulation (bounded UB), then one reduce
            for st in T.serial(NT):
                s0 = st * BS
                T.copy(grad_out[0:N, cid, s0 : s0 + BS], gt)
                T.copy(x[0:N, cid, s0 : s0 + BS], xt)
                T.vbrc(m1, brc)
                T.vsub(xt, brc, xt)
                T.vbrc(r1, brc)
                T.vmul(xt, brc, xt)          # xt = x_hat
                T.vadd(acc_g, gt, acc_g)
                T.vmul(gt, xt, prod)
                T.vadd(acc_gx, prod, acc_gx)

            T.reduce(acc_g, red, dims=[1], reduce_mode="sum", clear=True)
            T.reduce(red, gb, dims=[0], reduce_mode="sum", clear=True)
            T.reduce(acc_gx, red, dims=[1], reduce_mode="sum", clear=True)
            T.reduce(red, gw, dims=[0], reduce_mode="sum", clear=True)
            T.copy(gb, grad_bias[cid : cid + 1])
            T.copy(gw, grad_weight[cid : cid + 1])

            T.vmul(w1, r1, coef)
            T.vmul(coef, 1.0 / float(L), coef)

            # pass 2: re-read tiles, apply grad_x epilogue
            for st in T.serial(NT):
                s0 = st * BS
                T.copy(grad_out[0:N, cid, s0 : s0 + BS], gt)
                T.copy(x[0:N, cid, s0 : s0 + BS], xt)
                T.vbrc(m1, brc)
                T.vsub(xt, brc, xt)
                T.vbrc(r1, brc)
                T.vmul(xt, brc, xt)
                T.vmul(gt, float(L), prod)
                T.vbrc(gb, brc)
                T.vsub(prod, brc, prod)
                T.vbrc(gw, brc)
                T.vmul(xt, brc, brc)
                T.vsub(prod, brc, prod)
                T.vbrc(coef, brc)
                T.vmul(prod, brc, prod)
                T.copy(prod, grad_x[0:N, cid, s0 : s0 + BS])
    return main


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
    print("device:", torch.npu.get_device_name(0))
    torch.manual_seed(0)
    grad_out = torch.randn(N, C, S)
    x = torch.randn(N, C, S)
    weight = torch.randn(C)
    mean = torch.randn(C)
    rstd = torch.rand(C).abs() + 0.5
    gx, gw, gb = _f()(grad_out.npu(), x.npu(), weight.npu(), mean.npu(), rstd.npu())
    gx, gw, gb = gx.cpu(), gw.cpu(), gb.cpu()
    rgx, rgw, rgb = golden(grad_out, x, weight, mean, rstd)
    e_gx = (gx - rgx).abs().max().item()
    e_gw = (gw - rgw).abs().max().item()
    e_gb = (gb - rgb).abs().max().item()
    print(f"[TILED] max_abs_err grad_x={e_gx:.3e} grad_weight={e_gw:.3e} grad_bias={e_gb:.3e}")
    ok = e_gx < 1e-4 and e_gw < 1e-4 and e_gb < 1e-4
    print("TILED PASS" if ok else "TILED FAIL")
