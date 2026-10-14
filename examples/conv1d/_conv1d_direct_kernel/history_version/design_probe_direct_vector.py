# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Design-period probe (D-3) for _conv1d_direct_kernel (depthwise Conv1d, Vector).

Question: can the depthwise/direct conv redesign
    out[r=(bz,oc), ol] = sum_kw weight[oc,0,kw] * xpad[oc, ol + kw*d_W]
be expressed, compiled and run on the npuir Vector path in Expert mode with a
single T.Scope("Vector") -- using the GM-offset padded workspace, a weight
row-broadcast vmul ([K_W,BN]*[K_W,1]) and a reduce over dim 0?

Result (Expert): PASS, max_abs_err ~ 7.8e-4 vs F.conv1d(groups=C) golden.
Result (Developer): COMPILE FAIL -- `multiple entry functions detected
(previous: 'main_mix_aiv', this: 'main_mix_aic')` (Developer emits AIV+AIC
entries). Hence the design pins Expert + single T.Scope("Vector").

Toolchain: tilelang dev-root build (/home/developer/tilelang-mlir-ascend/build)
           + CANN 8.5.2 + Ascend 910B3 (get_aicore_num=20), 2026-10-14.
Run: python3 design_probe_direct_vector.py [Expert|Developer]
"""
import os, sys, traceback
MODE = sys.argv[1] if len(sys.argv) > 1 else "Expert"
os.environ["TILELANG_ASCEND_MODE"] = MODE
import torch, torch_npu  # noqa
import tilelang
import tilelang.language as T

N, C, Lin, K, PAD = 1, 4, 8, 3, 1
LOUT = (Lin + PAD + PAD - 1*(K-1) - 1)//1 + 1   # 8
LPAD = 16
BN = 8

@tilelang.jit(out_idx=[2], target="npuir")
def build():
    @T.prim_func
    def main(
        x: T.Tensor((N, C, Lin), "float16"),
        weight: T.Tensor((C, 1, K), "float16"),
        out: T.Tensor((N, C, LOUT), "float16"),
        xpad: T.Tensor((N, C, LPAD), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (kid, _):
            with T.Scope("Vector"):
                ub = T.alloc_ub([C, Lin], "float16")
                T.copy(x[0, 0:C, 0:Lin], ub)
                T.copy(ub, xpad[0, 0:C, PAD:PAD+Lin])   # GM offset destination
                for oc in T.serial(C):
                    xt = T.alloc_ub([K, BN], "float16")
                    for kw in T.serial(K):
                        T.copy(xpad[0, oc, kw:kw+BN], xt[kw:kw+1, 0:BN])
                    xt32 = T.alloc_ub([K, BN], "float32")
                    T.vcast(xt, xt32, round_mode="rint")
                    wc = T.alloc_ub([K, 1], "float16")
                    T.copy(weight[oc, 0, 0:K], wc[0:K, 0:1])
                    w32 = T.alloc_ub([K, 1], "float32")
                    T.vcast(wc, w32, round_mode="rint")
                    prod = T.alloc_ub([K, BN], "float32")
                    T.vmul(xt32, w32, prod)             # row-broadcast [K,BN]*[K,1]
                    acc = T.alloc_ub([1, BN], "float32")
                    T.reduce(prod, acc, dims=[0], reduce_mode="sum", clear=True)
                    ou = T.alloc_ub([1, BN], "float16")
                    T.vcast(acc, ou, round_mode="rint")
                    T.copy(ou[0:1, 0:BN], out[0, oc:oc+1, 0:LOUT])
    return main

def main():
    print("mode", MODE, "tilelang", tilelang.__version__)
    torch.manual_seed(0)
    x = torch.randn(N, C, Lin, dtype=torch.float16)
    w = torch.randn(C, 1, K, dtype=torch.float16)
    golden = torch.nn.functional.conv1d(x.float(), w.float(), padding=PAD, groups=C)[0]
    xp = torch.zeros(N, C, LPAD, dtype=torch.float16)
    kern = build()
    out = kern(x.to("npu"), w.to("npu"), xp.to("npu"))
    torch.npu.synchronize()
    o = out.cpu().float()
    err = float((o - golden).abs().max())
    print("max_abs_err", err, "out_shape", tuple(out.shape))
    print("[PROBE]", "PASS" if err < 1e-1 else "FAIL")
    return 0 if err < 1e-1 else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("[PROBE_FAIL]")
        traceback.print_exc()
        raise SystemExit(2)
