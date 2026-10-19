"""K4 `correct_initial_states` split prototype.

Cube:   corr = M @ h_prev
Vector: cp_h0 = ht + (fallback ? corr : 0)
Host loops over the context-parallel partitions.
"""

import argparse

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2], target="npuir")
def corr_kernel(CP: int, H: int, DK: int, DV: int, dtype: str = "float32"):
    @T.prim_func
    def main(
        h_prev: T.Tensor((CP, H, DK, DV), dtype),
        mt: T.Tensor((CP, H, DK, DK), dtype),
        corr: T.Tensor((CP, H, DK, DV), "float32"),
    ):
        with T.Kernel(CP * H, is_npu=True) as (bid, _), T.Scope("Cube"):
            cp = bid // H
            h = bid % H
            h_l1 = T.alloc_L1([DK, DV], dtype)
            m_l1 = T.alloc_L1([DK, DK], dtype)
            c_l0 = T.alloc_L0C([DK, DV], "float32")
            T.copy(h_prev[cp, h, 0:DK, 0:DV], h_l1)
            T.copy(mt[cp, h, 0:DK, 0:DK], m_l1)
            T.gemm(m_l1, h_l1, c_l0, size=[DK, DK, DV], initC=True)
            with T.rs("PIPE_FIX"):
                T.copy(c_l0, corr[cp, h, 0:DK, 0:DV])

    return main


@tilelang.jit(target="npuir")
def add_kernel(CP: int, H: int, DK: int, DV: int):
    @T.prim_func
    def main(
        ht: T.Tensor((CP, H, DK, DV), "float32"),
        corr: T.Tensor((CP, H, DK, DV), "float32"),
        fallback: T.Tensor((CP, H), "int32"),
        cp_h0: T.Tensor((CP, H, DK, DV), "float32"),
        s_idx: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(CP * H, is_npu=True) as (bid, _):
            cp = bid // H
            h = bid % H
            s = s_idx[0]
            ht_ub = T.alloc_ub((DK, DV), "float32")
            c_ub = T.alloc_ub((DK, DV), "float32")
            o_ub = T.alloc_ub((DK, DV), "float32")
            T.copy(ht[s, h, 0:DK, 0:DV], ht_ub)
            if fallback[s, h] > 0:
                T.copy(corr[s, h, 0:DK, 0:DV], c_ub)
                T.vadd(ht_ub, c_ub, o_ub)
            else:
                T.copy(ht_ub, o_ub)
            T.copy(o_ub, cp_h0[s + 1, h, 0:DK, 0:DV])

    return main


def k4_split(ht, mt, fallback, raw_h0, seq_map, DK, DV, H):
    CP = ht.shape[0]
    z = torch.zeros((CP, H, DK, DV), dtype=torch.float32, device=ht.device)
    cp_h0 = z.clone()
    kc = corr_kernel(CP, H, DK, DV, "float32")
    ka = add_kernel(CP, H, DK, DV)
    for bb in range(len(seq_map) - 1):
        lo, hi = int(seq_map[bb]), int(seq_map[bb + 1])
        if raw_h0 is not None:
            cp_h0[lo] = raw_h0[bb].float()
        for s in range(lo, hi - 1):
            corr = kc(cp_h0, mt)
            idx = torch.tensor([s], dtype=torch.int32, device=ht.device)
            ka(ht, corr, fallback, cp_h0, idx)
    return cp_h0


def golden_k4(ht, mt, fallback, raw_h0, seq_map):
    CP, H, DK, DV = ht.shape
    out = torch.zeros_like(ht)
    for bb in range(len(seq_map) - 1):
        lo, hi = int(seq_map[bb]), int(seq_map[bb + 1])
        if raw_h0 is not None:
            out[lo] = raw_h0[bb].float()
        for s in range(lo, hi - 1):
            out[s + 1] = ht[s].clone()
            for h in range(H):
                if int(fallback[s, h]) > 0:
                    out[s + 1, h] += mt[s, h] @ out[s, h]
    return out


def run_case(CP=4, H=2, DK=64, DV=64, has_raw=True, seed=0):
    torch.manual_seed(seed)
    dev = "npu"
    ht = torch.randn(CP, H, DK, DV, dtype=torch.float32, device=dev) * 0.1
    mt = torch.randn(CP, H, DK, DK, dtype=torch.float32, device=dev) * 0.05
    fallback = torch.randint(0, 2, (CP, H), dtype=torch.int32, device=dev)
    raw = torch.randn(1, H, DK, DV, dtype=torch.float32, device=dev) if has_raw else None
    seq_map = torch.tensor([0, CP], dtype=torch.int32)
    out = k4_split(ht, mt, fallback, raw, seq_map, DK, DV, H)
    torch.npu.synchronize()
    ref = golden_k4(ht.cpu(), mt.cpu(), fallback.cpu(), raw.cpu() if raw is not None else None, seq_map.cpu())
    err = float((out.cpu() - ref).abs().max())
    print(f"[K4-split] CP={CP} H={H} DK={DK} DV={DV} raw={has_raw} max_abs_err={err:.3e}")
    assert err < 2e-3, err
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--CP", type=int, default=4)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--DK", type=int, default=64)
    ap.add_argument("--DV", type=int, default=64)
    args = ap.parse_args()
    run_case(args.CP, args.H, args.DK, args.DV, True, 0)
    run_case(args.CP, args.H, args.DK, args.DV, False, 1)
    print("[K4-split] PASS")


if __name__ == "__main__":
    main()
