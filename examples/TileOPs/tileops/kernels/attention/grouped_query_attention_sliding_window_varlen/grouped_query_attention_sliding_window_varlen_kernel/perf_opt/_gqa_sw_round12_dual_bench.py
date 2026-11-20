"""Round-12 dual-kernel prototype driver (kernel-only msprof caliber).

Splits the frozen single kernel into K1 (QK^T + softmax -> ws_p/scale/ell/lse)
and K2 (P@V + accumulate -> output).  Fixed 8b-long fp16 shape (the manifest's
`llama-3.1-8b-long-w1024-float16`), B=4, q=k=2048, H=32, HKV=8, D=128,
causal, wl=1024.

`launch` mode pre-allocates all workspaces once (reach into the wrapper
closure) and then runs warmup+launch calls of k1 then k2, so the msprof
capture window contains only the two TileLang kernels.

  msprof op --kernel-name=_gqa_sw_k1_main ... python _gqa_sw_round12_dual_bench.py launch
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(HERE, "_gqa_sw_fwd_varlen_kernel_round12_dualkernel.py")
CASE = dict(batch=4, q=2048, k=2048, heads=32, heads_kv=8, dim=128,
            causal=True, wl=1024, wr=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["launch"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--launch", type=int, default=3)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("dual", IMPL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dual"] = mod
    spec.loader.exec_module(mod)

    c = CASE
    batch, d = c["batch"], c["dim"]
    sq, sk = [c["q"]] * batch, [c["k"]] * batch
    total_q, total_k = sum(sq), sum(sk)
    cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0).tolist()),
                        dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0).tolist()),
                        dtype=torch.int32)
    torch.manual_seed(0)
    dt = torch.float16
    q = torch.randn((total_q, c["heads"], d), dtype=dt, device="npu")
    k = torch.randn((total_k, c["heads_kv"], d), dtype=dt, device="npu")
    v = torch.randn((total_k, c["heads_kv"], d), dtype=dt, device="npu")
    cu_q_npu, cu_k_npu = cu_q.npu(), cu_k.npu()

    fn = mod._gqa_sw_fwd_varlen_kernel(
        batch, c["heads"], c["heads_kv"], total_q, total_k, d,
        c["causal"], c["wl"], c["wr"], "float16",
    )
    wrapped = fn(64, 64, 1)
    msq = max(sq)
    t0 = time.perf_counter()
    wrapped(q, k, v, cu_q_npu, cu_k_npu, msq)   # JIT compile both kernels
    torch.npu.synchronize()
    print(f"[dual] first call (compile k1+k2) {time.perf_counter()-t0:.1f}s",
          flush=True)

    cells = {n: cl.cell_contents for n, cl in
             zip(wrapped.__code__.co_freevars, wrapped.__closure__)}
    k1, k2, meta = cells["_compile"](int(msq))
    dev = q.device
    ws_s = torch.empty(meta["ws_s_shape"], dtype=meta["torch_ws_s"], device=dev)
    ws_p = torch.empty(meta["ws_p_shape"], dtype=meta["torch_dtype"], device=dev)
    ws_o = torch.empty(meta["ws_o_shape"], dtype=meta["torch_ws_o"], device=dev)
    ws_scale = torch.zeros(meta["ws_scale_shape"], dtype=torch.float32, device=dev)
    ws_ell = torch.zeros(meta["ws_ell_shape"], dtype=torch.float32, device=dev)
    zbuf = torch.zeros((meta["bn_eff"], d), dtype=meta["torch_dtype"], device=dev)
    mb = (ws_p.numel() * ws_p.element_size()
          + ws_s.numel() * ws_s.element_size()
          + ws_o.numel() * ws_o.element_size()
          + ws_scale.numel() * 4 + ws_ell.numel() * 4) / 2**20
    print(f"[dual] ws_p={tuple(ws_p.shape)} ws_s={tuple(ws_s.shape)} "
          f"ws_o={tuple(ws_o.shape)} total_ws={mb:.1f} MB", flush=True)

    def run_once():
        lse = k1(q, k, cu_q_npu, cu_k_npu, ws_s, ws_p, ws_scale, ws_ell)
        out = k2(v, cu_q_npu, cu_k_npu, ws_p, ws_scale, ws_ell, ws_o, zbuf)
        return out, lse

    for _ in range(args.warmup):
        run_once()
    torch.npu.synchronize()
    out, lse = None, None
    for _ in range(args.launch):
        out, lse = run_once()
    torch.npu.synchronize()
    print("LAUNCH_OK", tuple(out.shape), out.dtype, tuple(lse.shape),
          "finite:", bool(torch.isfinite(out).all().item()), flush=True)
    if args.dump:
        torch.save({"out": out.float().cpu(), "lse": lse.cpu()}, args.dump)
        print("[dual] dumped", args.dump, flush=True)


if __name__ == "__main__":
    main()
