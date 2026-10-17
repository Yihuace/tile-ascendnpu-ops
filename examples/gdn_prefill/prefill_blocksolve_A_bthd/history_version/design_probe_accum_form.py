"""Design probe #2 (Stage 1, D-3): bisect which Cube form makes bishengir crash.

V1  fp32 operand, gemm(I,I)->L0C, then gemm(N,I,initC=False)->same L0C   (accumulate, no store between)
V2  fp32 operand, gemm->store->gemm(initC=True)->store on the SAME L0C   (store-then-reuse)
V3  fp32 operand, gemm->store->gemm(other L1, initC=False)->store        (accumulate after store)
V4  V3 with fp16 operands + fp16 GM scratch (fallback path)
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 64


def _mk(dt, mode):
    @tilelang.jit(out_idx=[1], target="npuir")
    def build():
        @T.prim_func
        def main(x: T.Tensor((1, C, C), dt), out: T.Tensor((1, C, C), dt),
                 eye: T.Tensor((1, C, C), dt), ws: T.Tensor((1, C, C), dt)):
            with T.Kernel(1, is_npu=True) as (cid, _):
                with T.Scope("Cube"):
                    l1x = T.alloc_L1([C, C], dt)
                    l1i = T.alloc_L1([C, C], dt)
                    l1t = T.alloc_L1([C, C], dt)
                    l0 = T.alloc_L0C([C, C], "float32")
                    T.copy(x[0, 0:C, 0:C], l1x)
                    T.copy(eye[0, 0:C, 0:C], l1i)
                    if mode == "V1":
                        T.gemm(l1i, l1i, l0, size=[C, C, C], initC=True)      # I
                        T.gemm(l1x, l1i, l0, size=[C, C, C], initC=False)     # I + N
                        with T.rs("PIPE_FIX"):
                            T.copy(l0, out[0, 0:C, 0:C])
                    elif mode == "V2":
                        T.gemm(l1i, l1i, l0, size=[C, C, C], initC=True)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0, ws[0, 0:C, 0:C])
                        T.copy(ws[0, 0:C, 0:C], l1t)
                        T.gemm(l1x, l1t, l0, size=[C, C, C], initC=True)      # N@I (store-then-reuse)
                        with T.rs("PIPE_FIX"):
                            T.copy(l0, out[0, 0:C, 0:C])
                    elif mode == "V3":
                        T.gemm(l1i, l1i, l0, size=[C, C, C], initC=True)      # l0 = I
                        with T.rs("PIPE_FIX"):
                            T.copy(l0, ws[0, 0:C, 0:C])
                        T.copy(ws[0, 0:C, 0:C], l1t)
                        T.gemm(l1x, l1t, l0, size=[C, C, C], initC=False)     # l0 += N@I
                        with T.rs("PIPE_FIX"):
                            T.copy(l0, out[0, 0:C, 0:C])
        return main

    return build


torch.manual_seed(0)
N = (torch.tril(torch.randn(C, C, dtype=torch.float64), -1) * 0.25)
EYE = torch.eye(C, dtype=torch.float64)

for mode, dt, ref in (
    ("V1", "float32", EYE + N),
    ("V2", "float32", N),
    ("V3", "float32", EYE + N),
    ("V4", "float16", (EYE + N).half().double()),
):
    x = N.float().to(getattr(torch, dt))[None].npu()
    eye = EYE.float().to(getattr(torch, dt))[None].npu()
    ws = torch.zeros(1, C, C, dtype=getattr(torch, dt)).npu()
    try:
        out = _mk(dt, mode)()(x, eye, ws)
        torch.npu.synchronize()
        err = float((out[0].cpu().double() - ref).abs().max())
        print(f"PROBE2 {mode} ({dt}) OK   err={err:.3e}   (ref |x|max={float(ref.abs().max()):.3f})")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        print(f"PROBE2 {mode} ({dt}) FAIL {msg[:160]}")
