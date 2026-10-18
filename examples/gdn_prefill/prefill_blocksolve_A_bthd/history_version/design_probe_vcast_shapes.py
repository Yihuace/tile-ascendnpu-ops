"""Micro-probe (Stage 1 revision 1): which vcast / vbrc shapes are legal on npuir.

Isolates the failing op from design_probe_gate_sign.py:
  M1  vcast on a (64,1) fp16 UB tile      -> (64,1) fp32
  M2  vcast on a (64,64) fp16 UB tile     -> (64,64) fp32
  M3  vbrc  (64,1) fp32 -> (64,64) fp32   (column broadcast, g_i)
  M4  vbrc  (1,64) fp32 -> (64,64) fp32   (row broadcast, g_j)
  M5  vsub  (64,64),(64,64) -> (64,64)    + scalar vmul on the (64,1) tile

Run:  bash _probe_env.sh python3 design_probe_vcast_shapes.py
"""

import os

os.environ["TILELANG_ASCEND_MODE"] = "Expert"

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 64


@tilelang.jit(out_idx=[], target="npuir")
def build(mode: str):
    @T.prim_func
    def main(
        b16: T.Tensor((C, 1), "float16"),
        b16b: T.Tensor((C, C), "float16"),
        gc: T.Tensor((C, 1), "float32"),
        gr: T.Tensor((1, C), "float32"),
        o1: T.Tensor((C, 1), "float32"),
        o2: T.Tensor((C, C), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Vector"):
                ub16 = T.alloc_ub([C, 1], "float16")
                ub32 = T.alloc_ub([C, 1], "float32")
                ub16b = T.alloc_ub([C, C], "float16")
                ub32b = T.alloc_ub([C, C], "float32")
                ub32c = T.alloc_ub([C, C], "float32")
                gcol = T.alloc_ub([C, 1], "float32")
                grow = T.alloc_ub([1, C], "float32")
                with T.rs("PIPE_MTE2"):
                    T.copy(b16, ub16)
                    T.copy(b16b, ub16b)
                    T.copy(gc, gcol)
                    T.copy(gr, grow)
                if mode == "M1":
                    T.vcast(ub16, ub32, round_mode="rint")
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub32, o1)
                elif mode == "M2":
                    T.vcast(ub16b, ub32b, round_mode="rint")
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub32b, o2)
                elif mode == "M3":
                    T.vbrc(gcol, ub32b)
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub32b, o2)
                elif mode == "M4":
                    T.vbrc(grow, ub32b)
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub32b, o2)
                else:
                    T.vcast(ub16, ub32, round_mode="rint")
                    T.vmul(ub32, T.float32(-1.0), ub32)
                    T.vbrc(ub32, ub32b)
                    T.vbrc(gcol, ub32c)
                    T.vsub(ub32c, ub32c, ub32c)
                    with T.rs("PIPE_MTE3"):
                        T.copy(ub32b, o2)
                        T.copy(ub32, o1)

    return main


torch.manual_seed(0)
b16 = (torch.rand(C, 1, dtype=torch.float64)).half().npu()
b16b = (torch.rand(C, C, dtype=torch.float64)).half().npu()
gc = torch.rand(C, 1, dtype=torch.float64).float().npu()
gr = torch.rand(1, C, dtype=torch.float64).float().npu()
o1 = torch.zeros(C, 1, dtype=torch.float32).npu()
o2 = torch.zeros(C, C, dtype=torch.float32).npu()

for mode in ("M1", "M2", "M3", "M4", "M5"):
    try:
        build(mode)(b16, b16b, gc, gr, o1, o2)
        torch.npu.synchronize()
        if mode == "M1":
            err = float((o1.cpu().double() - b16.cpu().double()).abs().max())
        elif mode == "M2":
            err = float((o2.cpu().double() - b16b.cpu().double()).abs().max())
        elif mode == "M3":
            ref = gc.cpu().double().expand(C, C)
            err = float((o2.cpu().double() - ref).abs().max())
        elif mode == "M4":
            ref = gr.cpu().double().expand(C, C)
            err = float((o2.cpu().double() - ref).abs().max())
        else:
            ref = (-b16.cpu().double()).expand(C, C)
            err = float((o2.cpu().double() - ref).abs().max())
        print(f"PROBE-SHAPE {mode} OK    max_abs_err={err:.3e}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        print(f"PROBE-SHAPE {mode} FAIL  {msg[-260:]}")
