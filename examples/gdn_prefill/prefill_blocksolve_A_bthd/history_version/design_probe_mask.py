"""Micro-probe (Stage 1 revision 1): is the strict-lower bool mask built correctly?

The §3.3 mask is an arange -> reshape -> vbrc -> vcmp chain, used as the Cond of
vselect.  design_probe_gate_sign.py showed a large residual in the *masked-out*
upper triangle, so this probe dumps the mask itself (as fp32) plus the same mask
produced by an alternative "load a constant 0/1 tile from GM + vmul" path.

  MA : arange(1-D) -> reshape (1,64)/(64,1) -> vbrc -> vcmp("gt") -> vselect(1,0)
  MB : constant fp32 0/1 tile from GM, multiplied in (no bool, no arange)

Run:  bash _probe_env.sh python3 design_probe_mask.py
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
        m01: T.Tensor((C, C), "float32"),
        out: T.Tensor((C, C), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Vector"):
                idx = T.alloc_ub([C], "float32")
                idxc = T.alloc_ub([C], "float32")
                irow = T.alloc_ub([1, C], "float32")
                icol = T.alloc_ub([C, 1], "float32")
                ii = T.alloc_ub([C, C], "float32")
                jj = T.alloc_ub([C, C], "float32")
                mk = T.alloc_ub([C, C], "float32")
                one = T.alloc_ub([C, C], "float32")
                zero = T.alloc_ub([C, C], "float32")
                cond = T.alloc_ub([C, C], "bool")
                with T.rs("PIPE_MTE2"):
                    T.copy(m01, mk)
                T.clear(one)
                T.vmul(mk, T.float32(0.0), zero)
                if mode == "MA":
                    T.arange(idx, [1], 0)
                    T.reshape(idx, irow)
                    T.reshape(idx, icol)
                    T.vbrc(irow, ii)
                    T.vbrc(icol, jj)
                    T.vcmp(ii, jj, cond, "gt")
                    T.vselect(cond, mk, zero, one)
                elif mode == "MC":
                    # two independent 1-D aranges (one per axis) -> reshape -> vbrc -> vcmp
                    T.arange(idx, [1], 0)
                    T.arange(idxc, [1], 0)
                    T.reshape(idx, irow)
                    T.reshape(idxc, icol)
                    T.vbrc(irow, ii)
                    T.vbrc(icol, jj)
                    T.vcmp(ii, jj, cond, "gt")
                    T.vselect(cond, mk, zero, one)
                elif mode == "MD":
                    # no bool at all: clamp then multiply by the constant 0/1 tile
                    T.vmin(mk, T.float32(0.5), ii)   # ii = 0 where mask==0, 0.5 where mask==1
                    T.vsub(mk, ii, jj)               # jj = 1 where mask==1, 0 elsewhere
                    T.vmul(mk, jj, one)              # one == mask
                else:
                    T.copy(mk, one)
                with T.rs("PIPE_MTE3"):
                    T.copy(one, out)

    return main


torch.manual_seed(0)
M = torch.tril(torch.ones(C, C, dtype=torch.float32), -1).npu()
out = torch.zeros(C, C, dtype=torch.float32).npu()
ref = torch.tril(torch.ones(C, C, dtype=torch.float64), -1)

for mode in ("MA", "MC", "MD", "MB"):
    try:
        build(mode)(M, out)
        torch.npu.synchronize()
        got = out.cpu().double()
        print(f"PROBE-MASK {mode} max_abs_err={float((got - ref).abs().max()):.3e} "
              f"sum={float(got.sum()):.1f} (ref sum={float(ref.sum()):.1f})")
        if mode == "MA":
            bad_rows = (got != ref).any(dim=1).nonzero().flatten().tolist()
            print(f"PROBE-MASK MA mismatching rows: {bad_rows[:8]}{' ...' if len(bad_rows) > 8 else ''}")
            print(f"PROBE-MASK MA row0 first 8 : {got[0, :8].tolist()}")
            print(f"PROBE-MASK MA row5 first 8 : {got[5, :8].tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE-MASK {mode} FAIL")
        print(str(exc)[-900:])
