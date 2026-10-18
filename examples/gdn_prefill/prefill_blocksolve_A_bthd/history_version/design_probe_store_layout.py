"""Design probe #3 (Stage 1 revision 1, D-3): the A-epilogue store path.

Why this probe exists (REVIEW.md blocking issue 3): ``A`` is ``[1, S, head, 64]``,
so the 64 rows of one ``(chunk, head)`` inverse are **not** a contiguous tile --
row stride is ``head * 64`` elements.  A direct
``store_fixpipe(l0t, A[0, base, hid, 0], size=[64, 64])`` covers the wrong region
(front-end pads extent 1 into ``[1, 1, 64, 64]`` -> the (head, 64) plane).

Design candidate verified here (the adopted main path):

    Cube   : gemm -> L0C(fp32) --fixpipe(f32->f16)--> y16[64,64]   (contiguous scratch)
    Vector : y16 -> UB(fp16) --T.copy--> A3[base:base+64, hid:hid+1, 0:64]

i.e. one contiguous fixpipe store + one strided region store (the size-1 head axis
folds, so the copy is "64 rows x 64 contiguous elements, row stride = head*64").

Modes
  contig  : the same UB tile written to a flat [64,64] scratch (one 8 KB segment)
  strided : written into A3[.., hid:hid+1, ..]                 (64 x 128 B segments)

Checks
  C1 correctness : A3[0:64, hid, :] == x.half(), and every other (token, head)
                   slot of A3 stays 0 (no write outside the target plane).
  C2 timing      : per-store cost of both modes (reps-delta method) so the design
                   roofline prices the strided epilogue instead of guessing.

Measured (2026-10-17, CANN 9.0.0 / tilelang dev root / Ascend910B3): see stdout
(PROBE-STORE-* lines); numbers are transcribed into DESIGN.md §1.6.3 / §3.4.

Run:  bash _probe_env.sh python3 design_probe_store_layout.py
"""

import os
import time

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # MUST force: .bashrc exports Developer

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

C = 64
HEAD = 4          # small head count so the whole A3 fits comfortably
SPAD = 128        # token rows (2 chunks)
HID = 2           # the head whose inverse we write
DT_OUT = "float16"


@tilelang.jit(out_idx=[], target="npuir")
def build(mode: str, reps: int, out_dtype: str = DT_OUT):
    @T.prim_func
    def main(
        x: T.Tensor((1, C, C), "float32"),
        eye: T.Tensor((1, C, C), "float32"),
        y16: T.Tensor((1, C, C), out_dtype),
        flat: T.Tensor((C, C), out_dtype),
        A3: T.Tensor((SPAD, HEAD, C), out_dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                l1x = T.alloc_L1([C, C], "float32")
                l1i = T.alloc_L1([C, C], "float32")
                l0 = T.alloc_L0C([C, C], "float32")
                T.copy(x[0, 0:C, 0:C], l1x)
                T.copy(eye[0, 0:C, 0:C], l1i)
                T.gemm(l1i, l1x, l0, size=[C, C, C], initC=True)
                with T.rs("PIPE_FIX"):
                    T.copy(l0, y16[0, 0:C, 0:C])       # fixpipe f32 -> f16
                    T.sync_block_set(0)

            with T.Scope("Vector"):
                ub = T.alloc_ub([C, C], out_dtype)
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(y16[0, 0:C, 0:C], ub)
                with T.rs("PIPE_MTE3"):
                    for _ in T.serial(reps):
                        if mode == "contig":
                            T.copy(ub, flat[0:C, 0:C])
                        else:
                            T.copy(ub, A3[0:C, HID:HID + 1, 0:C])

    return main


torch.manual_seed(0)
X = (torch.eye(C, dtype=torch.float64)
     + torch.tril(torch.randn(C, C, dtype=torch.float64), -1) * 0.2).float()
EYE = torch.eye(C, dtype=torch.float64).float()

x_npu = X[None].npu()
eye_npu = EYE[None].npu()
y16 = torch.zeros(1, C, C, dtype=torch.float16).npu()
flat = torch.zeros(C, C, dtype=torch.float16).npu()
A3 = torch.zeros(SPAD, HEAD, C, dtype=torch.float16).npu()

k1 = build("strided", 1)
k1(x_npu, eye_npu, y16, flat, A3)
torch.npu.synchronize()

ref16 = X.double().half().double()
got = A3[0:C, HID, :].cpu().double()
err = float((got - X.double()).abs().max())
other = A3.cpu().double().clone()
other[:, HID, :] = 0.0
print(f"PROBE-STORE-STRIDED  target region == x.half()      : max_abs_err={err:.3e}  |ref|max={float(X.abs().max()):.3f}")
print(f"PROBE-STORE-STRIDED  other (token, head) slots clean: max|other|={float(other.abs().max()):.3e}")
print(f"PROBE-STORE-STRIDED  fixpipe f32->f16 quant check    : max_abs_err={float((y16[0].cpu().double() - ref16).abs().max()):.3e}")

# ---- timing (reps delta, cancels launch overhead) --------------------------
try:
    r_small, r_big = 2, 202
    res = {}
    for mode in ("contig", "strided"):
        ks = build(mode, r_small)
        kb = build(mode, r_big)
        torch.npu.synchronize()

        def _t(k, iters=20):
            for _ in range(3):
                k(x_npu, eye_npu, y16, flat, A3)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                k(x_npu, eye_npu, y16, flat, A3)
            torch.npu.synchronize()
            return (time.perf_counter() - t0) / iters

        ts, tb = _t(ks), _t(kb)
        res[mode] = (tb - ts) / (r_big - r_small)
        print(f"PROBE-STORE-TIMING {mode:7s} kernel(reps={r_small})={ts * 1e6:.1f}us  "
              f"kernel(reps={r_big})={tb * 1e6:.1f}us  per-store={res[mode] * 1e6:.3f}us")
    print(f"PROBE-STORE-TIMING strided/contig ratio = {res['strided'] / res['contig']:.2f}x "
          f"(8 KB fp16 tile, row stride = {HEAD * C * 2} B)")
except Exception as exc:  # noqa: BLE001
    print("PROBE-STORE-TIMING FAIL")
    print(str(exc)[:1200])
