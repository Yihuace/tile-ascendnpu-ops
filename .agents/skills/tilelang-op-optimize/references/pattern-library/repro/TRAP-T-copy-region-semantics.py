"""[repro] TRAP-T-copy-region-semantics -- pattern-library/traps-runtime.md

Phenomenon: T.copy region semantics (from tilelang/language/copy.py L42-95
source semantics): scalar-base + size extents FORWARD-FILL (size=[1,N] lands
on the last dim, [N,1] on the second-to-last -- this decides lse write-out
direction); mismatched src/dst element counts make the engine copy by src and
OVERWRITE the neighbouring GM tensor (symptoms mimic a cross-pipe race; only
appears on non-divisible tail shapes; correct form = src slice
[0:real_rows]); an [N,1] UB source with size=[1,N] reads out-of-bounds by
stride (MTE DDR fault) -- a row vector destined for a GM tail-contiguous
region must be transposed to [1,N] first (Developer shared sources do not
need this; UB sources MUST). Recommended general form: explicit SLICE on both
src and dst.

Assertion (trap, legal-form-pass): two GM tiles holding row-index (i) and
col-index (j) patterns are slice-loaded into UB, vadd'ed to ub_o[i,j] = i + j
(dense, position-sensitive -- any mis-strided or overflowing region write
breaks it), and written out via the SLICE form; the result matches exactly.

First verified: tilelang dev root build 2026-09-07 (HEAD 21586b5) + CANN 8.5.0
+ Ascend910B2C (debug_log D2/D3; original probes probe_copyforms.py /
probe_lsemulti2.py).
origin_task (provenance, may rot): multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
Re-verification log (append-only):
  2026-09-10 PASS slice form (this toolchain, arange-free variant).
  2026-09-10 NOTE the historical scalar-base+size=[HALF,D] form (probe_copyforms
    C6) now FAULTS on the current toolchain with "DDR address of the MTE
    instruction is out of range" (it passed on 2026-09-07). Per the version
    stamp rule that form is pending re-verification; the slice form is the
    verified survivor. See traps-runtime.md entry note.
  2026-09-10 OBSERVATION (undocumented hypothesis, not yet a library entry):
    T.arange(ub, [1,0] / [0,1], 0) writing a UB tile faults with the same MTE
    DDR error on the current toolchain (passed 2026-09-07); this repro was
    rewritten to derive patterns from GM inputs instead. Candidate D-class
    trap for evolver distillation after doc-legal-form checking.
"""

import os

import numpy as np
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

HALF = 32
D = 64


@tilelang.jit(out_idx=[-1], target="npuir")
def region_probe():
    @T.prim_func
    def main(
        din_r: T.Tensor((1, 64, 4, D), "float32"),
        din_c: T.Tensor((1, 64, 4, D), "float32"),
        out_o: T.Tensor((1, 64, 4, D), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("Vector"):
                ub_r = T.alloc_ub([HALF, D], "float32")
                ub_c = T.alloc_ub([HALF, D], "float32")
                ub_o = T.alloc_ub([HALF, D], "float32")
                with T.rs("PIPE_MTE2"):
                    T.copy(din_r[0, 0:HALF, 0, 0:D], ub_r[0:HALF, 0:D])
                    T.copy(din_c[0, 0:HALF, 0, 0:D], ub_c[0:HALF, 0:D])
                with T.rs("PIPE_V"):
                    T.vadd(ub_r, ub_c, ub_o)
                with T.rs("PIPE_MTE3"):
                    # verified form: explicit slices on src and dst
                    T.copy(ub_o[0:HALF, 0:D], out_o[0, 0:HALF, 0, 0:D])

    return main


def main():
    f = region_probe()
    shape = (1, 64, 4, D)
    r = torch.zeros(shape, device="npu")
    c = torch.zeros(shape, device="npu")
    r[0, :HALF, 0, :] = torch.arange(HALF, dtype=torch.float32, device="npu").reshape(
        HALF, 1
    )
    c[0, :HALF, 0, :] = torch.arange(D, dtype=torch.float32, device="npu").reshape(1, D)
    (o,) = f(r, c)
    torch.npu.synchronize()
    got = o.reshape(1, 64, 4, D)[0, :HALF, 0, :].cpu().numpy()
    rr, cc = np.meshgrid(np.arange(HALF), np.arange(D), indexing="ij")
    ok = bool((got == (rr + cc)).all())
    print(f"[H,D]ub -> 4D slice form: got[0,:4]={got[0, :4]} match={ok}")
    assert ok, "slice-form region write must deliver ub_o[i,j]=i+j exactly"
    print("TRAP-T-copy-region-semantics: PASS (slice form)")


if __name__ == "__main__":
    main()
