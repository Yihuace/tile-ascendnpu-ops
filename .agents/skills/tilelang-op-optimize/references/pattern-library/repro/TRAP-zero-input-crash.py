"""[repro] TRAP-zero-input-crash -- pattern-library/traps-runtime.md

Phenomenon: a zero-input kernel (out_idx-only, no input tensor at all)
reliably crashes at run time with "MTE DDR address out of range" (launch
argument corruption) -- the crash signature is identical to a real
out-of-bounds write and was once misdiagnosed as a copy-form bug (3 probe
rounds wasted). Keeping >= 1 input tensor in the probe kernel makes the same
form pass.

Assertion (trap, workaround-pass): the identical copy form with a dummy input
tensor produces the expected constant fill (all 7.0). To see the crash form,
remove `din` from the kernel signature/call (documented, not executed here --
intentional crashes are not part of the repro suite).

Diagnostic rule encoded (from debug_log D2/D6): when only some cores/rows are
corrupt, first do an element-level fp64 comparison to decide which side
deviates from the true value before assuming a sync/race bug.

First verified: tilelang dev root build 2026-09-07 (HEAD 21586b5) + CANN 8.5.0
+ Ascend910B2C (debug_log D2/D6).
origin_task (provenance, may rot): multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
Re-verification log (append-only):
  2026-09-10 PASS (formalized from /tmp/opencode/probe_copymin.py vbrc_mix).
"""

import os

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Expert")

HALF = 32
D = 64


@tilelang.jit(out_idx=[-1], target="npuir")
def copy_min():
    @T.prim_func
    def main(
        din: T.Tensor((1, 64, 4, D), "float32"),
        out_o: T.Tensor((1, 64, 4, D), "float32"),
    ):
        with T.Kernel(4, is_npu=True) as (cid, vid):
            with T.Scope("Vector"):
                ub_o = T.alloc_ub([HALF, D], "float32")
                with T.rs("PIPE_V"):
                    value = 7.0
                    T.vbrc(value, ub_o)
                with T.rs("PIPE_MTE3"):
                    T.copy(ub_o[0:HALF, 0:D], out_o[0, 0:HALF, 0, 0:D])

    return main


def main():
    f = copy_min()
    # dummy input keeps launch arguments sane (the crash form drops `din`)
    din = torch.zeros(1, 64, 4, D, dtype=torch.float32, device="npu")
    (o,) = f(din)
    torch.npu.synchronize()
    got = o.reshape(1, 64, 4, D)[0, :HALF, 0, :].cpu().numpy()
    ok = bool((got == 7.0).all())
    print(f"FORM=vbrc_mix all7={ok} got[0,:4]={got[0, :4]}")
    assert ok, "same copy form must pass once a dummy input tensor exists"
    print("TRAP-zero-input-crash: PASS (workaround form with dummy input)")


if __name__ == "__main__":
    main()
