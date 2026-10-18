"""Repro / regression watchdog: bishengir-compile SIGSEGV on Cube (mmadL1) buffer-form IR.

Gap: CG-2026-0003 (ConvertHFusionToHIVM walks buffer-form hivm ops and reads
``op.getResult(0)``; a 0-result op makes that read out of bounds).  The pass either
dies with SIGSEGV or survives depending on the incidental IR shape, so the *same*
kernel compiles for some (total_tokens, head) pairs and not others.

Measured 2026-10-17 (tilelang 0.1.2 dev root, CANN 9.0.0 / bishengir 1.1.0, Ascend910B3):
    S=64,  head=1,  dim_k=64   -> compiles, max_abs_err 3.8e-5
    S=100, head=1,  dim_k=64   -> compiles (tail chunk 36 rows)
    S=64,  head=16, dim_k=128  -> compiles
    S=64,  head=2,  dim_k=64   -> SIGSEGV
    S=1600/3000/4096, head 1..16 -> SIGSEGV

IR-level bisect (same source, one change at a time):
    drop the R3-a ``T.copy(kz -> l1k)``          -> compiles
    drop the 6-level doubling chain               -> compiles
    add 2 UB buffers to the Vector scope          -> a previously-good shape now crashes
=> the verdict is memory-corruption dependent, not a clean "always crashes", so a
   surviving shape may also be silently miscompiled; re-verify after any toolchain bump.

This script builds both a known-good and a known-blocked shape and prints one verdict
line.  Exit 0 = the documented behaviour is still what the docs say (either way), 1 = drift.

Run: python3 repro/TRAP-hfusion-mmadl1-segv.py
"""

import os
import sys

os.environ["TILELANG_ASCEND_MODE"] = "Expert"  # must assign: .bashrc exports Developer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

import prefill_blocksolve_A_bthd as op  # noqa: E402

GOOD = (64, 1, 1, 64)      # S, head, key_head, dim_k
BLOCKED = (64, 2, 1, 64)


def build(shape):
    total_tokens, head, key_head, dim_k = shape
    try:
        op.prefill_blocksolve_A_bthd_tl(
            total_tokens, 1, head, key_head, 64, dim_k, "float16", "int32", total_tokens
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return "crash" if "-11" in str(exc) or "bug report" in str(exc) else "error"


good = build(GOOD)
blocked = build(BLOCKED)
print(f"REPRO-GAP good{GOOD}={good}  blocked{BLOCKED}={blocked}")

if good == "ok" and blocked == "crash":
    print("REPRO-GAP verdict: CG-2026-0003 still reproduces (SIGSEGV on Cube buffer-form IR)")
    sys.exit(0)
if good == "ok" and blocked == "ok":
    print("REPRO-GAP verdict: toolchain changed -- the crash no longer reproduces; re-run the "
          "full L0/L1/L2 suite and retire CG-2026-0003 if it is clean")
    sys.exit(0)
print("REPRO-GAP verdict: DRIFT -- the known-good shape no longer builds; re-measure")
sys.exit(1)
