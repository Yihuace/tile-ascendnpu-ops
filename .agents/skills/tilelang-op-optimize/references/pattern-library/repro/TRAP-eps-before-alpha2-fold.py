# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-eps-before-alpha2-fold -- with the O2 shortcut
``S_vhat = alpha^2 * sum(v^2)``, the RMS epsilon must be added **after** the
``alpha^2`` fold.  Adding it before is invisible for ``alpha = O(1)`` and
silently wrong by many orders of magnitude when the gate saturates to 0.

Entry:  traps-compiler.md / TRAP-eps-before-alpha2-fold  (numerics trap)
Kind:   trap (reverse experience).  Assertions:
          * correct order -> matches the RMSNorm definition (rel err <= 1e-6)
            for both O(1) and saturated alpha;
          * wrong order   -> relative error > 1e3 at the saturated gate, while
            it agrees to < 1e-6 at O(1) alpha (i.e. no ordinary test catches it).

Origin: the two-stage split computes r_v from the *global* ``sum v^2`` (to
avoid a d-length pass in stage B), so the mean square is folded with
``alpha^2``.  The stage-3 implementation wrote
``s = (Sv/d + eps); s = s * alpha^2; r_v = sqrt(s)/s`` and passed every
manifest workload; the gate-saturation corner (h*50, W*0.5, rms_w=5, which
drives alpha to 1.5e-10) gave y_maxdiff = 7.7 because alpha^2 * eps underflows
the eps that should dominate.  Contract form is
``r_v = 1/sqrt(alpha^2 * mean(v^2) + eps)``.

First verified: 2026-10-31, tilelang 0.1.2+3ed7c914 / CANN 9.0.0 / Ascend910B3.
origin_task: examples/engram_decode_mcore/engram_decode_mcore (Stage 3, L1-3)
Re-verify: python repro/TRAP-eps-before-alpha2-fold.py
"""

import torch

EPS = 1e-6


def _reference(v, alpha):
    """RMSNorm definition: rrms = (mean((alpha*v)^2) + eps)^-1/2."""
    x = alpha * v
    return (x ** 2).mean(-1, keepdim=True).add(EPS).rsqrt()


def _correct(v, alpha, sv):
    return (alpha ** 2 * (sv / v.shape[-1]) + EPS).rsqrt()


def _wrong(v, alpha, sv):
    return ((sv / v.shape[-1] + EPS) * alpha ** 2).rsqrt()


def _rel(a, b):
    return ((a - b).abs() / b.abs().clamp_min(1e-30)).max().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 256
    v = torch.randn(2, d) * 100.0          # O(1e2) magnitudes, like the corner
    sv = (v ** 2).sum(-1, keepdim=True)

    for name, alpha in (("O(1) gate", torch.tensor([[0.7], [0.3]])),
                        ("saturated gate", torch.tensor([[1.5e-10], [1.4e-10]]))):
        ref = _reference(v, alpha)
        good = _correct(v, alpha, sv)
        bad = _wrong(v, alpha, sv)
        e_good, e_bad = _rel(good, ref), _rel(bad, ref)
        print(f"[{name:15s}] correct rel={e_good:.2e}  wrong rel={e_bad:.2e}")
        assert e_good <= 1e-6, f"{name}: correct order off ({e_good:.2e})"
        if name == "O(1) gate":
            assert e_bad <= 1e-6, "wrong order should be invisible at O(1) alpha"
        else:
            assert e_bad > 1e3, f"wrong order should blow up ({e_bad:.2e})"

    print("[ok] eps must be added after the alpha^2 fold "
          "(invisible at O(1) alpha, >1e3x error when the gate saturates)")
    print("OVERALL: TRAP_PASS")
