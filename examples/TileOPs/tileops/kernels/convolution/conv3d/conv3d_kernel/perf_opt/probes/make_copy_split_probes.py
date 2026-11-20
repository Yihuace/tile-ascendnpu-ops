#!/usr/bin/env python3
"""Generate copy-split probes from perf_opt/Conv3dBiasFwdOp.py.

These probes are *measurement-only* (numerically wrong on purpose): they delete
one of the two Cube-side GM->L1 copies so msprof can attribute the MTE2 time to
the weight (A) vs im2col input (B) copy.  Never merge these back.

Outputs:
  probes/probe_a_only.py : base path, B (input) copy removed
  probes/probe_b_only.py : base path, A (weight) copy removed
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "Conv3dBiasFwdOp.py")
src = open(SRC).read()

# Force the base (Stage-3 equivalent) path for every shape so the probe measures
# the same loop unet actually runs today.
old_gate = """        use_r1 = (
            _stride1
            and n_kc == 1
            and gy <= num_kernels
            and c_in_pad == 16
            and _l1_total <= _L1_BUDGET
        )"""
new_gate = """        use_r1 = False"""
assert old_gate in src
base = src.replace(old_gate, new_gate)

# --- A copy (weight) inside _cube_base ---------------------------------------
a_copy = """                                T.copy(
                                    w_flat[
                                        tap,
                                        by * block_m:by * block_m + tm,
                                        kc * block_k:kc * block_k + kcl,
                                    ],
                                    l1_a[0:tm, 0:kcl],
                                )
"""
assert a_copy in base
b_only = base.replace(a_copy, "")

# --- B copy (im2col input) inside _cube_base (non-poly branch only) ----------
b_copy = """                                    T.copy(
                                        x_pad[
                                            bz,
                                            kc * block_k:kc * block_k + kcl,
                                            off:off + tn,
                                        ],
                                        l1_b[0:kcl, 0:tn],
                                    )
"""
assert b_copy in base
a_only = base.replace(b_copy, "")

open(os.path.join(HERE, "probe_a_only.py"), "w").write(a_only)
open(os.path.join(HERE, "probe_b_only.py"), "w").write(b_only)
print("wrote probe_a_only.py / probe_b_only.py")
