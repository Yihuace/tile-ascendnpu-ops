#!/usr/bin/env python3
"""Copy-split probes (v2): shrink one Cube-side copy to 16x16 instead of deleting.

Deleting the copy makes l1_a/l1_b dead and the BiShengIR backend dies with
`Unknown core type: llvm.func @malloc` (a degenerate-buffer codegen trap), so
the probes shrink the copy extent to 16x16 (512B) instead -- the buffer stays
live and the kernel compiles.

  probes/probe_no_b.py : Cube B (im2col input) copy shrunk  -> measures A-side cost
  probes/probe_no_a.py : Cube A (weight)    copy shrunk     -> measures B-side cost
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "Conv3dBiasFwdOp.py")
src = open(SRC).read()

old_gate = """        use_r1 = (
            _stride1
            and n_kc == 1
            and gy <= num_kernels
            and c_in_pad == 16
            and _l1_total <= _L1_BUDGET
        )"""
base = src.replace(old_gate, "        use_r1 = False")

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
no_a = base.replace(a_copy, """                                T.copy(
                                    w_flat[tap, 0:16, 0:16],
                                    l1_a[0:16, 0:16],
                                )
""")

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
no_b = base.replace(b_copy, """                                    T.copy(
                                        x_pad[bz, 0:16, off:off + 16],
                                        l1_b[0:16, 0:16],
                                    )
""")

open(os.path.join(HERE, "probe_no_a.py"), "w").write(no_a)
open(os.path.join(HERE, "probe_no_b.py"), "w").write(no_b)
print("wrote probe_no_a.py / probe_no_b.py")
