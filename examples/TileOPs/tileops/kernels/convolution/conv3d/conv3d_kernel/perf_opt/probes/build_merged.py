#!/usr/bin/env python3
"""Assemble the merged Stage-4 candidate: R3 (channel-last, c_in_pad>=32) +
R2 (padded-N, c_in_pad==16) + R1 (tap-packed weight-resident) + Stage-3 fallback.

Both R2 and R3 are emitted as sibling compile-time branches (the tilelang parser
cannot cope with prim_funcs nested one level deeper -- round-2 trap), and their
gates are disjoint.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.dirname(HERE)

r2_src = open(os.path.join(PERF, "Conv3dBiasFwdOp_round1_r2_padded_n.py")).read()
r3_src = open(os.path.join(PERF, "Conv3dBiasFwdOp_round2_r3_ndhwc.py")).read()

R2_START = "        # ---- R2: padded-N multi-row spatial tiling"
R2_END = "        if use_r1:\n"
R3_START = "        # ---- R3: padded-N tiling + channel-last input"
assert r2_src.count(R2_START) == 1 and r2_src.count(R2_END) == 1
assert r3_src.count(R3_START) == 1 and r3_src.count(R2_END) == 1

r2_lo, r2_hi = r2_src.index(R2_START), r2_src.index(R2_END)
r3_lo, r3_hi = r3_src.index(R3_START), r3_src.index(R2_END)
r2_block = r2_src[r2_lo:r2_hi]
r3_block = r3_src[r3_lo:r3_hi]

# ---- R2 gate: multi-row tiling over the plane-strided NCDHW input -----------
old = """        use_r2 = (
            _stride1
            and dd == 1
            and dh == 1
            and dw == 1
            and n_kc == 1
            and gy <= num_kernels
            and out_w <= block_n
            and r2_rows >= 2
            and _l1_r2 <= _L1_BUDGET
            and block_m * n_tile2 * 4 <= _L0C_CAP
        )"""
new = """        use_r2 = (
            _stride1
            and dd == 1
            and dh == 1
            and dw == 1
            and n_kc == 1
            and gy <= num_kernels
            and out_w <= block_n
            and r2_rows >= 2
            and c_in_pad < 32
            and _l1_r2 <= _L1_BUDGET
            and block_m * n_tile2 * 4 <= _L0C_CAP
        )"""
assert old in r2_block
r2_block = r2_block.replace(old, new)

# ---- R3 gate: channel-last contiguous copies need a >= 64 B inner run -------
old = """            and c_in_pad >= 16
            and c_in_pad % 16 == 0"""
new = """            and c_in_pad >= 32
            and c_in_pad % 16 == 0"""
assert old in r3_block
r3_block = r3_block.replace(old, new)
# rename R3's local `hgroups` so the two sibling blocks cannot shadow each other
r3_block = r3_block.replace("hgroups", "hgroups3")

merged = r3_src[:r3_lo] + r2_block + r3_block + r3_src[r3_hi:]

out = os.path.join(PERF, "Conv3dBiasFwdOp_round3_merged.py")
open(out, "w").write(merged)
ast.parse(merged)
print("written", out, len(merged.splitlines()), "lines")
