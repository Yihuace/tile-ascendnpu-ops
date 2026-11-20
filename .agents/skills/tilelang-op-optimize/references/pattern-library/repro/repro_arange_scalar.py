# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""CONST-arange-scalar-materialize: T.arange (M,W) hoisted materialization is
scalar-executed (~1ns/elem); (1,N) arange + first-axis vbrc broadcast fixes it.

Delta form (effect manifests at full-kernel scale, PL-1.18 precedent): this
file asserts py_compile on the before/after skeletons; the measured evidence:
  msprof op Task Duration, (2048,4096) fp16 bm=2 resident chain:
    (bm,N)-arange form 46.97us  ->  (1,N)+vbrc form 38.41us  (-18.2%)
  saving 8.56us == (2,4096)=8192 elems x ~1.05ns (scalar-rate model).
  Origin: argmax-_argreduce_kernel-stage4-20260928, opt_log Iteration 4
  (profiles/stage4_r4/hidden-state-argmax-float16__r4{a,b}; note: in-kernel
  20x arange repetition as an amplification probe MISCOMPILES the (bm,N)
  form -- repeated T.arange into the same buffer is not idempotent; do not
  reuse that probe form).First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

SLOW_SKELETON = """
# hoisted once per kernel, per core: (bm, N) index materialization -- scalar
T.arange(idx_j, [0, 1], 0)          # (bm, N) fp32, ~1ns/elem scalar stores
T.vbrc(T.float32(BIG), sent_v)
# ... per-block chain: copy -> reduce_max -> vbrc -> vcmp -> vselect -> ...
"""

FAST_SKELETON = """
# materialize the index row once at (1, N), first-axis broadcast (documented
# vbrc src (1,N,K)->(M,N,K)); bm=1 must arange idx_j directly (same-shape
# vbrc -> empty broadcast dims verify fail, see TRAP-vbrc-same-shape):
if block_m >= 2:
    T.arange(idx_row, [0, 1], 0)    # (1, N) -- vector rate
    T.vbrc(idx_row, idx_j)          # (1,N) -> (bm,N) first-axis broadcast
else:
    T.arange(idx_j, [0, 1], 0)
T.vbrc(T.float32(BIG), sent_v)
"""


def main():
    import py_compile
    import tempfile
    import os

    for tag, code in (("SLOW", SLOW_SKELETON), ("FAST", FAST_SKELETON)):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        py_compile.compile(path, doraise=True)
        os.unlink(path)
        print(f"{tag} skeleton: py_compile OK")
    print("ASSERT PASS: CONST-arange-scalar-materialize delta skeletons valid")
    print("measured evidence: msprof 46.97 -> 38.41us (-18.2%), see docstring")


if __name__ == "__main__":
    main()
