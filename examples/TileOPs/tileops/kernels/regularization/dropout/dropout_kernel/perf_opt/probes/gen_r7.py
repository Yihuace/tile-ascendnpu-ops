#!/usr/bin/env python3
"""Generate R7 candidate kernels from the current perf_opt best (single-point diffs).

Each patch is a textual, reviewable diff of ``_make_dropout_kernel.py``; every
generated file keeps the same 11 vector-op chain unless a patch explicitly
changes it (probes only, never adopted).

usage: python probes/gen_r7.py
"""
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "_make_dropout_kernel.py")
SRC = open(BASE).read()


def write(name, src, header):
    path = os.path.join(HERE, name)
    open(path, "w").write(f"# R7 candidate: {header}\n" + src)
    print("wrote", name)


# ---------------------------------------------------------------- static tail
def patch_static_tail(src):
    assert "tail = T.min(block_size, N - off)" in src
    src = src.replace(
        "                        tail = T.min(block_size, N - off)",
        "                        # R7: N % block_size == 0 -> static extent\n"
        "                        tail = block_size if static_tail else T.min(\n"
        "                            block_size, N - off)",
    )
    old = "        n_num = T.ceildiv(N, block_size)"
    assert old in src
    src = src.replace(old, old + "\n        static_tail = (N % block_size == 0)")
    return src


# --------------------------------------------------------------- pipelining
def patch_pipe(src):
    old = "                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):"
    assert old in src
    return src.replace(
        old, "                for i in T.Pipelined(T.ceildiv(n_num, _CORE_NUM), num_stages=2):")


# ---------------------------------------------------------- int16 mask cast
def patch_i16(src):
    old = "                md = T.alloc_ub((md_extent,), dtype)"
    assert old in src
    src = src.replace(old, old + "\n                m16 = T.alloc_ub((block_size,), \"int16\")")
    # signed branch: mask cast -> m16
    old = "                            T.vcast(h, mf, round_mode=\"rint\")"
    assert old in src
    src = src.replace(old, "                            T.vcast(h, m16, round_mode=\"rint\")")
    # non-signed branch keeps mf
    old = "                            T.vcast(t, mf, round_mode=\"rint\")"
    assert old in src
    old_md = "                            T.vcast(mf, md, round_mode=\"rint\")"
    assert old_md in src
    src = src.replace(old_md, "                            T.vcast(m16, md, round_mode=\"rint\")")
    return src


# ------------------------------------------------------------ block size map
def patch_bs(src, f32, f16, bf16):
    old = '_DEFAULT_BLOCK_SIZE = {"float32": 8192, "float16": 8192, "bfloat16": 8192}'
    assert old in src
    return src.replace(
        old, f'_DEFAULT_BLOCK_SIZE = {{"float32": {f32}, "float16": {f16}, '
             f'"bfloat16": {bf16}}}')


# ------------------------------------------------------------------- probes
PROBE_OPS = {
    "p_fill": "                        T.fill(h, off + seed)      # counter = idx + seed\n",
    "p_vadd": "                        T.vadd(lane, h, h)         # h = counter\n",
    "p_mulA": "                        T.vmul(h, _A_I32, h)\n",
    "p_shl": "                        T.vshl(h, 13, t)\n",
    "p_xor": "                        T.vxor(h, t, h)\n",
    "p_mulB": "                        T.vmul(h, _B_I32, h)\n",
    "p_clamp": "                            T.vclamp(h, h, 0, 1)   # m = 1 iff h >= 1\n",
    "p_cast1": "                            T.vcast(h, mf, round_mode=\"rint\")\n",
    "p_cast2": "                            T.vcast(mf, md, round_mode=\"rint\")\n",
    "p_mulscale": "                            T.vmul(md, T.cast(scale, dtype), md)  # scale * mask\n",
    "p_mulx": "                            T.vmul(x_ub, md, x_ub)                # x * (scale*m)\n",
}


def main():
    # correctness-preserving candidates
    write("_r7_static.py", patch_static_tail(SRC), "static tile extent (N % bs == 0)")
    write("_r7_pipe.py", patch_pipe(SRC), "T.Pipelined(num_stages=2) tile loop")
    write("_r7_i16.py", patch_i16(SRC), "mask cast i32->i16->f16 (same 2 casts)")
    write("_r7_bs8080.py", patch_bs(SRC, 8080, 8080, 8192),
          "wave-exact block_size 8080 for fp32/fp16")
    write("_r7_static_i16.py", patch_i16(patch_static_tail(SRC)),
          "static extent + i16 mask cast")
    write("_r7_static8080.py", patch_bs(patch_static_tail(SRC), 8080, 8080, 8192),
          "wave-exact bs 8080 (+static extent where divisible)")
    # probes: one vector op deleted each (timing only, wrong results)
    for name, line in PROBE_OPS.items():
        assert line in SRC, name
        write(f"probes/{name}.py", SRC.replace(line, ""), f"probe: deleted {name[2:]}")


if __name__ == "__main__":
    main()
