#!/usr/bin/env python3
"""Generate Round-2 experiment candidates from the frozen Stage 3 baseline.

Never modifies the baseline; writes ``_make_dropout_kernel_round2_*.py`` under
``perf_opt/``.  Each candidate changes exactly one major optimization point.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "_make_dropout_kernel.py")
src = open(SRC).read()

DOC = '''"""Stage 4 candidate {name}: {desc}

Derived from the frozen Stage 3 baseline ``_make_dropout_kernel.py`` (read
only).  Single optimization point: {point}.  Deterministic replay contract
(``h`` stays a pure function of ``(seed, idx)``), the drop-rate contract and
all design constraints are preserved.

run: python {base} --level all
"""'''

ALLOC_OLD = '''                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                xf = T.alloc_ub((block_size,), _F32)
                lane = T.alloc_ub((block_size,), "int32")
                base = T.alloc_ub((block_size,), "int32")
                h = T.alloc_ub((block_size,), "int32")
                t = T.alloc_ub((block_size,), "int32")
                seedb = T.alloc_ub((block_size,), "int32")
                m31 = T.alloc_ub((block_size,), "int32")
                zerob = T.alloc_ub((block_size,), "int32")
                tp_b = T.alloc_ub((block_size,), "int32")
                mf = T.alloc_ub((block_size,), _F32)

                # Loop-invariant constants (filled once, outside the tile loop).
                T.fill(seedb, T.cast(seed, "int32"))
                T.fill(m31, T.cast(_M31, "int32"))
                T.fill(zerob, T.cast(0, "int32"))
                T.fill(tp_b, T.cast(tp_shift, "int32"))
                T.arange(lane, [1], 0)  # lane[i] = i
'''

C1_BUFFERS = '''                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                xf = T.alloc_ub((block_size,), _F32)
                lane = T.alloc_ub((block_size,), "int32")
                base = T.alloc_ub((block_size,), "int32")
                h = T.alloc_ub((block_size,), "int32")
                t = T.alloc_ub((block_size,), "int32")
                seedb = T.alloc_ub((block_size,), "int32")
                m31 = T.alloc_ub((block_size,), "int32")
                zerob = T.alloc_ub((block_size,), "int32")
                tp_b = T.alloc_ub((block_size,), "int32")
                mf = T.alloc_ub((block_size,), _F32)

                T.fill(seedb, T.cast(seed, "int32"))
                T.fill(m31, T.cast(_M31, "int32"))
                T.fill(zerob, T.cast(0, "int32"))
                T.fill(tp_b, T.cast(tp_shift, "int32"))
                T.arange(lane, [1], 0)  # lane[i] = i
'''

C2_BUFFERS = '''                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                xf = T.alloc_ub((block_size,), _F32)
                lane = T.alloc_ub((block_size,), "int32")
                h = T.alloc_ub((block_size,), "int32")
                t = T.alloc_ub((block_size,), "int32")
                mf = T.alloc_ub((block_size,), _F32)

                T.arange(lane, [1], 0)  # lane[i] = i
'''

HASH_OLD = '''                        # ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----
                        T.fill(base, off)
                        T.vadd(lane, base, h)      # h = idx
                        T.vxor(h, seedb, h)        # h = idx ^ seed'''

HASH_C1 = '''                        # ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----
                        T.fill(base, off)
                        T.vadd(lane, base, h)      # h = idx
                        T.vxor(h, seedb, h)        # h = idx ^ seed'''

HASH_C2 = '''                        # ---- PRNG: idx -> hash (right-shift-free, int32 wrap) ----
                        T.vadd(lane, off, h)             # h = idx (off is an i32 scalar)
                        T.vxor(h, seed_i32, h)           # h = idx ^ seed'''

MASK_OLD = '''                        T.vand(h, m31, t)          # t = hm
                        T.vsub(t, tp_b, t)         # d = hm - (256*tp - 1)
                        T.vmax(t, zerob, t)        # d = max(d, 0)
                        T.vmin(t, 1, t)            # m = min(d, 1)'''

MASK_C2 = '''                        T.vand(h, T.cast(_M31, "int32"), t)   # t = hm
                        T.vsub(t, T.cast(tp_shift, "int32"), t)  # d = hm - (256*tp-1)
                        T.vmax(t, T.cast(0, "int32"), t)      # d = max(d, 0)
                        T.vmin(t, 1, t)                       # m = min(d, 1)'''

# ---- C1: static-bulk full tiles + (compile-time) dynamic tail ----
C1_LOOP_OLD = '''                # In-core grid stride; upper bound is a static integer.
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_num:
                        off = task * block_size
                        tail = T.min(block_size, N - off)
'''
C1_LOOP_NEW = '''                # In-core grid stride; upper bound is a static integer.
                # PL-1.34: when ``block_size`` divides ``N`` exactly (all tune
                # workloads), every tile gets a *static* extent copy; the
                # dynamic ``tail`` slice only survives in the non-divisible
                # compile-time branch.
                for i in T.serial(T.ceildiv(n_num, _CORE_NUM)):
                    task = i * _CORE_NUM + cid
                    if task < n_num:
                        off = task * block_size
                        if non_divisible:
                            tail = T.min(block_size, N - off)
'''

C1_LOAD_OLD = '''                        T.copy(x[off:off + tail], x_ub[0:tail])'''
C1_LOAD_NEW = '''                        if non_divisible:
                            T.copy(x[off:off + tail], x_ub[0:tail])
                        else:
                            T.copy(x[off:off + block_size], x_ub)'''

C1_STORE_OLD = '''                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            T.copy(y_ub[0:tail], y[off:off + tail])
                        else:
                            T.copy(xc[0:tail], y[off:off + tail])'''
C1_STORE_NEW = '''                        if need_cast:
                            T.vcast(xc, y_ub, round_mode="rint")
                            if non_divisible:
                                T.copy(y_ub[0:tail], y[off:off + tail])
                            else:
                                T.copy(y_ub, y[off:off + block_size])
                        else:
                            if non_divisible:
                                T.copy(xc[0:tail], y[off:off + tail])
                            else:
                                T.copy(xc, y[off:off + block_size])'''

C1_VARDEF_OLD = '''        need_cast = dtype != _F32
        n_num = T.ceildiv(N, block_size)'''
C1_VARDEF_NEW = '''        need_cast = dtype != _F32
        n_num = T.ceildiv(N, block_size)
        # Compile-time flag: exact-multiple workload -> all-static copy path.
        non_divisible = (N % block_size) != 0'''
C1_VARDEF_NEW = '''        need_cast = dtype != _F32
        n_num = T.ceildiv(N, block_size)
        # Compile-time flag: exact-multiple workload -> all-static copy path.
        non_divisible = (N % block_size) != 0'''

C2_VARDEF_NEW = '''        need_cast = dtype != _F32
        n_num = T.ceildiv(N, block_size)
        seed_i32 = T.cast(seed, "int32")'''


def patch(text, pairs, name):
    for old, new in pairs:
        if old not in text:
            raise SystemExit(f"[{name}] anchor not found:\n{old[:120]}")
        text = text.replace(old, new, 1)
    return text


def _swap_docstring(text, name, desc, point, fname):
    start = text.index('"""')
    end = text.index('"""', start + 3) + 3
    return text[:start] + DOC.format(name=name, desc=desc, point=point,
                                     base=fname).rstrip("\n") + "\n" + text[end:]


def write_candidate(fname, name, desc, point, pairs):
    body = patch(src, pairs, name)
    body = _swap_docstring(body, name, desc, point, fname)
    open(os.path.join(HERE, fname), "w").write(body)
    print("wrote", fname)


# --- Candidate C1: static-bulk + compile-time tail ---
write_candidate(
    "_make_dropout_kernel_round2_staticbulk.py",
    "static-bulk copy (PL-1.34)",
    "static-extent tile copies + compile-time dynamic-tail branch",
    "static bulk copy / tail split",
    [(C1_VARDEF_OLD, C1_VARDEF_NEW), (C1_LOOP_OLD, C1_LOOP_NEW),
     (C1_LOAD_OLD, C1_LOAD_NEW), (C1_STORE_OLD, C1_STORE_NEW)],
)

# --- Candidate C2: UB diet (scalar operands, drop 5 constant buffers) ---
write_candidate(
    "_make_dropout_kernel_round2_ubdiet.py",
    "UB diet (scalar operands)",
    "drop lane/seed/m31/zero/tp constant buffers -> larger block_size headroom",
    "UB footprint reduction via scalar v-op operands",
    [(ALLOC_OLD, C2_BUFFERS), (HASH_OLD, HASH_C2), (MASK_OLD, MASK_C2),
     (C1_VARDEF_OLD, C2_VARDEF_NEW)],
)

# --- Candidate C3: C1 + C2 combined (only after single-point evidence) ---
c1 = open(os.path.join(HERE, "_make_dropout_kernel_round2_staticbulk.py")).read()
body = open(os.path.join(HERE, "_make_dropout_kernel_round2_ubdiet.py")).read()
# take C2's buffer diet + hash + mask into C1's file
for old, new in [(ALLOC_OLD, C2_BUFFERS), (HASH_OLD, HASH_C2), (MASK_OLD, MASK_C2)]:
    if old not in c1:
        raise SystemExit(f"[C3] anchor not found: {old[:60]}")
    c1 = c1.replace(old, new, 1)
c1 = c1.replace(C1_VARDEF_OLD, C2_VARDEF_NEW, 1)
c1 = c1.replace('"""Stage 4 candidate static-bulk copy (PL-1.34):',
                '"""Stage 4 candidate C1+C2 combined:', 1)
open(os.path.join(HERE, "_make_dropout_kernel_round2_sb_ubdiet.py"), "w").write(c1)
print("wrote _make_dropout_kernel_round2_sb_ubdiet.py")
