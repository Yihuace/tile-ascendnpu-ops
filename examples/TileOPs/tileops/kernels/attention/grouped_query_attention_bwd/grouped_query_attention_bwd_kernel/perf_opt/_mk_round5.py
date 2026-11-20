"""Generate the round-5 Stage-4 candidate files from the current best.

Each branch applies exactly ONE primary optimization change to the current best
(``perf_opt/_gqa_bwd_kernel.py``, candidate ``r2_fp32vcmp``):

  r5_grid    grid-order: the KV-block index becomes the slowest-varying grid
             dimension, so concurrently running programs own disjoint dq rows
             (removes the atomic contention of the original order).
  r5_atomrow atomic row loop  T.serial(BN)  ->  T.Parallel(BN)  (full-D row).
  r5_atomx4  atomic row loop  ->  T.Parallel(BN, dim//4) + T.atomic_addx4 [1,4]
             (the parallel-issue pattern used by deepseek_v32 sparse_mla_bwd).
"""

import sys

BASE = "_gqa_bwd_kernel.py"

GRID_OLD = """            bz = cid // (heads_kv * n_kv)
            rem = cid % (heads_kv * n_kv)
            kh = rem // n_kv
            jb = rem % n_kv
"""

GRID_NEW = """            # TUNED (r5_grid): KV-block index is the SLOWEST varying grid
            # dimension.  In the original order (jb fastest) all n_kv KV
            # blocks of one (batch, kv-head) land in the same wave of ~40
            # concurrent programs and every one of them atomically adds into
            # the same dq rows -> maximal contention on the dq atomic.  With
            # jb slowest a wave covers distinct (batch, kv-head) pairs, i.e.
            # disjoint dq rows; heavy jb = 0 programs are also scheduled first
            # (LPT), which does not hurt the makespan.
            jb = cid // (batch * heads_kv)
            rem = cid % (batch * heads_kv)
            kh = rem // batch
            bz = rem % batch
"""

ATOM_OLD = """                    for r in T.serial(BN):                                  # cross-KV-block atomic
                        T.atomic_add(dq[bz, q0 + r, h, 0], dq_ub[r, 0], [dim])
"""

ATOM_ROW = """                    # TUNED (r5_atomrow): the row loop is issued in
                    # parallel instead of serially.  The serial form costs
                    # ~100 ns per 512 B row op (issue + wait chain), i.e. the
                    # dominant part of the measured dq-atomic cost.
                    for r in T.Parallel(BN):                                # cross-KV-block atomic
                        T.atomic_add(dq[bz, q0 + r, h, 0], dq_ub[r, 0], [dim])
"""

ATOM_X4 = """                    # TUNED (r5_atomx4): parallel issue with 4-element
                    # atomic units (pattern of
                    # examples/deepseek_v32/sparse_mla_bwd.py).
                    for r, dd in T.Parallel(BN, dim // 4):                  # cross-KV-block atomic
                        T.atomic_addx4(dq[bz, q0 + r, h, dd * 4], dq_ub[r, dd * 4], [1, 4])
"""

VARIANTS = {
    "r5_grid": [(GRID_OLD, GRID_NEW)],
    "r5_atomrow": [(ATOM_OLD, ATOM_ROW)],
    "r5_atomx4": [(ATOM_OLD, ATOM_X4)],
}


def main():
    src = open(BASE).read()
    for name, patches in VARIANTS.items():
        out = src
        for old, new in patches:
            assert out.count(old) == 1, (name, out.count(old))
            out = out.replace(old, new)
        out = out.replace(
            '"""GQA backward main kernel (NPU, target="npuir") -- Stage-4 tuned (perf_opt).',
            f'"""GQA backward main kernel (NPU, target="npuir") -- Stage-4 tuned (perf_opt, {name}).',
            1,
        )
        path = f"_gqa_bwd_kernel_{name}.py"
        open(path, "w").write(out)
        print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
