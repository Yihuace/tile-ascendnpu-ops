# Copyright (c) Huawei Technologies Co., Ltd. 2026.

"""Stage 4 (round 3) latency-breakdown probe for the 1d RoPE kernel.

Purpose: decompose the measured `msprof op Task Duration` of the current best
`perf_opt/_make_rope_neox_1d.py` into

    fixed(launch + tile-loop skeleton) + memory(copy floor) + compute

The probe reuses the *exact* buffer set / tile sizing / DMA forms of the
current best kernel (r2_contig) and only removes work, selected at import time
by the environment variable ``ROPE_PROBE``:

  * ``empty`` : same 40-core grid + same `iters` loop + same guards, no DMA and
                no arithmetic (a single 1-row copy keeps the kernel alive).
                -> fixed term (launch ramp + loop skeleton + wave tail)
  * ``copy``  : contiguous full-row x -> y staging copy only (2 MB read +
                2 MB write for the 8k-d128 bf16 case), no cos/sin, no math.
                -> pure-copy floor for the same tile/grid shape
  * ``mem``   : full traffic (x + cos + sin read, y write) with the same
                6 casts but only 2 arithmetic ops (y1 = a + c, y2 = b + s).
                -> memory floor with cos/sin traffic + cast cost
  * ``full``  : identical to the current best kernel (reference, normally the
                current best file itself is measured instead).

The probe is a measurement instrument only: it is not a deliverable, it is not
precision-checked (run the runner with ``--skip-l0``) and it never records to
``perf_records.jsonl``.

run:
  ROPE_PROBE=copy python _bench_kernel.py --impl _probe_1d_breakdown.py \
      --layout 1d --S 8192 --d 128 --dtype bfloat16
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import functools

import tilelang
import tilelang.language as T

_CORE_NUM = 40
_UB_BUDGET_BYTES = 150 * 1024
_UB_HARD_BYTES = 176 * 1024
_ELEM_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_COMPUTE_DTYPE = "float32"
_DEFAULT_BLOCK_SIZE = {"float32": 1024, "float16": 2048, "bfloat16": 2048}

_PROBE = os.environ.get("ROPE_PROBE", "mem")


@functools.lru_cache(maxsize=8)
def _make_rope_neox_1d(seq_len: int, head_dim: int, dtype: str, block_size: int = 2048,
                       tile_rows: int = None, core_num: int = None) -> object:
    half = head_dim // 2
    f32 = _COMPUTE_DTYPE
    need_cast = dtype != f32
    mode = _PROBE

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(blk: int):
        cores = core_num if core_num is not None else _CORE_NUM
        if tile_rows is not None:
            block_rows = max(1, min(int(tile_rows), seq_len))
        else:
            _bytes_per_row = (half * (4 * _ELEM_BYTES[dtype] + 7 * 4)
                              + 2 * head_dim * _ELEM_BYTES[dtype])
            _rows_max = max(1, _UB_BUDGET_BYTES // _bytes_per_row)
            _rows_hard = max(1, _UB_HARD_BYTES // _bytes_per_row)
            _iters = max(1, -(-seq_len // (_rows_max * cores)))
            block_rows = max(1, min(-(-seq_len // (_iters * cores)), _rows_hard, seq_len))
        n_num = (seq_len + block_rows - 1) // block_rows
        iters = (n_num + cores - 1) // cores

        @T.prim_func
        def main(
            x: T.Tensor((seq_len, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((seq_len, head_dim), dtype),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, _):
                a_ub = T.alloc_ub((block_rows, half), dtype)
                b_ub = T.alloc_ub((block_rows, half), dtype)
                c_ub = T.alloc_ub((block_rows, half), dtype)
                s_ub = T.alloc_ub((block_rows, half), dtype)
                af = T.alloc_ub((block_rows, half), f32)
                bf = T.alloc_ub((block_rows, half), f32)
                cf = T.alloc_ub((block_rows, half), f32)
                sf = T.alloc_ub((block_rows, half), f32)
                y1f = T.alloc_ub((block_rows, half), f32)
                y2f = T.alloc_ub((block_rows, half), f32)
                t = T.alloc_ub((block_rows, half), f32)
                x_stage = T.alloc_ub((block_rows, head_dim), dtype)
                o_stage = T.alloc_ub((block_rows, head_dim), dtype)

                A = af if need_cast else a_ub
                B = bf if need_cast else b_ub
                C = cf if need_cast else c_ub
                S = sf if need_cast else s_ub

                for i in T.serial(iters):
                    task = i * cores + cid
                    if task < n_num:
                        r0 = task * block_rows
                        tr = T.min(block_rows, seq_len - r0)

                        if mode == "tiny":
                            # launch + minimal DMA floor (kernel stays alive)
                            T.copy(x[cid : cid + 1, 0:head_dim],
                                   y[cid : cid + 1, 0:head_dim])
                        elif mode == "tiny_ub":
                            # launch + one UB->UB move (vector/DMA setup cost)
                            T.copy(x[cid : cid + 1, 0:head_dim], x_stage[0:1, 0:head_dim])
                            T.copy(x_stage[0:1, 0:head_dim], y[cid : cid + 1, 0:head_dim])
                        elif mode == "empty":
                            if cid == 0 and i == 0:
                                T.copy(x[0:1, 0:head_dim], y[0:1, 0:head_dim])
                        elif mode == "copy":
                            T.copy(x[r0 : r0 + tr, 0:head_dim],
                                   x_stage[0:tr, 0:head_dim])
                            T.copy(x_stage[0:tr, 0:head_dim],
                                   y[r0 : r0 + tr, 0:head_dim])
                        else:  # "mem" / "full"
                            T.copy(x[r0 : r0 + tr, 0:head_dim],
                                   x_stage[0:tr, 0:head_dim])
                            T.copy(x_stage[0:block_rows, 0:half], a_ub)
                            T.copy(x_stage[0:block_rows, half:head_dim], b_ub)
                            T.copy(cos_table[r0 : r0 + tr, 0:half], c_ub[0:tr, 0:half])
                            T.copy(sin_table[r0 : r0 + tr, 0:half], s_ub[0:tr, 0:half])

                            if need_cast:
                                T.vcast(a_ub, af, round_mode="rint")
                                T.vcast(b_ub, bf, round_mode="rint")
                                T.vcast(c_ub, cf, round_mode="rint")
                                T.vcast(s_ub, sf, round_mode="rint")

                            if mode == "mem":
                                # 2 arithmetic ops instead of the 6-op chain
                                T.vadd(A, C, y1f)
                                T.vadd(B, S, y2f)
                            else:
                                T.vmul(A, C, y1f)
                                T.vmul(B, S, t)
                                T.vsub(y1f, t, y1f)
                                T.vmul(B, C, y2f)
                                T.vmul(A, S, t)
                                T.vadd(y2f, t, y2f)

                            if need_cast:
                                T.vcast(y1f, a_ub, round_mode="rint")
                                T.vcast(y2f, b_ub, round_mode="rint")
                                T.copy(a_ub, o_stage[0:block_rows, 0:half])
                                T.copy(b_ub, o_stage[0:block_rows, half:head_dim])
                                T.copy(o_stage[0:tr, 0:head_dim],
                                       y[r0 : r0 + tr, 0:head_dim])
                            else:
                                T.copy(y1f[0:tr, 0:half], o_stage[0:tr, 0:half])
                                T.copy(y2f[0:tr, 0:half], o_stage[0:tr, half:head_dim])
                                T.copy(o_stage[0:tr, 0:head_dim],
                                       y[r0 : r0 + tr, 0:head_dim])

        return main

    return kernel
