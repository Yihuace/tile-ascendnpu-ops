import tilelang
import tilelang.language as T

from . import _jit_options


@tilelang.jit(**_jit_options())
def build_table_low_precision(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    block_tokens: int,
    bn_groups: int,
):
    tokens = batch * seq_len
    half = dim // 2
    half_pad = (half + 15) // 16 * 16
    seq_blocks = (seq_len + block_tokens - 1) // block_tokens
    bn_units = batch * heads
    base_units = bn_units // bn_groups
    extra_groups = bn_units % bn_groups
    max_units = (bn_units + bn_groups - 1) // bn_groups
    work_blocks = seq_blocks * bn_groups
    programs = min(work_blocks, 48)
    iterations = (work_blocks + programs - 1) // programs

    @T.prim_func
    def main(
        x: T.Tensor((tokens, heads, dim), dtype),
        cos: T.Tensor((seq_len, half), dtype),
        sin: T.Tensor((seq_len, half), dtype),
        out: T.Tensor((tokens, heads, dim), dtype),
    ):
        with T.Kernel(programs, is_npu=True) as (pid, _):
            x_lo_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            x_hi_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            out_lo_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            out_hi_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            table_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            x_lo = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            x_hi = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            cos_tile = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            sin_tile = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            out_lo = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            tmp = T.alloc_ub((block_tokens, 1, half_pad), "float32")

            for iteration in T.serial(iterations):
                work = pid + iteration * programs
                with T.If(work < work_blocks), T.Then():
                    seq_block = work // bn_groups
                    bn_group = work % bn_groups
                    seq_start = seq_block * block_tokens
                    rows = T.min(block_tokens, seq_len - seq_start)
                    T.copy(
                        cos[seq_start : seq_start + rows, 0:half],
                        table_io[:rows, 0, 0:half],
                    )
                    T.vcast(table_io, cos_tile, round_mode="round")
                    T.copy(
                        sin[seq_start : seq_start + rows, 0:half],
                        table_io[:rows, 0, 0:half],
                    )
                    T.vcast(table_io, sin_tile, round_mode="round")

                    unit_start = bn_group * base_units + T.min(bn_group, extra_groups)
                    unit_end = (bn_group + 1) * base_units + T.min(
                        bn_group + 1, extra_groups
                    )
                    for local_unit in T.serial(max_units):
                        unit = unit_start + local_unit
                        with T.If(unit < unit_end), T.Then():
                            batch_index = unit // heads
                            head_index = unit % heads
                            token_start = batch_index * seq_len + seq_start
                            T.copy(
                                x[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    0:half,
                                ],
                                x_lo_io[:rows, 0:1, 0:half],
                            )
                            T.vcast(x_lo_io, x_lo, round_mode="round")
                            T.vmul(x_lo, cos_tile, out_lo)
                            T.copy(
                                x[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    half:dim,
                                ],
                                x_hi_io[:rows, 0:1, 0:half],
                            )
                            T.vcast(x_hi_io, x_hi, round_mode="round")
                            T.vmul(x_hi, sin_tile, tmp)
                            T.vsub(out_lo, tmp, out_lo)
                            T.vcast(out_lo, out_lo_io, round_mode="rint")
                            T.copy(
                                out_lo_io[:rows, 0:1, 0:half],
                                out[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    0:half,
                                ],
                            )
                            T.vmul(x_hi, cos_tile, x_hi)
                            T.vmul(x_lo, sin_tile, tmp)
                            T.vadd(x_hi, tmp, x_hi)
                            T.vcast(x_hi, out_hi_io, round_mode="rint")
                            T.copy(
                                out_hi_io[:rows, 0:1, 0:half],
                                out[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    half:dim,
                                ],
                            )

    return main


__all__ = ["build_table_low_precision"]
