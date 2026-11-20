import tilelang
import tilelang.language as T

from . import _jit_options


@tilelang.jit(**_jit_options())
def build_table_fp32(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    block_tokens: int,
    bn_groups: int,
):
    tokens = batch * seq_len
    half = dim // 2
    half_pad = (half + 7) // 8 * 8
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
        x: T.Tensor((tokens, heads, dim), "float32"),
        cos: T.Tensor((seq_len, half), "float32"),
        sin: T.Tensor((seq_len, half), "float32"),
        out: T.Tensor((tokens, heads, dim), "float32"),
    ):
        with T.Kernel(programs, is_npu=True) as (pid, _):
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
                        cos_tile[:rows, 0, 0:half],
                    )
                    T.copy(
                        sin[seq_start : seq_start + rows, 0:half],
                        sin_tile[:rows, 0, 0:half],
                    )

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
                                x_lo[:rows, 0:1, 0:half],
                            )
                            T.copy(
                                x[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    half:dim,
                                ],
                                x_hi[:rows, 0:1, 0:half],
                            )
                            T.vmul(x_lo, cos_tile, out_lo)
                            T.vmul(x_hi, sin_tile, tmp)
                            T.vsub(out_lo, tmp, out_lo)
                            T.copy(
                                out_lo[:rows, 0:1, 0:half],
                                out[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    0:half,
                                ],
                            )
                            T.vmul(x_hi, cos_tile, x_hi)
                            T.vmul(x_lo, sin_tile, tmp)
                            T.vadd(x_hi, tmp, tmp)
                            T.copy(
                                tmp[:rows, 0:1, 0:half],
                                out[
                                    token_start : token_start + rows,
                                    head_index : head_index + 1,
                                    half:dim,
                                ],
                            )

    return main


__all__ = ["build_table_fp32"]
