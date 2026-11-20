import tilelang
import tilelang.language as T

from . import _jit_options


@tilelang.jit(**_jit_options())
def build_blocked_low_precision(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    block_tokens: int,
    block_heads: int,
    partition_by_seq: bool,
    merge_copyout: bool,
):
    tokens = batch * seq_len
    half = dim // 2
    half_pad = (half + 15) // 16 * 16
    head_blocks = heads // block_heads
    seq_blocks = (seq_len + block_tokens - 1) // block_tokens
    if partition_by_seq:
        work_blocks = batch * head_blocks * seq_blocks
    else:
        token_blocks = (tokens + block_tokens - 1) // block_tokens
        work_blocks = token_blocks * head_blocks
    has_tail = (seq_len if partition_by_seq else tokens) % block_tokens != 0
    programs = min(work_blocks, 48)
    iterations = (work_blocks + programs - 1) // programs
    out_width = dim if merge_copyout else 1

    @T.prim_func
    def main(
        x: T.Tensor((tokens, heads, dim), dtype),
        cos: T.Tensor((seq_len, half), dtype),
        sin: T.Tensor((seq_len, half), dtype),
        out: T.Tensor((tokens, heads, dim), dtype),
    ):
        with T.Kernel(programs, is_npu=True) as (pid, _):
            x_lo_io = T.alloc_ub((block_tokens, block_heads, half_pad), dtype)
            x_hi_io = T.alloc_ub((block_tokens, block_heads, half_pad), dtype)
            out_lo_io = T.alloc_ub((block_tokens, block_heads, half_pad), dtype)
            out_hi_io = T.alloc_ub((block_tokens, block_heads, half_pad), dtype)
            out_io = T.alloc_ub((block_tokens, block_heads, out_width), dtype)
            cos_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            sin_io = T.alloc_ub((block_tokens, 1, half_pad), dtype)
            x_lo = T.alloc_ub((block_tokens, block_heads, half_pad), "float32")
            x_hi = T.alloc_ub((block_tokens, block_heads, half_pad), "float32")
            cos_tile = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            sin_tile = T.alloc_ub((block_tokens, 1, half_pad), "float32")
            out_lo = T.alloc_ub((block_tokens, block_heads, half_pad), "float32")
            tmp = T.alloc_ub((block_tokens, block_heads, half_pad), "float32")

            for iteration in T.serial(iterations):
                work = pid + iteration * programs
                with T.If(work < work_blocks), T.Then():
                    seq_block = work % seq_blocks
                    group = work // seq_blocks
                    token_block = work // head_blocks
                    flat_token_start = token_block * block_tokens
                    head_block = (
                        group % head_blocks if partition_by_seq else work % head_blocks
                    )
                    seq_start = (
                        seq_block * block_tokens
                        if partition_by_seq
                        else flat_token_start % seq_len
                    )
                    token_start = (
                        group // head_blocks * seq_len + seq_start
                        if partition_by_seq
                        else flat_token_start
                    )
                    row_size = (
                        T.min(block_tokens, seq_len - seq_start)
                        if partition_by_seq
                        else (
                            T.min(block_tokens, tokens - token_start)
                            if has_tail
                            else block_tokens
                        )
                    )
                    head_start = head_block * block_heads

                    T.copy(
                        cos[seq_start : seq_start + row_size, 0:half],
                        cos_io[:row_size, 0, 0:half],
                    )
                    T.vcast(cos_io, cos_tile, round_mode="round")
                    T.copy(
                        x[
                            token_start : token_start + row_size,
                            head_start : head_start + block_heads,
                            0:half,
                        ],
                        x_lo_io[:row_size, :, 0:half],
                    )
                    T.vcast(x_lo_io, x_lo, round_mode="round")
                    T.vmul(x_lo, cos_tile, out_lo)
                    T.copy(
                        x[
                            token_start : token_start + row_size,
                            head_start : head_start + block_heads,
                            half:dim,
                        ],
                        x_hi_io[:row_size, :, 0:half],
                    )
                    T.vcast(x_hi_io, x_hi, round_mode="round")
                    T.copy(
                        sin[seq_start : seq_start + row_size, 0:half],
                        sin_io[:row_size, 0, 0:half],
                    )
                    T.vcast(sin_io, sin_tile, round_mode="round")
                    T.vmul(x_hi, sin_tile, tmp)
                    T.vsub(out_lo, tmp, out_lo)
                    T.vcast(out_lo, out_lo_io, round_mode="rint")
                    if merge_copyout:
                        T.copy(
                            out_lo_io[:row_size, :, 0:half],
                            out_io[:row_size, :, 0:half],
                        )
                    else:
                        T.copy(
                            out_lo_io[:row_size, :, 0:half],
                            out[
                                token_start : token_start + row_size,
                                head_start : head_start + block_heads,
                                0:half,
                            ],
                        )
                    T.vmul(x_lo, sin_tile, tmp)
                    T.vmul(x_hi, cos_tile, x_hi)
                    T.vadd(x_hi, tmp, x_hi)
                    T.vcast(x_hi, out_hi_io, round_mode="rint")
                    if merge_copyout:
                        T.copy(
                            out_hi_io[:row_size, :, 0:half],
                            out_io[:row_size, :, half:dim],
                        )
                        T.copy(
                            out_io[:row_size, :, 0:dim],
                            out[
                                token_start : token_start + row_size,
                                head_start : head_start + block_heads,
                                0:dim,
                            ],
                        )
                    else:
                        T.copy(
                            out_hi_io[:row_size, :, 0:half],
                            out[
                                token_start : token_start + row_size,
                                head_start : head_start + block_heads,
                                half:dim,
                            ],
                        )

    return main
