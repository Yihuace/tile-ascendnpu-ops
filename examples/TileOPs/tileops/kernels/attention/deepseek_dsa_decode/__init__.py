"""DeepSeek sparse attention decode kernel package (sparse MLA, KV cache + top-k)."""

from tileops.kernels.attention.deepseek_dsa_decode.deepseek_dsa_decode import SparseMlaKernel

__all__ = ["SparseMlaKernel"]
