"""GQAPrefillPagedWithKVCacheFwdOp -- GQA prefill over a paged KV cache (NPU).

Adapted from the GPU TileOPs op.  The GPU kernel is a fused Cube+Vector attention
that cannot be compiled on this toolchain (see the kernel wrapper docstring and
``kernels/attention/gqa_fwd/TERMINAL_STATE.md``), so the NPU port uses three
single-engine kernels with the paged gather, dequant, head repeat and causal mask
done on the host.

Signature follows the GPU manifest:

    q                (total_q, H, D)
    k_new, v_new     (total_q, H_kv, D)
    k_pages, v_pages (physical_tokens, H_kv, D)
    k_scale, v_scale (1,)
    cu_seqlens_q     (B + 1,)
    cache_seqlens    (B,)
    block_table      (B, max_pages_per_req)
    -> o             (total_q, H, D)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention.gqa_fwd.gqa_fwd import GQAPrefillPagedKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["GQAPrefillPagedWithKVCacheFwdOp"]


class GQAPrefillPagedWithKVCacheFwdOp(Op):
    """Grouped-query attention prefill reading K/V from a paged cache."""

    _op_name = "gqa_fwd"
    kernel_cls = GQAPrefillPagedKernel

    def __init__(
        self,
        page_size: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.page_size = page_size
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            page_size=page_size, is_causal=is_causal, sm_scale=sm_scale
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    def _validate(self, q, k_new, v_new, k_pages, v_pages, cu_seqlens_q,
                  cache_seqlens, block_table) -> None:
        backend = get_device_backend()
        for name, t in (("q", q), ("k_new", k_new), ("v_new", v_new),
                        ("k_pages", k_pages), ("v_pages", v_pages)):
            if not backend.is_device_tensor(t):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got {t.device}")
        if q.dim() != 3 or k_new.dim() != 3:
            raise ValueError("q and k_new must be 3-D (tokens, heads, dim)")
        H, D = q.shape[1], q.shape[2]
        if k_new.shape[2] != D:
            raise ValueError(f"head dim mismatch: q {D} vs k_new {k_new.shape[2]}")
        if H % k_new.shape[1]:
            raise ValueError(f"H={H} must be a multiple of H_kv={k_new.shape[1]}")
        B = cu_seqlens_q.numel() - 1
        if cache_seqlens.numel() != B or block_table.shape[0] != B:
            raise ValueError("cu_seqlens_q / cache_seqlens / block_table disagree on B")

    def forward(self, q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
                cu_seqlens_q, cache_seqlens, block_table, *args: Any,
                **kwargs: Any) -> torch.Tensor:
        self._validate(q, k_new, v_new, k_pages, v_pages, cu_seqlens_q,
                       cache_seqlens, block_table)
        out = self.kernel(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale,
                          cu_seqlens_q, cache_seqlens, block_table)

        # Record what this call actually touched.  The benchmark harness asks for
        # the roofline AFTER the op has run (benchmark_base.py:545 calls
        # `self._op.eval_roofline()`), and this op is constructed without shapes,
        # so the sizes have to be captured here.
        cu = cu_seqlens_q.tolist()
        cache = cache_seqlens.tolist()
        pairs = 0
        for b in range(len(cache)):
            q_len = int(cu[b + 1]) - int(cu[b])
            pairs += q_len * (int(cache[b]) + q_len)
        self._last = {
            "total_q": int(q.shape[0]),
            "H": int(q.shape[1]),
            "D": int(q.shape[2]),
            "H_kv": int(k_new.shape[1]),
            "cache_total": int(sum(int(c) for c in cache)),
            "pairs": pairs,
            "elem_bytes": int(q.element_size()),
        }
        return out

    @property
    def total_memory(self) -> float:
        """External traffic: q, the new K/V, the referenced cache pages, and o.

        Same convention as the other ops (e.g. ``AbsFwdOp`` counts one read plus
        one write): only traffic that crosses the device boundary, not the
        score-matrix intermediates the decomposed kernels exchange through GM.
        """
        c = getattr(self, "_last", None)
        if not c:
            return 0.0
        Q, H, D = c["total_q"], c["H"], c["D"]
        Hkv, C, eb = c["H_kv"], c["cache_total"], c["elem_bytes"]
        q_read = Q * H * D * eb                 # q
        o_write = Q * H * D * eb                 # o
        kv_new = 2 * Q * Hkv * D * eb            # k_new + v_new
        kv_cache = 2 * C * Hkv * D * eb          # referenced cache pages, K and V
        return float(q_read + o_write + kv_new + kv_cache)

    def eval_roofline(self) -> tuple[int, int]:
        """``(flops, bytes)``.

        Two matmuls over every (query, key) pair -- ``S = Q @ K^T`` and
        ``O = P @ V`` -- each costing ``2 * M * N * K`` per head.  The pair count is
        summed per request, so unequal ``q_lens``/``cache_lens`` are handled
        exactly instead of being approximated by totals.
        """
        c = getattr(self, "_last", None)
        if not c:
            return 0, 0
        flops = 2 * c["H"] * c["pairs"] * c["D"]
        return int(flops), int(self.total_memory)
