"""FP8LightningIndexerOp (spec-conformant interface, NPU).

Lightning indexer: per-query fp32 logits over an fp8-quantized index cache.
Layout: ``index_q [batch, seq_len, heads, index_dim]``; ``index_k [batch,
seq_len_kv, kv_group, index_dim]``; ``weights [seq_len, heads]`` (fp32);
``cu_seqlen_ks`` / ``cu_seqlen_ke [seq_len]`` (int32); optional ``index_k_scale
[batch, seq_len_kv, kv_group]`` (fp32); output ``logits [batch, seq_len,
seq_len_kv, kv_group]`` (fp32).

The public ``forward`` accepts either bf16 index tensors that are quantized to
fp8 inside the op (``torch_quant_forward``), or pre-quantized fp8 tensors with
an explicit ``index_k_scale`` (``tl_quant_forward``).

Adaptation from GPU (TileOPs) to NPU:

- O1: the ``index_q.is_cuda`` / ``index_k.is_cuda`` device check becomes
  ``backend.is_device_tensor(...)``.
- O3: ``tune`` parameter removed from ``__init__`` / ``_get_kernel`` and from
  the kernel constructor (NPU uses heuristic config selection only).
- O4: ``from tileops.utils import get_sm_version`` replaced by
  ``from tileops.device import get_device_backend``.
- O5: ``compile_boundary.register_instance`` removed (not used on NPU).
- O6: op-specific flow preserved unchanged -- input validation, the two
  quantization forward paths, kernel cache/dispatch, and the fp8 quantization
  helper (``per_custom_dims_cast_to_fp8``) are ported verbatim.
- ``eval_roofline`` is ported from the GPU manifest's
  ``tileops.perf.formulas.fp8_lightning_indexer_roofline`` (the NPU project
  has no ``tileops.perf`` module; the roofline arithmetic is device-agnostic).
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention_indexing.fp8_lightning_indexer import FP8LightningIndexerKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["FP8LightningIndexerOp"]


class FP8LightningIndexerOp(Op):

    def __init__(self,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.batch = None
        self.seq_len = None
        self.heads = None
        self.index_dim = None
        self.seq_len_kv = None
        self.kv_group = None
        self.clean_logits = clean_logits
        self.config = config

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_lightning_indexer_kernel": FP8LightningIndexerKernel}

    @property
    def _config_cache_key(self) -> tuple:
        if not self.config:
            return ()
        return tuple(sorted((key, repr(value)) for key, value in self.config.items()))

    def _get_kernel(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
        device_index: int | None,
    ) -> Kernel:
        # O3: ``tune`` removed from the cache key and the kernel constructor.
        key = (
            batch,
            seq_len,
            heads,
            index_dim,
            seq_len_kv,
            kv_group,
            self.clean_logits,
            self._config_cache_key,
            device_index,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fp8_lightning_indexer_kernel"](
                batch,
                seq_len,
                heads,
                index_dim,
                seq_len_kv,
                kv_group,
                self.clean_logits,
                config=self.config)
        return self._kernel_cache[key]

    def _resolve_and_bind(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        index_k_scale: Optional[torch.Tensor],
    ) -> None:
        # O1: CUDA-specific device check -> backend-agnostic device check.
        backend = get_device_backend()
        if not backend.is_device_tensor(index_q) or not backend.is_device_tensor(index_k):
            raise ValueError(
                f"{type(self).__name__} expects device-resident inputs")
        if index_q.ndim != 4 or index_k.ndim != 4:
            raise ValueError("FP8LightningIndexerOp expects index_q/index_k to be 4D tensors")
        batch, seq_len, heads, index_dim = index_q.shape
        k_batch, seq_len_kv, kv_group, k_dim = index_k.shape
        if k_batch != batch or k_dim != index_dim:
            raise ValueError("index_q and index_k must agree on batch and index_dim")
        if heads % kv_group != 0:
            raise ValueError("heads must be divisible by kv_group")
        if weights.shape != (seq_len, heads):
            raise ValueError("weights must have shape [seq_len, heads]")
        if weights.dtype != torch.float32:
            raise ValueError(f"weights must be float32, got {weights.dtype}")
        if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
            raise ValueError("cu_seqlen_ks/cu_seqlen_ke must have shape [seq_len]")
        if cu_seqlen_ks.dtype != torch.int32 or cu_seqlen_ke.dtype != torch.int32:
            raise ValueError("cu_seqlen_ks and cu_seqlen_ke must be int32")
        if index_k_scale is not None:
            if index_k_scale.shape != (batch, seq_len_kv, kv_group):
                raise ValueError("index_k_scale must have shape [batch, seq_len_kv, kv_group]")
            if index_k_scale.dtype != torch.float32:
                raise ValueError(f"index_k_scale must be float32, got {index_k_scale.dtype}")
            if index_q.dtype != torch.float8_e4m3fn or index_k.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "index_q and index_k must be float8_e4m3fn when index_k_scale is provided"
                )

        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.kernel = self._get_kernel(
            batch, seq_len, heads, index_dim, seq_len_kv, kv_group, index_q.device.index)

    def torch_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                            weights: torch.Tensor, cu_seqlen_ks: torch.Tensor,
                            cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        # NPU adaptation (capability gap CG-2026-0022): npuir / torch_npu expose
        # no device-side fp8 cast path, so a bf16 index carrier (the manifest
        # bf16 workloads) is passed through unchanged with an identity
        # (all-ones) fp32 de-quantization scale -- the kernel's ``has_scale``
        # identity-scale semantics.  The kernel accumulates in fp32, so passing
        # the unquantized carrier reproduces the unquantized reference; skipping
        # the quantization can only remove quantization error.
        if index_q.dtype != torch.float8_e4m3fn:
            index_k_scale = torch.ones((self.batch, self.seq_len_kv, self.kv_group),
                                       device=index_q.device, dtype=torch.float32)
            return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks,
                               cu_seqlen_ke)

        # fp8 index carrier: device-side fp8 quantization (restricted on NPU by
        # CG-2026-0022); kept for interface parity with the source op.
        index_q = index_q.to(torch.float8_e4m3fn)
        index_k, index_k_scale = self.per_custom_dims_cast_to_fp8(index_k, (0,), False)

        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def tl_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                         index_k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def forward(self,
                index_q: torch.Tensor,
                index_k: torch.Tensor,
                weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor,
                cu_seqlen_ke: torch.Tensor,
                index_k_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._resolve_and_bind(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale)
        if index_k_scale is None:
            return self.torch_quant_forward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)
        return self.tl_quant_forward(index_q, index_k, index_k_scale, weights, cu_seqlen_ks,
                                     cu_seqlen_ke)

    def per_custom_dims_cast_to_fp8(self, x: torch.Tensor, dims: Tuple[int],
                                    use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        x_absmax = x.to(torch.float32).abs().amax(dim=-1, keepdim=True).clamp(1e-4)
        sf = x_absmax / 448.0
        if use_ue8m0:
            assert sf.view(-1).amax().item() > 0
            sf = torch.pow(2.0, torch.ceil(torch.log2(x_absmax)))
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze(-1)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the configured workload shape.

        Ported verbatim (device-agnostic arithmetic) from the GPU repo's
        ``tileops.perf.formulas.fp8_lightning_indexer_roofline``: 2 FLOPs per
        score element per index dim, ``(q + k)`` read into the index-cache
        quantization, the int32 window metadata, and the fp32
        weights/logits/scales traffic.  The op does not retain the observed
        input dtype, so the GPU formula's conservative bf16 (2-byte) default is
        used for the index tensors.
        """
        batch, seq_len, heads, index_dim = self.batch, self.seq_len, self.heads, self.index_dim
        seq_len_kv, kv_group = self.seq_len_kv, self.kv_group
        scores = batch * seq_len * seq_len_kv * kv_group
        q_elems = batch * seq_len * heads * index_dim
        k_elems = batch * seq_len_kv * kv_group * index_dim
        weights = seq_len * heads
        # Conservative bf16 default (GPU formula: op has no retained input dtype).
        index_elem = torch.bfloat16.itemsize
        flops = 2 * scores * index_dim
        nbytes = (q_elems + k_elems) * index_elem + batch * seq_len_kv * kv_group * 4
        nbytes += weights * 4
        nbytes += 2 * seq_len * 4 + scores * 4
        return int(flops), int(nbytes)
