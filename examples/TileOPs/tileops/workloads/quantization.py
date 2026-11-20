"""Workload definitions for the quantization op family.

Ported from TileOPs ``workloads/fp8_quant.py`` -- ``FP8QuantWorkload`` only.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"``.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class FP8QuantWorkload(WorkloadBase):
    """Workload definition for FP8QuantOp.

    Generates the single ``[batch, seq_len_kv, kv_group, index_dim]`` input
    tensor via ``randn`` -- the GPU workload's generation logic is preserved
    (unary op, no reshape; the fp8 scale/output dtypes are fixed by the
    kernel).
    """

    def __init__(
        self, batch: int, seq_len_kv: int, kv_group: int, index_dim: int, in_dtype: torch.dtype
    ):
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = in_dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        # W1: device via the device backend (was ``device="cuda"``).
        device = get_device_backend().name
        input_tensor = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            dtype=self.in_dtype,
            device=device,
        )
        return (input_tensor,)
