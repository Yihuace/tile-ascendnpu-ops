"""Workload definitions for the bmm (batched GEMM) op family (NPU).

Ported from TileOPs ``workloads/bmm.py`` -- the ``BmmWorkload`` (BmmFwdOp)
subset only.  The shared GPU file also defines ``BmmFp8Workload`` for
``BmmFp8Op``, which is not migrated.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class BmmWorkload(WorkloadBase):
    """Workload for batched matmul: a=[B,M,K], b=[B,K,N] -> d=[B,M,N]."""

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        a = torch.randn(self.batch, self.m, self.k, device=device, dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device=device, dtype=self.dtype)
        return a, b
