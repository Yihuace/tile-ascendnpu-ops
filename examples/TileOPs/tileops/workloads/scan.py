"""Workload definitions for the scan op family.

Ported from TileOPs ``workloads/reduction.py`` — the ``CumulativeWorkload``
subset for the ``CumprodFwdOp`` migration unit only (the shared GPU file also
defines the cumulative workload used by ``CumsumFwdOp``, which is not
migrated).

Adaptation point W1: ``device`` is resolved from
:func:`get_device_backend` instead of the hard-coded ``"cuda"`` in the
original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class CumprodFwdWorkload(WorkloadBase):
    """Inputs for cumprod over the last dimension.

    Ported from the GPU ``CumulativeWorkload`` with ``op_kind`` fixed to
    cumprod.  Values default to a narrow band around 1.0 so a long scan stays
    in range; pass ``use_small_range`` explicitly to override.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        use_small_range: bool | None = None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.use_small_range = True if use_small_range is None else use_small_range

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        if self.use_small_range:
            x = torch.rand(*self.shape, dtype=self.dtype, device=device) * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device=device)
        return (x,)
