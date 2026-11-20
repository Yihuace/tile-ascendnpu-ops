"""Workload definitions for the pool op family (NPU-adapted).

Ported from TileOPs ``workloads/pool.py`` -- ``AdaptivePool2dWorkload`` only
(the shared GPU file also defines the 1d/2d/3d avg/max pool bench cases and
the ``AvgPoolWorkload`` / ``MaxPoolWorkload`` / ``MeanPoolingWorkload``
classes, which are not migrated yet).

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase


class MaxPool2dWorkload(WorkloadBase):
    """One NCHW tensor for the windowed 2D max-pool op.

    Ported from TileOPs ``workloads/pool.py:MaxPoolWorkload`` (the 2-D
    instantiation; the shared GPU class also serves the 1d/3d max-pool ops,
    which are not migrated).  ``kernel_size`` / ``stride`` / ``padding`` /
    ``dilation`` / ``ceil_mode`` shape no input; they ride along because the op
    needs them (and because it lets the correctness/bench fixtures share one
    class).  ``stride=None`` means "default to ``kernel_size``", resolved by
    the Op layer.

    Adaptation point W1: ``device`` comes from :func:`get_device_backend`
    instead of the hard-coded ``"cuda"`` in the original TileOPs code.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None,
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            device=get_device_backend().name,
            dtype=self.dtype,
        )
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous view: transpose the last two dims twice so strides
            # differ but shape semantics stay N,C,H,W.
            x = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not x.is_contiguous()
        return (x,)


MaxPool2dFwdOpWorkload = MaxPool2dWorkload




class MaxPool1dWorkload(WorkloadBase):
    """One NCL tensor for the windowed 1D max-pool op.

    Ported from TileOPs ``workloads/pool.py:MaxPool1dBenchCase`` (the 1-D
    instantiation; the shared GPU module also holds the 1d/2d/3d avg/max pool
    bench cases, which are not migrated).  ``kernel_size`` / ``stride`` /
    ``padding`` / ``dilation`` / ``ceil_mode`` shape no input; they ride along
    because the op needs them (and because it lets the correctness/bench
    fixtures share one class).  ``stride=None`` means "default to
    ``kernel_size``", resolved by the Op layer.

    Adaptation point W1: ``device`` comes from :func:`get_device_backend`
    instead of the hard-coded ``"cuda"`` in the original TileOPs code.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None,
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.l_in,
            device=get_device_backend().name,
            dtype=self.dtype,
        )
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous view: transpose the last two dims twice so strides
            # differ but shape semantics stay N,C,L.
            x = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not x.is_contiguous()
        return (x,)


MaxPool1dWithIndicesWorkload = MaxPool1dWorkload


MaxPool1dIndicesFwdOpWorkload = MaxPool1dWithIndicesWorkload


class AdaptivePool2dWorkload(WorkloadBase):
    """One NCHW tensor for the adaptive 2D pool family.

    ``output_size`` shapes no input; it rides along because the op needs it
    (and because it lets the correctness/bench fixtures share one class).

    Adaptation point W1: ``device`` comes from :func:`get_device_backend`
    instead of the hard-coded ``"cuda"`` in the original TileOPs code.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        output_size: int | None | tuple[int | None, int | None],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.output_size = output_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            device=get_device_backend().name,
            dtype=self.dtype,
        ).contiguous()
        return (x,)


AdaptiveMaxPool2dIndicesFwdOpWorkload = AdaptivePool2dWorkload


AdaptiveMaxPool2dFwdOpWorkload = AdaptivePool2dWorkload


AdaptiveAvgPool2dFwdOpWorkload = AdaptivePool2dWorkload
