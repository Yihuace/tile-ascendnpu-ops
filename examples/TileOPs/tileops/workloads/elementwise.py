"""Workload definitions for the elementwise op family."""

from math import prod

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import RandnWorkload


class ShapedRandnWorkload(RandnWorkload):
    """One ``randn`` tensor of arbitrary rank, with its element count.

    Adaptation point: ``device`` is resolved from :func:`get_device_backend`
    via ``RandnWorkload.gen_inputs`` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple, dtype):
        super().__init__(tuple(shape), dtype)
        self.n_total = prod(self.shape)


class MishWorkload(ShapedRandnWorkload):
    """Workload definition for MishFwdOp (shape + dtype, randn input)."""


class HardsigmoidWorkload(ShapedRandnWorkload):
    """Workload definition for HardsigmoidFwdOp (shape + dtype, randn input)."""


class HardswishWorkload(ShapedRandnWorkload):
    """Workload definition for HardswishFwdOp (shape + dtype, randn input)."""


class ClampScalarWorkload(ShapedRandnWorkload):
    """Workload definition for ClampScalarFwdOp (shape + dtype, randn input).

    Scalar bounds (``min`` / ``max``) are op parameters, not tensors, so the
    workload only generates the single input tensor; the bound values travel
    through the manifest workload rows into the Op constructor.
    """


class LerpTensorManifestWorkload:
    """Workload definition for LerpTensorFwdOp (Tensor-weight torch.lerp).

    Ported from TileOPs ``workloads/elementwise.py``.  All three operands
    (``input`` / ``end`` / ``weight``) share the manifest ``input_shape``;
    the Op layer performs the 3-way broadcast.  ``weight`` uses ``rand``
    (interpolation weights are naturally in [0, 1]) while ``input`` /
    ``end`` use ``randn``.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = shape
        self.end_shape = shape
        self.weight_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        end = torch.randn(self.end_shape, device=device, dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device=device, dtype=self.dtype)
        return x, end, weight


class AddWorkload:
    """Workload definition for AddFwdOp (scalar-alpha binary add with broadcast).

    Ported from TileOPs ``workloads/elementwise.py`` (``AddBroadcastWorkload``
    and ``AddSameShapeCase``).  Both operands are drawn with ``randn``; the Op
    layer performs the ``broadcast_shapes(input, other)`` broadcast and flattens
    the operands before kernel dispatch.  ``alpha`` is an op parameter, not a
    tensor, so the workload only generates the two input tensors.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, input_shape: tuple[int, ...], other_shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = tuple(input_shape)
        self.other_shape = tuple(other_shape)
        self.shape = self.input_shape
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.other_shape))
        self.n_total = prod(self.out_shape) if self.out_shape else 1
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        a = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        b = torch.randn(self.other_shape, device=device, dtype=self.dtype)
        return a, b
