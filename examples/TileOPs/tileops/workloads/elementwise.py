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


class ClampMinWorkload(ShapedRandnWorkload):
    """Workload definition for ClampMinFwdOp (shape + dtype, randn input).

    ``torch.clamp_min`` is the scalar-min specialization of ``torch.clamp``;
    the lower bound ``min`` is an op parameter, not a tensor, so the workload
    only generates the single input tensor.
    """


class BitwiseNotWorkload:
    """Workload definition for BitwiseNotFwdOp (bool / integer input).

    Ported from TileOPs ``workloads/elementwise.py`` (``BitwiseNotWorkload``).
    ``torch.bitwise_not`` operates on bool / integer element types, so this
    generator draws a bool tensor for the bool dtype, a full-range ``uint8``
    tensor for uint8, and a small symmetric-range tensor otherwise.  The op is
    exact (no tolerance), so any valid input exercises the all-ones complement.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        if self.dtype == torch.bool:
            x = torch.rand(self.shape, device=device) > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, self.shape, device=device, dtype=self.dtype)
        else:
            lo = max(torch.iinfo(self.dtype).min, -128)
            hi = min(torch.iinfo(self.dtype).max, 128)
            x = torch.randint(lo, hi, self.shape, device=device, dtype=self.dtype)
        return (x,)


class IsnanWorkload:
    """Workload definition for IsnanFwdOp (float NaN predicate).

    Ported from TileOPs ``workloads/elementwise.py`` (``_draw_special``: a
    quarter each of NaN, +Inf, -Inf and finite values, so the predicate sees
    both truth values).  The op is exact (bool output), so any such input
    exercises the predicate.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.shape, device=device, dtype=self.dtype)
        flat = x.view(-1)
        quarter = flat.numel() // 4
        flat[:quarter] = float("nan")
        flat[quarter : 2 * quarter] = float("inf")
        flat[2 * quarter : 3 * quarter] = float("-inf")
        return (x,)


class IsinfWorkload:
    """Workload definition for IsinfFwdOp (float infinity predicate).

    Ported from TileOPs ``workloads/elementwise.py`` (``_draw_special``: a
    quarter each of NaN, +Inf, -Inf and finite values, so the predicate sees
    both truth values).  The NaN quarter is the discriminant for the ordered
    ``==`` ``isinf`` body: ``isinf(NaN)`` must answer false, so a NaN region
    that wrongly reads true is caught.  The op is exact (bool output), so any
    such input exercises the predicate.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.shape, device=device, dtype=self.dtype)
        flat = x.view(-1)
        quarter = flat.numel() // 4
        flat[:quarter] = float("nan")
        flat[quarter : 2 * quarter] = float("inf")
        flat[2 * quarter : 3 * quarter] = float("-inf")
        return (x,)


class IsfiniteWorkload:
    """Workload definition for IsfiniteFwdOp (float finiteness predicate).

    Ported from TileOPs ``workloads/elementwise.py`` (``_draw_special``: a
    quarter each of NaN, +Inf, -Inf and finite values).  All three special
    quarters must answer False and the finite quarter True, so this draw
    exercises both predicate outcomes and the NaN / +-Inf / -Inf discrimination
    of the ordered ``|x| < +inf`` body.  The op is exact (bool output), so any
    such input exercises the predicate.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = tuple(shape)
        self.input_shape = self.shape
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.shape, device=device, dtype=self.dtype)
        flat = x.view(-1)
        quarter = flat.numel() // 4
        flat[:quarter] = float("nan")
        flat[quarter : 2 * quarter] = float("inf")
        flat[2 * quarter : 3 * quarter] = float("-inf")
        return (x,)


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


class LerpScalarWorkload:
    """Workload definition for LerpFwdOp (scalar-weight torch.lerp).

    The scalar overload ``torch.lerp(input, end, weight: float)`` takes two
    tensor operands (``input`` / ``end``) that broadcast together plus a scalar
    ``weight`` op parameter, so the workload generates the two input tensors
    only.  Both are drawn with ``randn``; the Op layer performs the
    ``broadcast_shapes(input, end)`` broadcast and flattens the operands before
    kernel dispatch.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        end_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = tuple(input_shape)
        self.end_shape = tuple(end_shape)
        self.shape = self.input_shape
        self.out_shape = tuple(torch.broadcast_shapes(self.input_shape, self.end_shape))
        self.n_total = prod(self.out_shape) if self.out_shape else 1
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        a = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        b = torch.randn(self.end_shape, device=device, dtype=self.dtype)
        return a, b


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


class GeluAndMulWorkload:
    """Workload definition for GeluAndMulFwdOp (fused gated exact GELU).

    Ported from TileOPs ``workloads/elementwise.py`` (``GatedRandnWorkload``:
    one ``(M, 2 * N)`` randn tensor carrying the gate and value halves of a
    fused gated op).  ``M`` / ``N`` are the manifest workload keys; ``x`` has
    ``2 * N`` columns, so the output has ``M * N`` elements.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = int(M)
        self.N = int(N)
        self.m = self.M
        self.n = self.N
        self.dtype = dtype
        self.x_shape = (self.M, 2 * self.N)
        self.shape = self.x_shape
        self.n_total = self.M * self.N

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.M, 2 * self.N, device=device, dtype=self.dtype)
        return (x,)


class GeluTanhAndMulWorkload:
    """Workload definition for GeluTanhAndMulFwdOp (fused gated tanh-GELU).

    Ported from TileOPs ``workloads/elementwise.py`` (``GatedRandnWorkload``:
    one ``(M, 2 * N)`` randn tensor carrying the gate and value halves of a
    fused gated op).  ``M`` / ``N`` are the manifest workload keys; ``x`` has
    ``2 * N`` columns, so the output has ``M * N`` elements.  Only the
    activation differs from :class:`GeluAndMulWorkload` (tanh vs erf).

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = int(M)
        self.N = int(N)
        self.m = self.M
        self.n = self.N
        self.dtype = dtype
        self.x_shape = (self.M, 2 * self.N)
        self.shape = self.x_shape
        self.n_total = self.M * self.N

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.M, 2 * self.N, device=device, dtype=self.dtype)
        return (x,)


class SubWorkload:
    """Workload definition for SubFwdOp (scalar-alpha binary sub with broadcast).

    Ported from TileOPs ``workloads/elementwise.py`` (``RandnPairWorkload`` /
    ``BinarySameShapeCase``).  Both operands are drawn with ``randn``; the Op
    layer performs the ``broadcast_shapes(input, other)`` broadcast and flattens
    the operands before kernel dispatch.  ``alpha`` is an op parameter, not a
    tensor, so the workload only generates the two input tensors.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        other_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
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
