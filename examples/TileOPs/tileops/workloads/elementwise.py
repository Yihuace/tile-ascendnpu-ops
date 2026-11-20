"""Workload definitions for the elementwise op family."""

from math import prod
from typing import Optional

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import RandnWorkload, WorkloadBase


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

    def __init__(
        self, input_shape: tuple[int, ...], other_shape: tuple[int, ...], dtype: torch.dtype
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


class AbsWorkload(ShapedRandnWorkload):
    """Workload definition for AbsFwdOp (shape + dtype, randn input).

    randn spans both signs, so every element exercises the sign branch of
    ``|x|`` — a non-negative-only generator would make the op trivially
    identity and mask sign-handling bugs.
    """


class CeilWorkload(ShapedRandnWorkload):
    """Workload definition for CeilFwdOp (shape + dtype, randn input).

    randn keeps values well inside the int32 round-trip range used by the NPU
    kernel's ceil implementation.
    """


class FloorWorkload(ShapedRandnWorkload):
    """Workload definition for FloorFwdOp (shape + dtype, randn input).

    See CeilWorkload for the int32 round-trip range note.
    """


class ExpWorkload(ShapedRandnWorkload):
    """Workload definition for ExpFwdOp (shape + dtype, randn input).

    Input magnitudes stay moderate so ``exp`` remains inside fp16 range
    (exp(3*sigma) ~ 8.1e3 vs fp16 max 6.55e4).
    """


class CosWorkload(ShapedRandnWorkload):
    """Workload definition for CosFwdOp (shape + dtype, randn input)."""


class ClampScalarWorkload(ShapedRandnWorkload):
    """Workload definition for ClampScalarFwdOp (shape + dtype, randn input).

    Scalar bounds (``min`` / ``max``) are op parameters, not tensors, so the
    workload only generates the single input tensor; the bound values travel
    through the manifest workload rows into the Op constructor.
    """


class HardsigmoidWorkload(ShapedRandnWorkload):
    """Workload definition for HardsigmoidFwdOp (shape + dtype, randn input)."""


class HardswishWorkload(ShapedRandnWorkload):
    """Workload definition for HardswishFwdOp (shape + dtype, randn input)."""


class ClampMinWorkload(ShapedRandnWorkload):
    """Workload definition for ClampMinFwdOp (shape + dtype, randn input).

    ``torch.clamp_min`` is the scalar-min specialization of ``torch.clamp``;
    the lower bound ``min`` is an op parameter, not a tensor, so the workload
    only generates the single input tensor.
    """


class BitwiseNotWorkload(WorkloadBase):
    """Workload definition for BitwiseNotFwdOp (bool/integer input).

    Ported from TileOPs ``workloads/elementwise.py`` (``BitwiseNotWorkload``).
    Preserves the op's input-generation logic: a bool tensor via ``rand > 0.5``,
    a full-range ``uint8`` tensor, and small-range signed integers for the
    remaining integer dtypes.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of the hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = tuple(shape)
        self.input_shape = tuple(shape)
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        device = get_device_backend().name
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device=device) > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device=device, dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device=device, dtype=self.dtype)
        return (x,)


class Expm1Workload(ShapedRandnWorkload):
    """Workload definition for Expm1FwdOp (shape + dtype, randn input).

    Ported from the GPU ``RandnFlatWorkload`` in ``workloads/elementwise.py``:
    a single ``randn`` vector of ``n_total`` elements, matching the GPU
    ``test_unary_math.py`` ``_randn`` generator and the ``draw_normal`` bench
    generator.  Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` via ``RandnWorkload.gen_inputs`` instead of
    the hard-coded ``"cuda"``.
    """


class ErfWorkload(ShapedRandnWorkload):
    """Workload definition for ErfFwdOp (shape + dtype, randn input).

    Ported from the GPU ``RandnFlatWorkload`` in ``workloads/elementwise.py``:
    a single ``randn`` vector of ``n_total`` elements, matching the GPU
    ``test_unary_math.py`` ``_randn`` generator and the ``draw_normal`` bench
    generator.  Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` via ``RandnWorkload.gen_inputs`` instead of
    the hard-coded ``"cuda"``.
    """


class EluWorkload(ShapedRandnWorkload):
    """Workload definition for EluFwdOp (shape + dtype, randn input).

    Ported from the GPU elementwise activation workloads: a single ``randn``
    tensor of the manifest ``input_shape``, matching the GPU
    ``test_activation.py`` ``_randn`` generator and the
    ``bench_elementwise_manifest.py`` ``ShapedRandnWorkload`` draw.
    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` via ``RandnWorkload.gen_inputs`` instead of
    the hard-coded ``"cuda"``.
    """


class HardtanhWorkload(ShapedRandnWorkload):
    """Workload definition for HardtanhFwdOp (shape + dtype, randn input).

    Ported from the GPU elementwise activation workloads: a single ``randn``
    tensor of the manifest ``input_shape``, matching the GPU
    ``test_activation.py`` ``_randn`` generator and the
    ``bench_elementwise_manifest.py`` ``ShapedRandnWorkload`` draw.
    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` via ``RandnWorkload.gen_inputs`` instead of
    the hard-coded ``"cuda"``.
    """


class LeakyReluWorkload(ShapedRandnWorkload):
    """Workload definition for LeakyReluFwdOp (shape + dtype, randn input).

    Ported from the GPU elementwise activation workloads: a single ``randn``
    tensor of the manifest ``input_shape``, matching the GPU
    ``test_activation.py`` ``_randn`` generator and the
    ``bench_elementwise_manifest.py`` ``ShapedRandnWorkload`` draw.
    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` via ``RandnWorkload.gen_inputs`` instead of
    the hard-coded ``"cuda"``.
    """


class TensorClampBenchCase:
    """Workload adapter for the Tensor-bound ``ClampFwdOp``.

    Ported from the GPU ``workloads/elementwise.py`` (``TensorClampBenchCase``).
    Holds the post-broadcast output shape so :class:`ManifestBenchmark` can
    read a single ``n_total`` while the bench builds per-operand tensors from
    the manifest-declared ``input_shape`` / ``min_shape`` / ``max_shape`` keys.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of the hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        input_shape: tuple,
        dtype: torch.dtype,
        min_shape: Optional[tuple] = None,
        max_shape: Optional[tuple] = None,
    ):
        self.input_shape = input_shape
        self.min_shape = min_shape
        self.max_shape = max_shape
        broadcast_args = [input_shape]
        if min_shape is not None:
            broadcast_args.append(min_shape)
        if max_shape is not None:
            broadcast_args.append(max_shape)
        self.shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = get_device_backend().name
        x = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        tensors: list[torch.Tensor] = [x]
        if self.min_shape is not None:
            tensors.append(torch.randn(self.min_shape, device=device, dtype=self.dtype) - 0.5)
        if self.max_shape is not None:
            tensors.append(torch.randn(self.max_shape, device=device, dtype=self.dtype) + 0.5)
        return tuple(tensors)


class _GenerativeWorkload:
    """Shape/dtype descriptor for generative ops (no input tensors).

    Ported verbatim from TileOPs ``workloads/elementwise.py`` (shared by the
    ``AlibiFwdOp`` / ``SinusoidalFwdOp`` manifests).  The full output is
    synthesized from the construction-time parameters, so there is no input
    tensor to draw: ``gen_inputs()`` returns an empty tuple.  This class
    performs no device-touching allocation, so adaptation point W1
    (``device="cuda"`` -> ``backend.name``) is a no-op here.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple:
        return ()


class TensorClampManifestWorkload:
    """Manifest-driven workload for the Tensor-bound clamp family.

    Ported from TileOPs ``workloads/elementwise.py`` ``TensorClampBenchCase``.
    Holds the post-broadcast output shape so :class:`ManifestBenchmark` can
    read a single ``n_total`` while the bench builds per-operand tensors from
    the manifest-declared ``input_shape`` / ``max_shape`` keys.

    ``input`` uses ``randn`` while ``max`` uses ``randn + 0.5`` (a slightly
    shifted upper bound, so both the clamped and the pass-through branch are
    exercised).

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        dtype: torch.dtype,
        max_shape: tuple[int, ...] | None = None,
    ):
        self.input_shape = input_shape
        self.max_shape = max_shape
        broadcast_args = [input_shape]
        if max_shape is not None:
            broadcast_args.append(max_shape)
        self.shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        x = torch.randn(self.input_shape, device=device, dtype=self.dtype)
        mx = torch.randn(self.max_shape, device=device, dtype=self.dtype) + 0.5
        return x, mx


class RandnPairWorkload(WorkloadBase):
    """Two same-shape ``randn`` vectors -- the default binary-op input.

    Ported from TileOPs ``workloads/elementwise.py`` ``RandnPairWorkload``.
    Used by the comparison correctness tests (two flat ``(n_total,)``
    ``randn`` tensors of the requested float dtype).

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        a = torch.randn(self.n_total, dtype=self.dtype, device=device)
        b = torch.randn(self.n_total, dtype=self.dtype, device=device)
        return a, b


class BitwiseWorkload(WorkloadBase):
    """Workload definition for bitwise binary ops (flat same-dtype inputs).

    Ported from TileOPs ``workloads/elementwise.py`` ``BitwiseWorkload``.
    Used by the correctness tests, which generate two flat ``(n_total,)``
    tensors with the requested ``dtype``:

    - ``torch.bool``: ``torch.randint(0, 2, ..., dtype=torch.bool)``.
    - integer dtypes: ``torch.randint(lo, hi, ..., dtype=dtype)`` where
      ``[lo, hi)`` is the intersection of the ``[-1000, 1000)`` window with
      the dtype's representable range, so unsigned / sub-byte types such as
      ``uint8`` / ``int8`` stay in bounds.

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, n_total: int, dtype: torch.dtype = torch.int32):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        if self.dtype is torch.bool:
            a = torch.randint(0, 2, (self.n_total,), dtype=torch.bool, device=device)
            b = torch.randint(0, 2, (self.n_total,), dtype=torch.bool, device=device)
        else:
            info = torch.iinfo(self.dtype)
            lo = max(-1000, int(info.min))
            hi = min(1000, int(info.max) + 1)
            a = torch.randint(lo, hi, (self.n_total,), dtype=self.dtype, device=device)
            b = torch.randint(lo, hi, (self.n_total,), dtype=self.dtype, device=device)
        return a, b


class PositivePairWorkload(WorkloadBase):
    """Two same-shape vectors in ``[0.1, 1.1)`` -- for ops undefined at/below 0.

    Ported from TileOPs ``workloads/elementwise.py`` ``PositivePairWorkload``.
    Used by the floor-divide correctness tests (both operands strictly
    positive, so the divisor never reaches zero).

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        a = torch.rand(self.n_total, dtype=self.dtype, device=device) + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device=device) + 0.1
        return a, b


class BinaryManifestWorkload:
    """Manifest-driven workload for two-input elementwise ops.

    Ported from TileOPs ``workloads/elementwise.py``
    ``BinaryManifestWorkload``.  Generates ``(input, other)`` tensors of
    the manifest-declared shapes and dtype.  ``BitwiseAndFwdOp``'s
    benchmark uses it with ``integer=True`` (both operands are random
    integers, matching the bitwise input domain).

    Adaptation point W1: ``device`` is resolved from
    :func:`get_device_backend` instead of hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        other_shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        positive: bool = False,
        integer: bool = False,
        logical: bool = False,
    ):
        self.input_shape = input_shape
        self.other_shape = other_shape
        self.a_shape = input_shape
        self.b_shape = other_shape
        self.shape = tuple(torch.broadcast_shapes(input_shape, other_shape))
        self.n_total = prod(self.shape)
        self.dtype = dtype
        self.positive = positive
        self.integer = integer
        self.logical = logical

    def _tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        device = get_device_backend().name
        if self.dtype is torch.bool:
            return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        if self.integer:
            return torch.randint(-1000, 1000, shape, device=device, dtype=self.dtype)
        if self.positive:
            return torch.rand(shape, device=device, dtype=self.dtype) + 0.1
        if self.logical:
            return (torch.randn(shape, device=device, dtype=self.dtype) > 0).to(self.dtype)
        return torch.randn(shape, device=device, dtype=self.dtype)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tensor(self.input_shape), self._tensor(self.other_shape)
