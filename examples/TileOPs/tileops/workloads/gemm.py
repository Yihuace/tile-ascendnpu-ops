"""Workload definitions for the gemm op family (NPU).

Ported from TileOPs ``workloads/gemm.py`` -- the ``GemmWorkload`` and
``GemmW4A16Workload`` (+ ``quantize_weight_int4``) subsets.  The shared GPU
file also defines ``GemmFp8Workload``, which belongs to ``GemmFp8Op`` and is
not migrated.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` in the original TileOPs code.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128

def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = W4A16_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine group-wise quantize and pack a logical ``[N, K]`` weight tensor."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1).clamp_max(0)
    group_max = grouped.amax(dim=-1).clamp_min(0)
    scale = ((group_max - group_min) / 15.0).clamp_min(1e-12)
    zero = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )

    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = ((quantized.float() - zero.float().unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(
        n, k
    )
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


class GemmW4A16Workload(WorkloadBase):
    """Input generator for ``GemmW4A16Op`` (W4A16 affine-quantized GEMM).

    Produces ``(activation, packed_weight, weight_scale, weight_zero)`` and
    keeps the dequantized reference weight for the correctness test.  W1: the
    device is resolved from :func:`get_device_backend`.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        group_size: int = W4A16_GROUP_SIZE,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self._dequantized_weight: torch.Tensor | None = None

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        activation = torch.randn(self.m, self.k, device=device, dtype=self.dtype)
        # Quantize/pack on CPU, then move to the device.  The int4 packing uses
        # bitwise ops (``<<``) that would otherwise be lazily compiled on the
        # NPU; input generation must not depend on NPU op compilation, which is
        # fragile when ``import tilelang`` has set ``TVM_LIBRARY_PATH`` to the
        # repo's ``build/tvm`` (CANN's TBE/AOE then loads the wrong TVM).
        source_weight = torch.randn(self.n, self.k, dtype=torch.float32) * 0.25
        packed, scale, zero, dequantized = quantize_weight_int4(
            source_weight, group_size=self.group_size
        )
        self._dequantized_weight = dequantized.to(self.dtype).to(device).contiguous()
        return activation, packed.to(device), scale.to(device), zero.to(device)

    @property
    def dequantized_weight(self) -> torch.Tensor:
        if self._dequantized_weight is None:
            raise RuntimeError("dequantized_weight is available after gen_inputs()")
        return self._dequantized_weight


class GemmWorkload(WorkloadBase):
    """Input generator for ``GemmOp`` (``d = op(a) @ op(b)``).

    ``m`` / ``n`` / ``k`` are the logical (transpose-independent) GEMM dims;
    ``trans_a`` / ``trans_b`` pick the storage layout of ``a`` / ``b``.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        trans_a: bool = False,
        trans_b: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_device_backend().name
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device=device, dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device=device, dtype=self.dtype)
        return a, b