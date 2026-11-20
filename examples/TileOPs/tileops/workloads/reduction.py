"""Workload definitions for the reduction op family."""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import RandnWorkload, WorkloadBase


class LogSumExpWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""


class ArgmaxWorkload(RandnWorkload):
    """Workload definition for ArgmaxFwdOp (spec interface: shape + dtype)."""


class ArgminWorkload(RandnWorkload):
    """Workload definition for ArgminFwdOp (spec interface: shape + dtype)."""


class _LogicalWorkload(WorkloadBase):
    """Shared workload base for logical reduce ops (any, all, count_nonzero).

    Generates inputs with a mix of zeros and non-zeros for meaningful
    logical reduction testing. Boolean, integer, float, and complex
    dtypes are supported.

    Ported from the GPU ``workloads/reduction.py``; the device is resolved
    from the device backend (W1) instead of the hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


class AllWorkload(_LogicalWorkload):
    """Workload definition for AllFwdOp."""


class AnyWorkload(_LogicalWorkload):
    """Workload definition for AnyFwdOp."""


class CountNonzeroWorkload(_LogicalWorkload):
    """Workload definition for CountNonzeroFwdOp."""


def _make_logical_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor with a mix of zeros and non-zeros.

    When the first dimension is large enough (>4), the first row is forced
    to all-zero (meaningful for ``any``) and the second row to all-nonzero
    (meaningful for ``all``).
    """
    device = get_device_backend().name
    m = shape[0] if len(shape) >= 1 else 1

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
        if m > 4:
            x[0] = False
            x[1] = True
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(*shape, dtype=torch.float32, device=device)
        imag = torch.randn(*shape, dtype=torch.float32, device=device)
        x = torch.complex(real, imag).to(dtype)
        if m > 4:
            x[0] = 0 + 0j
            x[1] = 1 + 1j
    elif dtype in (torch.int32, torch.int64):
        x = torch.randint(-5, 6, shape, dtype=dtype, device=device)
        if m > 4:
            x[0] = 0
            x[1] = 1
    else:
        x = torch.randn(*shape, dtype=dtype, device=device)
        if m > 4:
            x[0] = 0.0
            x[1] = 1.0

    return x


class AmaxWorkload(RandnWorkload):
    """Workload definition for AmaxFwdOp (spec interface: shape + dtype)."""


class AminWorkload(RandnWorkload):
    """Workload definition for AminFwdOp (spec interface: shape + dtype)."""


class L1NormWorkload(RandnWorkload):
    """Workload definition for L1NormFwdOp (shape + dtype, randn input).

    ``randn`` spans both signs so the ``|x|`` step inside the kernel is
    actually exercised (a non-negative generator would make l1 equal a plain
    sum and hide sign-handling bugs).
    """


class L2NormWorkload(RandnWorkload):
    """Workload definition for L2NormFwdOp (shape + dtype, randn input)."""


class InfNormWorkload(RandnWorkload):
    """Workload definition for InfNormFwdOp (shape + dtype, randn input).

    Note: the golden path for ``ord=inf`` is a pure max, so random input never
    produces NaN rows -- NaN propagation is covered by a dedicated test that
    injects NaNs explicitly.
    """
