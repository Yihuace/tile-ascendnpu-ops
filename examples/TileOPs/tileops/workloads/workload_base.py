"""Base classes for workload definitions shared between tests and benchmarks.

Extracted from TileOPs ``workloads/workload_base.py`` with one adaptation:
``RandnWorkload.gen_inputs`` uses the device backend instead of hard-coding
``"cuda"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import torch

from tileops.device import get_device_backend

_F = TypeVar("_F", bound=Callable[..., Any])


class WorkloadBase(ABC):
    """Abstract base for workload definitions (input generation + parameters)."""

    @abstractmethod
    def gen_inputs(self) -> tuple[Any, ...]:
        raise NotImplementedError


class RandnWorkload(WorkloadBase):
    """Workload base for ops whose inputs are generated via ``torch.randn``.

    Adaptation point: ``device`` is resolved from :func:`get_device_backend`
    instead of the hard-coded ``"cuda"`` in the original TileOPs code.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        backend = get_device_backend()
        device = backend.name
        x = torch.randn(*self.shape, dtype=self.dtype, device=device)
        return (x,)


class FixtureMeta(type):
    """Metaclass that makes Fixture subclasses usable as @decorators."""

    def __call__(cls, fn: _F) -> _F:
        import pytest

        params = cls.get_params() if hasattr(cls, "get_params") else cls.PARAMS
        for names, values in reversed(params):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class FixtureBase(metaclass=FixtureMeta):
    """Base class for reusable parametrize decorators."""

    PARAMS = []
