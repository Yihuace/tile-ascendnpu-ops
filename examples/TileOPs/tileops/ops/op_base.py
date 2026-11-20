"""Op base class — device-agnostic.

Adaptation from GPU (TileOPs) to NPU:

- Removed ``from tileops.utils import get_sm_version`` and arch-compat
  checking against CUDA SM integers.  NPU architecture validation uses
  chip family name strings via the device backend.
- Removed ``from .compile_boundary import register_instance`` — the
  ``torch.compile`` dispatch boundary was GPU/TileLang-specific (custom_op
  wrappers around TileLang JIT kernels).  NPU PyTorch-native ops are
  already dynamo-traceable.
- Removed ``__init_subclass__`` manifest codegen hooks (dtype validator,
  roofline codegen) — those relied on the full TileOPs manifest machinery.
  Concrete ops implement ``_validate_dtypes`` and ``eval_roofline`` directly.
- Removed ``autotune`` method — NPU kernels use heuristic config only.
- ``_cache_key`` is preserved for kernel cache deduplication.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Hashable, Optional, Union

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel


class Op(ABC):
    """Base class for NPU benchmark operations.

    Mirrors the original TileOPs ``Op`` interface: hardware-aware kernel
    dispatch, correctness testing, performance profiling.

    Subclasses set:
    - ``default_kernel_map``: maps kernel keys to Kernel classes.
    - ``_static_axes``: frozenset of (input_index, axis) pairs for cache keys.
    """

    kernel: Optional[Kernel] = None
    kernel_map: Optional[dict[str, Kernel]] = None
    dtype: Optional[torch.dtype] = None
    input_shapes: Optional[list[tuple]] = None
    _static_axes: frozenset[tuple[int, int]] = frozenset()

    @property
    @abstractmethod
    def default_kernel_map(self) -> dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def _install_kernel_map(self, candidate_map: Optional[dict[str, Kernel]] = None) -> None:
        """Validate and install the resolved kernel map.

        NPU adaptation: arch-compat check uses device backend name matching
        instead of CUDA SM integer comparison.
        """
        default_map = self.default_kernel_map
        if default_map is None or len(default_map) == 0:
            self.kernel_map = dict(candidate_map) if candidate_map else {}
            return

        resolved: dict[str, Kernel] = {}
        backend = get_device_backend()
        for name, default_kernel in default_map.items():
            if candidate_map is not None and name in candidate_map:
                kernel_type = candidate_map[name]
            else:
                kernel_type = default_kernel
            if (
                kernel_type is not None
                and kernel_type.supported_archs is not None
                and not _arch_supported(kernel_type.supported_archs, backend)
            ):
                raise ValueError(
                    f"{kernel_type.__name__} is not supported on backend "
                    f"{backend.name} (supported: {kernel_type.supported_archs})"
                )
            resolved[name] = kernel_type
        self.kernel_map = resolved

    def dispatch_kernel(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map."""
        self._install_kernel_map(kernel_map)

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        return self.forward(*args, **kwargs)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Concrete ops override this with plain-Python arithmetic over
        ``self.*`` attributes.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.eval_roofline() must be implemented by the concrete Op."
        )

    def _cache_key(self, *input_shapes: tuple[int, ...]) -> Hashable:
        """Return a cache key for kernel dispatch given forward-time input shapes."""
        if not self._static_axes and type(self)._cache_key is Op._cache_key:
            cls = type(self)
            warnings.warn(
                f"{cls.__name__}: Op._cache_key() called with empty _static_axes "
                f"and no subclass override.",
                stacklevel=2,
            )
        return tuple(
            s
            for i, shape in enumerate(input_shapes)
            for axis, s in enumerate(shape)
            if (i, axis) not in self._static_axes
        )


def _arch_supported(supported: list[str], backend: object) -> bool:
    """Check whether the current backend matches any supported arch pattern.

    NPU adaptation: ``supported`` is a list of device name patterns
    (e.g. ``["Ascend910B"]``).  ``None`` means all architectures are supported.
    """
    if supported is None:
        return True
    device_name = backend.get_device_name(0) if backend.is_available() else ""
    return any(arch in device_name for arch in supported)
