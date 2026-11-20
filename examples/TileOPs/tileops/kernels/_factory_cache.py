"""Process-global build cache for tilelang kernel factories.

Why this exists (measured S0, ``reports/tileops/_measure/``):

    op wall − Σkernel was 97.6–100% for 12 of 13 operators; MaxPool2d
    alexnet-ceil/bf16 cost 234,941 µs per call while its device kernel ran
    37 µs.  The custom_op wrappers call

        factory(*scalars, dtype)(block_m)(x)

    on **every** forward.  ``factory`` is ``lru_cache``d in some modules and
    not in others, but the *build* callable it returns re-runs
    ``@tilelang.jit`` + ``@T.prim_func`` construction on every call.  The
    on-disk ``tilelang.compile`` cache saves the compiler run, not the
    PrimFunc/JIT-wrapper construction, so ~35–250 ms of host work is repeated
    per call.

``kernel_factory_cache`` memoises both levels, process-globally:

  * ``factory(*args, **kwargs)`` -> one build callable per argument tuple;
  * ``build(block_m, ...)``       -> one compiled kernel per
    ``(block_m, ..., current NPU device index)``.

The caches live at module scope (created when the decorated factory is
imported), so they cover every Op instance, the ``torch.library.custom_op``
path, the benchmark/test entries and any direct factory user.

Device index is part of the build key because a factory may query device
properties while building (e.g. ``NPUUtils.get().get_aicore_num()``); for
``target="npuir"`` the compiled artifact itself is device-index independent,
but including it keeps multi-device processes from sharing a kernel whose
tiling was derived from another device's core count.

The decorator is signature-transparent: ``functools.wraps`` preserves name /
doc, and ``cache_clear()`` / ``cache_info()`` are provided for tests and
probes (``cache_info`` reports dict counters, not ``functools``' named tuple).
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable

import torch

__all__ = ["current_device_index", "kernel_factory_cache"]

_LOCK = threading.Lock()


def current_device_index() -> int | None:
    """Current NPU device index, or ``None`` when no NPU is available."""
    try:
        if torch.npu.is_available():
            return int(torch.npu.current_device())
    except Exception:  # noqa: BLE001 - device query must never break a build
        return None
    return None


def _memoize_build(build: Callable) -> Callable:
    """Cache one build callable's results per (args, kwargs, device)."""
    cache: dict[Any, Any] = {}
    lock = threading.Lock()

    @functools.wraps(build)
    def build_cached(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())), current_device_index())
        with lock:
            if key in cache:
                return cache[key]
        value = build(*args, **kwargs)
        with lock:
            return cache.setdefault(key, value)

    def cache_clear() -> None:
        with lock:
            cache.clear()

    def cache_info() -> dict:
        with lock:
            return {"kernels": len(cache)}

    build_cached.cache_clear = cache_clear  # type: ignore[attr-defined]
    build_cached.cache_info = cache_info  # type: ignore[attr-defined]
    build_cached.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
    return build_cached


def kernel_factory_cache(factory: Callable) -> Callable:
    """Memoise ``factory`` and the build callable it returns."""
    factories: dict[Any, Callable] = {}

    @functools.wraps(factory)
    def factory_cached(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        with _LOCK:
            build = factories.get(key)
        if build is None:
            produced = factory(*args, **kwargs)
            build = _memoize_build(produced) if callable(produced) else produced
            with _LOCK:
                build = factories.setdefault(key, build)
        return build

    def cache_clear() -> None:
        with _LOCK:
            factories.clear()

    def cache_info() -> dict:
        with _LOCK:
            builds = list(factories.values())
        kernels = sum(b.cache_size() for b in builds if hasattr(b, "cache_size"))
        return {"factories": len(builds), "kernels": kernels}

    factory_cached.cache_clear = cache_clear  # type: ignore[attr-defined]
    factory_cached.cache_info = cache_info  # type: ignore[attr-defined]
    return factory_cached
