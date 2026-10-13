"""Shared utility functions (device-agnostic)."""

import torch

str2dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def get_device_str() -> str:
    """Return the current backend's device string (e.g. ``"npu"``, ``"cuda"``)."""
    from tileops.device import get_device_backend

    return get_device_backend().name


def is_hopper() -> bool:
    """Whether the current CUDA device is SM90 (Hopper). Always False on NPU."""
    backend = __import__("tileops.device", fromlist=["get_device_backend"]).get_device_backend()
    if backend.name != "cuda":
        return False
    return torch.cuda.get_device_capability() == (9, 0)
