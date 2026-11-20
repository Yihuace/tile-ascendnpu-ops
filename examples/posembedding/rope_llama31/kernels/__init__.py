"""TileLang Llama 3.1 RoPE kernels."""

import tilelang


def _jit_options():
    return {
        "target": "npuir",
        "out_idx": [-1],
        "pass_configs": {
            tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: True,
            tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC: False,
        },
    }


from .blocked_fp32 import build_blocked_fp32
from .blocked_low_precision import build_blocked_low_precision
from .table_fp32 import build_table_fp32
from .table_low_precision import build_table_low_precision

__all__ = [
    "build_blocked_fp32",
    "build_blocked_low_precision",
    "build_table_fp32",
    "build_table_low_precision",
]
