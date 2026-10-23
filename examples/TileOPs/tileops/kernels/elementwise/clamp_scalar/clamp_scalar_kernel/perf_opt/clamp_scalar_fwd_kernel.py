# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Thin forwarder -- Stage-4 R2 delivered kernel lives in ``ClampScalarFwdOp.py``.

The Stage-5 wrapper imports this module path
(``...clamp_scalar_kernel.perf_opt.clamp_scalar_fwd_kernel``), and the R1
records / opt_log refer to it as the deployed artifact.  Since R2 the single
source of truth is ``perf_opt/ClampScalarFwdOp.py`` (state
``kernel_opt_py_path``); this file only re-exports it so the wrapper import
stays unchanged.

Run the hierarchical tests against the real artifact::

    python ClampScalarFwdOp.py --level all
"""

try:  # imported as ``...perf_opt.clamp_scalar_fwd_kernel`` (wrapper path)
    from .ClampScalarFwdOp import TUNED_BLOCK_SIZE, clamp_scalar_fwd_kernel
except ImportError:  # loaded as a top-level module (script / importlib probe)
    from ClampScalarFwdOp import TUNED_BLOCK_SIZE, clamp_scalar_fwd_kernel

__all__ = ["clamp_scalar_fwd_kernel", "TUNED_BLOCK_SIZE"]
