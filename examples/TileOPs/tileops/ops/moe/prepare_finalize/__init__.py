"""MoE prepare-and-finalize implementations (NPU).

Ported from GPU TileOPs ``tileops/ops/moe/prepare_finalize/__init__.py``.
Host-side orchestration only -- no TileLang kernel.
"""

from .no_dp_ep import MoEPrepareAndFinalizeNoDPEP

__all__ = ["MoEPrepareAndFinalizeNoDPEP"]
