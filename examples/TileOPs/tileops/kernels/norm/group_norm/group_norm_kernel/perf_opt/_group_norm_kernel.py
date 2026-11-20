# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Stage 4 (perf_opt) re-export shim for the GroupNorm forward kernel.

The read-only wrapper ``group_norm.py`` carries a pre-wired baseline/perf_opt
import switch whose perf_opt branch imports::

    from .group_norm_kernel.perf_opt._group_norm_kernel import ALIGNMENT
    from .group_norm_kernel.perf_opt._group_norm_kernel import _align_up
    from .group_norm_kernel.perf_opt._group_norm_kernel import _group_norm_kernel

This module makes that switch a pure one-line toggle by re-exporting those
three names from the tuned implementation :mod:`.GroupNormFwdOp` -- it holds
**no implementation of its own** (a copied body would drift from the tuned
kernel, which is what the gate lint measures).

The factory/interface contract is unchanged::

    _group_norm_kernel(M, D, eps, dtype_str) -> _func(block_m) -> main
    main(x, weight, bias, y)                                    # out_idx=[3]
    ALIGNMENT == 1, _align_up(n, 1) == n

Only the *defaults* differ (``perf_opt.GroupNormFwdOp``): ``block_m`` comes
from the measured UB budget of this structure, the wrapper's structural
``block_m == 1`` is read as "unspecified -> tuned value", and the persistent
grid is balance-shrunk.  All of that lives inside the factory, so the wrapper
needs no paired parameter change -- see ``perf_opt/opt_log.md``.
"""

from .GroupNormFwdOp import ALIGNMENT, _align_up, _group_norm_kernel

__all__ = ["ALIGNMENT", "_align_up", "_group_norm_kernel"]
