"""TopkSelectorKernel -- NPU kernel wrapper for TopkSelectorOp.

Adaptation from GPU (TileOPs) to NPU
------------------------------------
The GPU kernel is a CUDA-thread radix-select: it needs ``threads=1024``,
``T.get_thread_binding()``, a 256-bin shared histogram with
``T.atomic_add(..., return_prev=True)``, ``T.sync_threads()`` and
``T.dynamic()`` shapes.  **None of those exist on npuir** (K3/K9 remove
``threads``; there are no in-block atomics, no CUDA-semantics barriers, and
shapes must be static), so the algorithm was redesigned rather than ported.

NPU design -- two simple kernels plus a HOST-side bisection
-----------------------------------------------------------
The exact top-k is found by bisecting a threshold and counting:

    count_kernel(x, thr)  -> cnt    # per row: how many elements are >= thr
    select_kernel(x, thr) -> idx    # per row: the indices of those >= thr

The bisection loop runs on the HOST (a loop there costs nothing); each kernel is
a single pass with no unrolled rounds.

WHY NOT ONE KERNEL (measured; see the per-op repro/ probes)
  * a kernel holding the compaction PLUS two or more unrolled copies of the
    bisection body SEGFAULTS the build (``rounds=1`` and every write-form
    combination build and run; ``rounds>=2`` always crash);
  * multi-block GM writes are UNRELIABLE -- ``T.Kernel(M)`` (one block per row)
    lost half the rows, while ``T.Kernel(1)`` + a serial loop over rows was
    correct, so every kernel here uses a single block with a serial row loop;
  * the row is streamed in ``tile``-wide pieces so the row need not fit in UB.

Measured primitive rules this relies on
  * a ``T.Parallel`` assignment loop's RHS must not read a (1,1) buffer, and a GM
    scalar must be copied into a (1,1) UB buffer first, then used as an OPERAND
    of a row-wide v-op (``T.vadd(row, nts[0, 0], diff)``);
  * a GUARDED scatter silently writes nothing -- the mask is folded into the
    POSITION arithmetic and non-selected lanes are aimed at a sink slot;
  * counting is done in the fp32 domain (the 1-D int32 reduce is wrong).

Performance note (honest): the single-block + serial-row architecture is correct
but serial, so large ``M`` is slow.  Kernel tuning was NOT done for this op (the
migration harness skips the tuning stage); correctness was prioritised.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

from ._topk_selector_kernel._topk_selector_kernel import (
    ROUNDS,
    _fused_fits_ub,
    _row_fits_ub,
    _topk_bisect_kernel,
    _topk_count_kernel,
    _topk_fused_kernel,
    _topk_select_kernel,
)

__all__ = ["TopkSelectorKernel", "ROUNDS"]


class TopkSelectorKernel(Kernel):
    """Exact top-k indices along the last dimension.

    Args:
        M: Number of rows (product of all dims except the last).
        N: Width of each row (the dimension top-k selects along).
        k: How many indices to return per row.
        rounds: Bisection rounds (host side).
        tile: Stream each row in pieces of this width (``None`` = whole row).
            Needed when a row does not fit in UB.
        config: Optional kernel config override (unused; kept for the base API).
    """

    # K5: CUDA SM list -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernels; pin Developer regardless of the ambient
    # TILELANG_ASCEND_MODE.
    ascend_mode = "Developer"

    def __init__(
        self,
        M: int,
        N: int,
        k: int,
        rounds: int = ROUNDS,
        tile: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if M < 1 or N < 1 or k < 1:
            raise ValueError(f"bad shape M={M} N={N} k={k}")
        if k > N:
            raise ValueError(f"k={k} exceeds N={N}")
        self.M = M
        self.N = N
        self.k = k
        self.rounds = rounds
        self.tile = tile
        # Fast path: when the whole row fits in UB the ENTIRE round loop runs inside one
        # kernel, so the op issues 2 launches instead of rounds+1.  Measured: a launch
        # costs ~0.11 ms regardless of payload, so the old host bisection spent
        # 40 x 0.11 = 4.4 ms of pure launch overhead while the device work was 0.34 ms.
        # Fused first: one launch per call keeps the harness's measurement contract
        # (N samples of the SAME kernel) intact.  Only when the fused form does not
        # fit UB -- N=8192 needs 200 KB -- fall back to the two-kernel path.
        self._kf = (_topk_fused_kernel(M, N, k, rounds)
                    if tile is None and _fused_fits_ub(N) else None)
        self._kb = (_topk_bisect_kernel(M, N, k, rounds)
                    if tile is None and not _fused_fits_ub(N) and _row_fits_ub(N) else None)
        self._kc = _topk_count_kernel(M, N, tile=tile)
        self._ks = _topk_select_kernel(M, N, k, tile=tile)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """No tunable block/thread config on the NPU path (K9)."""
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (M, k) int32 index tensor for the (M, N) input."""
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if self._kf is not None:
            return self._kf(x)                       # 1 launch: bisect + select fused
        if self._kb is not None:
            return self._ks(x, self._kb(x))          # 2 launches total
        lo = x.min(dim=1).values.clone()
        hi = x.max(dim=1).values.clone()
        for _ in range(self.rounds):
            mid = (lo + hi) * 0.5
            c = self._kc(x, mid)
            ge = c >= float(self.k)
            lo = torch.where(ge, mid, lo)
            hi = torch.where(ge, hi, mid)
        return self._ks(x, lo)
