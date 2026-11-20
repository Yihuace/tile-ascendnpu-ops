"""Workload definitions for the regularization op family (NPU).

Ported from the GPU ``DropoutBenchmarkWorkload`` defined inline in
``{gpu_repo_root}/benchmarks/ops/bench_dropout.py`` (there is no GPU
``workloads/regularization.py``; the GPU bench subclasses the shared
``ShapedRandnWorkload``).  The workload draws one ``randn`` tensor of the
manifest ``input_shape`` and carries the op-specific parameters ``p`` /
``training``.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
(via ``RandnWorkload.gen_inputs``) instead of the hard-coded ``"cuda"`` in the
original TileOPs code.
"""

from math import prod

import torch
import torch.nn.functional as F

from tileops.workloads.workload_base import RandnWorkload

__all__ = ["DropoutOpWorkload"]


class DropoutOpWorkload(RandnWorkload):
    """Workload for ``torch.nn.functional.dropout`` (inverted dropout).

    One ``randn`` tensor of ``shape`` with element count ``n_total``.  The
    op-specific parameters ``p`` / ``training`` are carried so the benchmark
    can construct both the op and the PyTorch performance baseline.

    Args:
        shape: Input tensor shape (the manifest ``input_shape``).
        dtype: Data type (float16, bfloat16, or float32).
        p: Drop probability in [0, 1].
        training: If ``False`` the baseline is an identity pass-through.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype, p: float = 0.5, training: bool = True):
        super().__init__(tuple(shape), dtype)
        self.n_total = prod(self.shape)
        self.p = p
        self.training = training

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch native dropout -- retained as a performance baseline.

        ``torch.nn.functional.dropout`` is a native operator (not an
        accuracy-only golden), so it is a meaningful benchmark baseline.
        """
        return F.dropout(x, p=self.p, training=self.training)
