"""Stage 4 tuned drop-in kernel (perf_opt).

Re-exports the same names as the Stage 3 package ``engram_gate_conv_fwd_kernel``
(``ALIGNMENT`` / ``_align_up`` / ``_engram_gate_conv_fwd_kernel``) so the
integration wrapper's baseline/perf_opt switch block can import this module
without further glue.
"""

from ._engram_gate_conv_fwd_kernel import ALIGNMENT
from ._engram_gate_conv_fwd_kernel import _align_up
from ._engram_gate_conv_fwd_kernel import _engram_gate_conv_fwd_kernel

__all__ = ["ALIGNMENT", "_align_up", "_engram_gate_conv_fwd_kernel"]
