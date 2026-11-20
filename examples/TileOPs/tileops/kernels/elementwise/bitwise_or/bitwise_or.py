"""BitwiseOr forward kernel using TileLang (NPU-adapted).

Implements element-wise bitwise OR with PyTorch broadcast semantics:
    y = input | other

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3
(the elementwise family pattern):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_make_binary_explicit``, inlined
    with ``BitwiseOrFwdKernel.op_func = a | b`` as
    ``bitwise_or_fwd_kernel``) is extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is.  It serves as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator,
    grid/sync, ``threads``/``npt`` collapse into ``block_size``,
    padding strategy) are handled by the NPU component during
    re-implementation:
      - K11: factory signature keeps its positional shape/stride args
        but drops the GPU ``threads``/``num_per_thread`` execution
        parameters; the returned callable takes a single ``block_size``
        (the GPU ``threads * num_per_thread`` product).

  **Part B -- Kernel class** (ported in full from GPU ``BinaryKernel`` +
    ``BitwiseOrFwdKernel`` / ``BitwiseOrBoolStorageFwdKernel``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7/K10: no ``@torch.library.custom_op`` wrapper (the GPU binary
         kernel has none either); ``forward`` calls the compiled kernel
         directly.
    K8/K9: ``autotune_configs`` / ``autotune()`` / ``tune`` param --
         removed; heuristic config selection only,
         ``init_config(config)`` takes no ``tune`` argument.
    K11: GPU config ``{"strategy", "threads", "num_per_thread"}``
         collapsed into ``{"block_size": threads * num_per_thread}``;
         the strategy dispatch (direct / explicit_parallel /
         register_copy) collapses to the single NPU ``block_size``
         kernel (the extracted GPU function corresponds to the GPU
         ``explicit_parallel`` default).

  The GPU op routes ``torch.bool`` tensors through a dedicated
  ``BitwiseOrBoolStorageFwdKernel`` (public bool storage viewed as
  ``uint8``) because the CUDA codegen cannot lower vectorised
  ``boolx<N>`` loads.  Both GPU kernel classes are ported here; on the
  NPU side they share the single extracted
  ``bitwise_or_fwd_kernel`` (the pointwise ``a | b`` and
  ``T.bitwise_or(a, b)`` are identical on ``uint8`` storage), keeping one
  TileLang kernel function for the op per the skill's elementwise model.
"""

import math
import sys
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A imports: the extracted GPU TileLang kernel functions.  The
# conductor's ``integrate_kernel.py`` rewrites this import into the
# baseline/perf_opt source-selection block once the NPU kernel component
# lands the ``target="npuir"`` re-implementation in ``bitwise_or_kernel/``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .bitwise_or_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .bitwise_or_kernel import bitwise_or_fwd_kernel, coalesce_broadcast_dims
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# Stage 4 round 7 (v3 int32-pack): block_size the packed int32 view compiles
# with (the bool-carrier default exceeds the int32 UB budget).
from .bitwise_or_kernel.perf_opt.BitwiseOrFwdOp import (
    TUNED_PACK_BLOCK_SIZE,
    bitwise_or_fwd_kernel,
    coalesce_broadcast_dims,
)

__all__ = ["BitwiseOrBoolStorageFwdKernel", "BitwiseOrFwdKernel"]

# Manifest dtype contract for the bitwise family (GPU ``_BITWISE_DTYPES``).
_BITWISE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

# K11: GPU ``threads(256) * num_per_thread`` collapsed into ``block_size``.
# Stage 4 tuning replaced the K11 GPU-derived defaults with per-dtype tuned
# constants (perf_opt module: TUNED_BLOCK_SIZE / TUNED_BOOL_BLOCK_SIZE).
# Toggle this paired pair together with the source block above.
# --- baseline (Stage 3) ---
# _DEFAULT_BLOCK_SIZE = 256 * 8                # non-1-byte dtypes
# _BOOL_STORAGE_DEFAULT_BLOCK_SIZE = 256 * 16  # bool/uint8 int8 carrier
# --- perf_opt (Stage 4 tuned) ---
_DEFAULT_BLOCK_SIZE = 4096  # non-1-byte dtypes
_BOOL_STORAGE_DEFAULT_BLOCK_SIZE = 8192  # bool/uint8 int8 carrier


def _kernel_dtype_str(dtype: torch.dtype) -> str:
    """Map a public op dtype to the npuir-supported kernel dtype string.

    npuir cannot carry ``uint8`` GM tensors (CG-2026-0018: the output is
    silently remapped to float32), so the 1-byte unsigned / ``bool``
    storage paths are carried as ``int8`` instead.  Bitwise OR is a
    per-bit operation, so the same byte has an identical bit pattern under
    ``int8`` and ``uint8``; the substitution is bit-exact (mirrors the
    BitwiseAnd migration's DESIGN.md R3 adapter).  This is the
    integration-time adapter for the frozen Op/bool-storage view targets
    (``view(torch.uint8)``).
    """
    if dtype in (torch.uint8, torch.bool):
        return "int8"
    return Kernel.dtype_to_str(dtype)


def _as_kernel_input(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a 1-byte unsigned/bool tensor as ``int8`` for the kernel.

    The Op layer hands the bool-storage fast path ``uint8``-viewed tensors
    (and may hand the plain path a ``uint8`` tensor); the integrated NPU
    kernel only accepts ``int8`` storage.  ``view`` preserves the bit
    pattern exactly (same 1-byte itemsize).
    """
    if tensor.dtype in (torch.uint8, torch.bool):
        return tensor.view(torch.int8)
    return tensor


def _restore_public_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reinterpret the kernel's ``int8`` output back to the public dtype.

    ``_as_kernel_input`` carries 1-byte ``uint8``/``bool`` storage as
    ``int8``, so the compiled kernel returns an ``int8`` tensor.  The frozen
    Op layer (``_eager_forward``) treats the kernel as *same-dtype-in /
    same-dtype-out* (the bool fast path then does its own
    ``result.view(torch.bool)``).  Presenting the ``int8`` carrier back as
    the original 1-byte dtype (bit-identical, same itemsize) keeps that
    contract: the plain ``uint8`` path returns ``uint8`` instead of
    ``int8``.
    """
    if dtype in (torch.uint8, torch.bool) and tensor.dtype == torch.int8:
        return tensor.view(dtype)
    return tensor


# ---------------------------------------------------------------------------
# Stage 4 round 7 (v3): host-side int32 packing of the 1-byte path
#
# A bitwise op is bit-parallel: ``a | b`` never carries between bits, so one
# int32 word computes four int8 bytes at once with a bit-identical result --
# *provided* the four bytes are four consecutive last-dim elements of the
# operand (which is exactly what ``Tensor.view(torch.int32)`` regroups on a
# contiguous buffer).  The packing therefore happens on the HOST; the kernel
# never sees a 1-byte dtype and never bit-reinterprets anything in GM/UB
# (``T.view`` over GM/UB is a separately falsified trap, CG-2026-0047).
#
# Because the kernel receives a real int32 tensor, the packed route reuses the
# already-tuned int32 path instead of the int8 path, whose
# channel-broadcast form is stuck at a few percent (short 1-byte rows: the
# per-row 1-byte scalar fetch plus the int8 ``T.vbrc`` f16 detour).  It also
# does NOT re-enter the falsified int8 2-D tile (CG-2026-0043): that gap is
# about an int8 *kernel* input, and there is none here.
# ---------------------------------------------------------------------------
_PACK_WORDS = 4  # 4 int8 bytes -> 1 int32 word


def _packed_shapes(out_shape, a_shape, b_shape):
    """Return ``(a_packed_shape, b_packed_shape, a_full_last, b_full_last)``.

    Packing is legal only when *all* of the following hold, else the caller
    falls back to the existing int8 path:

    * the broadcast output's last participating dim ``L`` is a multiple of 4
      (otherwise the tail bytes cannot be regrouped);
    * each operand is either *full* on that dim (``L`` -> ``L/4`` after
      packing) or *unit* on it (``1`` -> byte replicated into every one of the
      4 lanes, i.e. int32 value ``b | b<<8 | b<<16 | b<<24``).
    """
    if not out_shape:
        return None
    last = int(out_shape[-1])
    if last % _PACK_WORDS != 0:
        return None
    ndim = len(out_shape)
    packed = []
    full_flags = []
    for shape in (a_shape, b_shape):
        padded = (1,) * (ndim - len(shape)) + tuple(int(d) for d in shape)
        dim = padded[-1]
        if dim == last:
            packed.append(padded[:-1] + (last // _PACK_WORDS,))
            full_flags.append(True)
        elif dim == 1:
            packed.append(padded)
            full_flags.append(False)
        else:  # pragma: no cover - broadcast_shapes forbids this
            return None
    return packed[0], packed[1], full_flags[0], full_flags[1]


def _replicate_byte_to_int32(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-byte operand across the packed lanes.

    ``tensor`` holds one byte per broadcast row; the packed int32 word must
    carry that same byte in all four lanes.  The arithmetic form
    ``u | u<<8 | u<<16 | u<<24`` (with ``u`` the zero-extended byte in int32)
    wraps two's-complement for ``u >= 0x80`` and reproduces the byte four times
    bit-exactly; the shift/or form is used because it is endian-agnostic (an
    ``expand``-then-``view(int32)`` would depend on the host byte order).
    """
    u = tensor.contiguous().view(torch.uint8).to(torch.int32)
    return (u | (u << 8) | (u << 16) | (u << 24)).contiguous()


def _pack_packed_operand(tensor: torch.Tensor, full_last: bool) -> torch.Tensor:
    """Return the int32 view (or byte-replicated word) of a 1-byte operand."""
    carrier = _as_kernel_input(tensor)  # bool/uint8 -> int8 (same itemsize)
    if full_last:
        return carrier.contiguous().view(torch.int32)
    return _replicate_byte_to_int32(carrier)


# ---------------------------------------------------------------------------
# Stage 4 round 8 (v4): host-side int32 packing of the int64 path
#
# Same idea as the 1-byte packing above, one level up: an int64 element is two
# int32 words (low, high) and a bitwise op never carries between bits, so
# applying the op to the two word lanes reproduces every bit of the 64-bit
# result.  ``Tensor.view(torch.int32)`` on the contiguous host buffer regroups
# an ``(..., L)`` int64 tensor as ``(..., 2L)`` int32 words *without copying*
# -- the packing is pure host-side metadata; nothing is reinterpreted inside
# GM/UB (``T.view`` over GM/UB is a separately falsified trap, CG-2026-0047).
# The int32 path is the already-tuned one, so a packed int64 operand inherits
# its per-dtype ``block_size`` / 2-D tiling.
#
# Two shapes of packing are possible:
#
#   plain  both operands are *full* on the last dim (same-shape, leading-dim
#          broadcast, row broadcast ``[1, L]``).  Doubling the last dim gives
#          the packed shapes and the existing int32 kernel runs unchanged.
#
#   pair   one operand is *unit* on the last dim (channel / scalar broadcast).
#          Its 64-bit value becomes a two-word ``[low, high]`` row that must be
#          repeated along the last dim.  A 2-D ``[r_blk, 1]`` tile can only
#          broadcast a *single* word, so this case needs the dedicated
#          ``pack_words=2`` pair tile in the kernel module: tile
#          ``[r_pos, 2]`` x broadcast ``[1, 2]``, i.e. the same rank-2
#          ``[M, N] op [1, N]`` form the row-broadcast tile already uses.
# ---------------------------------------------------------------------------
_PACK_WORDS_I64 = 2  # one int64 -> two int32 words (low word, high word)
# Stage 4 round 8: element positions folded into one pair-tile row.  A row of
# W = 2*K words is what the DMA actually transfers; K = 1 gives 8-byte rows and
# a 12.8M-burst kernel (13.9 ms on cnn-feat-4d), so K is chosen as large as
# the "one tile row group" allows while keeping >= _PAIR_MIN_ROWS rows per task
# (each task re-reads one W-word broadcast row, i.e. b/a traffic = 1/rows).
_PAIR_POS_FOLD = 392
_PAIR_MIN_ROWS = 4


def _pair_fold(npos):
    """Largest K <= ``_PAIR_POS_FOLD`` that divides ``npos`` and keeps >=
    ``_PAIR_MIN_ROWS`` tile rows available."""
    best = 0
    for k in range(1, _PAIR_POS_FOLD + 1):
        if npos % k == 0 and npos // k >= _PAIR_MIN_ROWS:
            best = k
    return best


def _expand_pair_row(tensor: torch.Tensor, run: int, fold: int) -> torch.Tensor:
    """``(run, 2)`` int32 word pairs -> ``(run, 2*fold)`` repeated pattern.

    The broadcast operand holds one 64-bit value per block; in word space that
    is ``[low, high]``.  A tile row of ``fold`` positions needs the pair
    repeated ``fold`` times, which is a *host* expansion of a
    ``run * 2 * fold``-word buffer (459 KB for the cnn-feat shape, ~0.2 us of
    device traffic) instead of a per-element vector op.
    """
    pairs = tensor.contiguous().view(torch.int32).reshape(run, _PACK_WORDS_I64)
    return (
        pairs.unsqueeze(1)
        .expand(run, fold, _PACK_WORDS_I64)
        .reshape(run, fold * _PACK_WORDS_I64)
        .contiguous()
    )


def _compact_strides(shape):
    """Row-major (C-order) strides of ``shape``."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _pair_geometry(cshape, a_strides, b_strides, a_numel, b_numel, pos_is_a):
    """Return ``(R_lead, run, npos)`` for the packed pair tile, or ``None``.

    The pair tile needs (i) the *position* operand to be fully materialised and
    contiguous (its rows are copied as whole contiguous blocks), and (ii) the
    broadcast operand to vary only over a *suffix* of the coalesced outer dims,
    laid out compactly -- then the flat block index ``lead * run + ch`` gives
    the broadcast row directly (``block % run``).
    """
    cshape = tuple(int(x) for x in cshape)
    npos = cshape[-1]
    natural = _compact_strides(cshape)
    pos_strides, pos_numel = (a_strides, a_numel) if pos_is_a else (b_strides, b_numel)
    bc_strides, bc_numel = (b_strides, b_numel) if pos_is_a else (a_strides, a_numel)
    total = 1
    for dim in cshape:
        total *= dim
    if tuple(int(s) for s in pos_strides) != tuple(natural) or pos_numel != total:
        return None
    run_dims = [i for i in range(len(cshape) - 1) if int(bc_strides[i]) != 0]
    if not run_dims:
        # fully broadcast operand (scalar): a single [low, high] row.
        if bc_numel != 1:
            return None
        return total // npos, 1, npos
    i0 = len(cshape) - 1 - len(run_dims)
    if run_dims != list(range(i0, len(cshape) - 1)):
        return None  # non-suffix broadcast pattern -> keep the int64 path
    run = 1
    for dim in cshape[i0:-1]:
        run *= dim
    if bc_numel != run:
        return None
    # broadcast operand must be compact over the run dims (b[ch] == flat row ch)
    compact = _compact_strides(cshape[i0:-1])
    if [int(bc_strides[i]) for i in run_dims] != compact:
        return None
    return total // (npos * run), run, npos


def _packed_shapes_i64(out_shape, a_shape, b_shape):
    """Return the int64 -> int32 packing plan, or ``None`` to keep int64.

    Packing applies when *all* of the following hold; anything else keeps the
    existing int64 path (correct, and for these shapes measurably better):

    * the output's last participating dim ``L`` is even -- a conservative gate
      that keeps the low/high pairing trivially aligned (odd ``L`` falls back);
    * both operands are int64 (checked again at run time);
    * exactly one operand is *unit* on the last dim, i.e. channel / scalar
      broadcast, and :func:`_pair_geometry` accepts the block structure.

    A *full* last dim on both sides (same shape / row broadcast) is
    deliberately NOT packed: measured on round 8, the packed int32 run of
    ``hidden-state-prefill-int64`` is 128.5 us vs 126.4 us for the int64 path
    (block_size 4096 and 7168 alike -- the 2x word count halves the per-task
    tile, and above a 128 MB footprint the stream rate drops from 2.05 to
    1.57 TB/s), while the shared L2 read-miss counter drops from 147 MB to
    134 MB, i.e. the reported roofline ratio falls 61.4 % -> 58.0 %.
    ``row-broadcast-int64`` is the same story (167.9 us vs 168.9 us, ratio
    59.2 % -> 44.4 %: the word-space 2-D row tile is UB-capped at r_blk=2).
    """
    if not out_shape:
        return None
    last = int(out_shape[-1])
    if last < _PACK_WORDS_I64 or last % _PACK_WORDS_I64 != 0:
        return None
    ndim = len(out_shape)
    padded = []
    for shape in (a_shape, b_shape):
        p = (1,) * (ndim - len(shape)) + tuple(int(d) for d in shape)
        if p[-1] not in (1, last):
            return None  # pragma: no cover - broadcast_shapes forbids this
        padded.append(p)
    a_full = padded[0][-1] == last
    b_full = padded[1][-1] == last
    if a_full == b_full:
        return None  # both full (see docstring) or both unit: no position side
    pos_is_a = a_full
    out_c, cshape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
    geom = _pair_geometry(
        cshape,
        a_strides,
        b_strides,
        math.prod(padded[0]),
        math.prod(padded[1]),
        pos_is_a,
    )
    if geom is None:
        return None
    r_lead, run, npos = geom
    fold = _pair_fold(npos)
    if not fold:
        return None
    return {
        "form": "pair",
        "geom": geom,
        "pos_is_a": pos_is_a,
        "fold": fold,
        "width": fold * _PACK_WORDS_I64,
    }


class BitwiseOrFwdKernel(Kernel):
    """Element-wise bitwise OR with broadcast: ``y = a | b``.

    Integer dtype path (GPU ``BitwiseOrFwdKernel(BinaryKernel)``). The
    GPU broadcast lowering (``coalesce_broadcast_dims``) is preserved:
    the kernel receives the coalesced output shape plus per-operand
    strides (0 for broadcast dims) and computes offsets inside the
    TileLang body.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K11: ``default_config`` returns ``{"block_size": 2048}``
      (the GPU ``threads * num_per_thread`` product).
    - K8/K9: autotune removed; ``init_config(config)`` takes no ``tune``.
    - K7/K10: no custom_op wrapper; ``forward`` dispatches directly.

    Args:
        a_shape: Shape of input ``a``.
        b_shape: Shape of input ``b`` (broadcast against ``a_shape``).
        dtype: Torch dtype (one of the manifest bitwise dtypes).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel (CG-2026-0010).
    ascend_mode = "Developer"

    SUPPORTED_DTYPES = _BITWISE_DTYPES

    # K11: block_size = GPU ``threads * num_per_thread`` product.
    _DEFAULT_BLOCK_SIZE = _DEFAULT_BLOCK_SIZE

    def __init__(
        self,
        a_shape,
        b_shape,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        # Broadcast lowering is the kernel's own choice (GPU parity).
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            self.a_shape,
            self.b_shape,
        )
        self.out_shape = out_shape
        self.N_total = math.prod(out_shape)
        self.a_numel = math.prod(self.a_shape) if self.a_shape else 1
        self.b_numel = math.prod(self.b_shape) if self.b_shape else 1
        self.dtype = dtype
        # npuir-side kernel dtype: uint8/bool storage is carried as int8
        # (CG-2026-0018).  ``self.dtype`` stays the public op dtype for the
        # dtype contract / config surface.
        self.kernel_dtype_str = _kernel_dtype_str(dtype)
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for
        # target="npuir" so its returned callable takes a single
        # ``block_size`` (K11).
        self.kernel = bitwise_or_fwd_kernel(
            self.N_total,
            self.kernel_dtype_str,
            self.coalesced_shape,
            self.a_strides,
            self.b_strides,
            self.a_numel,
            self.b_numel,
        )
        self._compiled_fn = None
        self._compiled_block_size: Optional[int] = None
        # Stage 4 round 7 (v3): host-side int32 packing plan for the 1-byte
        # path.  ``None`` means "not applicable -> keep the int8 carrier".
        self.pack_plan = (
            _packed_shapes(self.out_shape, self.a_shape, self.b_shape)
            if self.kernel_dtype_str == "int8"
            else None
        )
        self._pack_kernel = None
        self._pack_compiled_fn = None
        if self.pack_plan is not None:
            a_packed, b_packed, self._a_full_last, self._b_full_last = self.pack_plan
            pack_out, pack_cshape, pack_a_strides, pack_b_strides = coalesce_broadcast_dims(
                a_packed, b_packed
            )
            self._pack_kernel = bitwise_or_fwd_kernel(
                math.prod(pack_out),
                "int32",
                pack_cshape,
                pack_a_strides,
                pack_b_strides,
                math.prod(a_packed),
                math.prod(b_packed),
            )
        # Stage 4 round 8 (v4): host-side int64 -> int32 packing plan.
        # ``None`` means "not applicable -> keep the int64 kernel".
        self.pack_plan_i64 = (
            _packed_shapes_i64(self.out_shape, self.a_shape, self.b_shape)
            if self.kernel_dtype_str == "int64"
            else None
        )
        self._pack64_kernel = None
        self._pack64_compiled_fn = None
        self._pack64_swap = False
        self._pack64_pair = None
        if self.pack_plan_i64 is not None:
            plan = self.pack_plan_i64
            r_lead, run, npos = plan["geom"]
            words = _PACK_WORDS_I64
            width = plan["width"]
            self._pack64_swap = not plan["pos_is_a"]
            self._pack64_pair = (run, plan["fold"])
            # Word-space geometry of the pair tile: a/y are word rows of
            # ``width`` words (= ``fold`` element positions, contiguous in
            # memory), b is (run, width) with the per-channel [low, high]
            # pattern repeated ``fold`` times.
            self._pack64_kernel = bitwise_or_fwd_kernel(
                r_lead * run * npos * words,
                "int32",
                (r_lead, run, npos, width),
                (run * npos * words, npos * words, words, 1),
                (0, 0, 0, 1),
                r_lead * run * npos * words,
                run * width,
                pack_words=words,
            )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: ``block_size = threads * npt``)."""
        return {"block_size": self._DEFAULT_BLOCK_SIZE}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the bitwise-or kernel on two flat contiguous 1-D tensors.

        K11: the (NPU-reimplemented) factory callable takes a single
        ``block_size`` argument and is cached per block_size.

        Stage 4 round 7 (v3): if the packing plan applies to the *runtime*
        tensors (both contiguous, ``a.numel() % 4 == 0``) the 1-byte operands
        are regrouped into int32 words on the host and the int32 kernel runs
        instead; the int32 result is then viewed back to the 1-byte carrier so
        the frozen Op contract (same dtype in/out) is unchanged.

        Stage 4 round 8 (v4): same for int64 -> two int32 words.  Both views
        are pure host metadata (no copy, no GM/UB reinterpretation); the
        result is viewed back to int64 before returning.
        """
        if self.pack_plan_i64 is not None and self._pack64_eligible(a, b):
            if self._pack64_compiled_fn is None:
                self._pack64_compiled_fn = self._pack64_kernel(TUNED_PACK_BLOCK_SIZE)
            # The pair tile needs the fully materialised operand on the
            # position side; the three bitwise ops commute, so swapping the
            # argument order is bit-identical.
            pos, bc = (b, a) if self._pack64_swap else (a, b)
            pos_w = pos.view(torch.int32)  # (..., 2L): [low, high] word pairs
            if self._pack64_pair is None:
                bc_w = bc.view(torch.int32)
            else:
                run, fold = self._pack64_pair
                bc_w = _expand_pair_row(bc, run, fold)
            return self._pack64_compiled_fn(pos_w, bc_w).view(torch.int64)
        if self.pack_plan is not None and self._pack_eligible(a, b):
            if self._pack_compiled_fn is None:
                self._pack_compiled_fn = self._pack_kernel(TUNED_PACK_BLOCK_SIZE)
            packed = self._pack_compiled_fn(
                _pack_packed_operand(a, self._a_full_last),
                _pack_packed_operand(b, self._b_full_last),
            )
            # Present the int32 words back as the 1-byte carrier (same byte
            # count, bit-identical) so `_restore_public_dtype` / the Op layer
            # keep seeing their expected dtype.
            return _restore_public_dtype(packed.view(torch.int8), self.dtype)
        block_size = self.config["block_size"]
        if self._compiled_fn is None or self._compiled_block_size != block_size:
            self._compiled_fn = self.kernel(block_size)
            self._compiled_block_size = block_size
        # Carry uint8/bool storage as int8 for the npuir kernel (R3), then
        # present the int8 carrier back as the public op dtype so the frozen
        # Op's same-dtype-in/out contract holds (R3).
        out = self._compiled_fn(_as_kernel_input(a), _as_kernel_input(b))
        return _restore_public_dtype(out, self.dtype)

    def _pack_eligible(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Runtime preconditions of the packed path (never raises).

        ``view(dtype)`` on a 1-byte buffer requires the source to be
        contiguous and its element count to be a multiple of 4; when that is
        not observable at build time the int8 path stays in charge.  There is
        deliberately no try/except: a silent fallback on a malformed view
        would hide a real layout bug.
        """
        if a.dtype.itemsize != 1 or b.dtype.itemsize != 1:
            return False
        if not (a.is_contiguous() and b.is_contiguous()):
            return False
        if a.numel() != self.a_numel or b.numel() != self.b_numel:
            return False
        if self._a_full_last and a.numel() % _PACK_WORDS != 0:
            return False
        return not (self._b_full_last and b.numel() % _PACK_WORDS != 0)

    def _pack64_eligible(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Runtime preconditions of the int64 -> int32 packed path.

        ``view(torch.int32)`` on an int64 buffer is a pure metadata split, but
        it is only defined for a contiguous buffer, and it reinterprets bytes,
        so it is only bit-exact on a little-endian host (the word order is
        [low, high]).  Everything else (numel, dtype) must match the geometry
        the plan was built for; otherwise the int64 kernel stays in charge.
        There is deliberately no try/except: a silent fallback on a malformed
        view would hide a real layout bug.
        """
        if a.dtype != torch.int64 or b.dtype != torch.int64:
            return False
        if not (a.is_contiguous() and b.is_contiguous()):
            return False
        if a.numel() != self.a_numel or b.numel() != self.b_numel:
            return False
        return sys.byteorder == "little"


class BitwiseOrBoolStorageFwdKernel(BitwiseOrFwdKernel):
    """Element-wise bitwise OR on uint8-backed bool storage.

    GPU ``BitwiseOrBoolStorageFwdKernel(_Uint8StorageBinaryKernel)``: the
    public ``torch.bool`` tensors are viewed as ``uint8`` by the Op layer
    (TileLang cannot lower vectorised ``boolx<N>`` loads on the CUDA
    target).  On NPU the two kernel variants share the same extracted
    ``bitwise_or_fwd_kernel``; this class narrows the dtype contract and
    keeps the GPU bool-storage default ``block_size``.  The Op layer's
    frozen ``view(torch.uint8)`` is honoured at the kernel boundary and
    reinterpreted as ``int8`` (bit-identical; npuir cannot carry uint8 GM
    tensors).
    """

    SUPPORTED_DTYPES = (torch.uint8,)
    _DEFAULT_BLOCK_SIZE = _BOOL_STORAGE_DEFAULT_BLOCK_SIZE
