"""Shared base class for reduce ops (NPU).

Ported from GPU TileOPs ``tileops/ops/reduction/reduce.py`` —
``_ReduceOpBase`` only (the concrete simple/Welford reduce ops belong to
their own migration units and are not included here).

Adaptation from GPU (TileOPs) to NPU:

- ``x.is_cuda`` device check → ``backend.is_device_tensor(x)`` (O1) in
  ``_validate_input_tensor`` and ``_maybe_scalar``.
- ``tune`` parameter removed (O3); the kernel constructor is called
  without ``tune`` and takes ``(M, N, op_kind, dtype, config=None,
  device_index=None)``.
- ``_kernel_cls`` default is ``None`` (the GPU version defaulted to
  ``ReduceKernel``, which has not been migrated to NPU); subclasses
  must override ``_kernel_cls``.
- Op-specific flow (validate → transpose → reshape → pad → dispatch →
  reshape) preserved unchanged (O6).
"""

from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.ops._multidim import (
    EmptyDimPolicy,
    flatten_for_multidim,
    normalize_dim,
    restore_multidim_shape,
)
from tileops.ops.op_base import Op

# Op kinds that accept 0-D (scalar) input. The kernel path assumes
# ``ndim >= 1`` (and the Welford kernel's Bessel correction is undefined for
# ``N == 1``), so the Op layer computes the scalar result directly without
# invoking PyTorch's reduction ops. Mapping a degenerate single-element
# reduction to its closed-form result is pure arithmetic, not a fallback.
# Note: argmax/argmin are deliberately absent — a 0-D input falls through
# to the kernel path, which raises the pre-existing
# ``ValueError("Input tensor must be at least 1D")``.
_SCALAR_REDUCE_KINDS = frozenset(
    {
        "sum",
        "mean",
        "amin",
        "amax",
        "prod",
        "std",
        "var",
        "var_mean",
        "all",
        "any",
        "count_nonzero",
    }
)

__all__ = ["_ReduceOpBase"]


class _ReduceOpBase(Op):
    """Common base for all reduce ops (simple, Welford, argreduce, logical, vector_norm).

    Consolidates shared init params (dtype, dim, keepdim), initializes
    and owns an internal kernel cache, and handles input preparation
    (validate, transpose, reshape to 2D, pad) and output reshaping.
    Subclasses declare ``_op_kind``, ``_kernel_key``, ``_kernel_cls``, and
    override hooks as needed.  ``forward()`` is provided by this base class;
    only ops with non-standard returns (e.g. ``VarMeanFwdOp``) need to
    override it.

    Hooks for subclass customization:

    - ``_kernel_key``: kernel map key (default ``"reduce"``).
    - ``_kernel_cls``: kernel class (must be overridden on NPU).
    - ``_validate_dim()``: validate ``dim`` at init (default: accept int/list/None).
    - ``_pad_value()``: identity element for alignment padding (default ``0.0``).
    - ``_build_kernel_kwargs()``: extra kwargs for kernel constructor.
    - ``_pre_kernel(x)``: transform 2D input before kernel call (default identity).
      Returns ``(x, context)`` where *context* is passed to ``_post_kernel``.
    - ``_post_kernel(y, context)``: transform kernel output (default identity).
    """

    _op_kind: str = ""  # overridden by subclasses
    _kernel_key: str = "reduce"  # overridden by subclasses for different kernel families
    _kernel_cls: type = None  # overridden by subclasses (NPU: no ReduceKernel yet)
    _kernel_handles_padding: bool = False  # True when kernel accepts (M, N) with masked loads
    _empty_dim_policy: EmptyDimPolicy = "reject"

    def __init__(
        self,
        dtype: torch.dtype,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        """Construct a reduce op.

        Args:
            dtype: Input data type.
            dim: Reduction dimension (default ``None``, i.e. full reduction).
                Accepts ``int``, ``list[int]``, ``tuple[int, ...]``, or
                ``None``.
            keepdim: Whether to retain reduced dims as size 1.
            kernel_map: Optional override for kernel dispatch.
        """
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._validate_dim()
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}
        self._last_roofline_mn: tuple[int, int] | None = None

    # Dim validation (subclasses may override)

    def _validate_dim(self) -> None:
        """Validate the ``dim`` parameter.

        Default: accept ``int``, ``list[int]``/``tuple[int]``, or ``None``.
        Subclasses that only support single-dim reduction (e.g. argreduce)
        should override to reject non-scalar values.

        ``bool`` values are rejected explicitly. Python's ``bool`` subclasses
        ``int`` (so ``isinstance(True, int)`` is true), but a boolean dim has
        no meaningful interpretation as a tensor axis and almost always
        signals a caller bug.
        """
        dim = self.dim
        if isinstance(dim, bool):
            raise TypeError(
                f"dim must not be bool (subclasses int but is not a valid axis), got {dim!r}"
            )
        if dim is None or isinstance(dim, int):
            return
        if isinstance(dim, (list, tuple)):
            for d in dim:
                if isinstance(d, bool) or not isinstance(d, int):
                    raise TypeError(f"All elements of dim must be int (not bool), got {dim!r}")
            return
        raise TypeError(
            f"dim must be int, list[int], tuple[int, ...], or None, got {type(dim).__name__}"
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    # Pad value (subclasses may override; used only when
    # _kernel_handles_padding is False)

    def _pad_value(self) -> float:
        """Return the identity element used when padding to alignment.

        Only used when ``_kernel_handles_padding`` is ``False`` (i.e. the
        kernel expects pre-padded input from the Op layer).
        """
        return 0.0

    # Extra kernel kwargs (subclasses may override)

    def _build_kernel_kwargs(self) -> dict:
        """Return extra keyword arguments for the kernel constructor.

        Override in subclasses to pass additional params like ``correction``.
        """
        return {}

    # Pre/post kernel hooks (subclasses may override)

    def _pre_kernel(self, x: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Transform 2D input before kernel call.

        Returns ``(x, context)`` where *context* is an opaque value
        passed through to ``_post_kernel``.  Default: identity.
        """
        return x, None

    def _post_kernel(self, y: torch.Tensor, context: object) -> torch.Tensor:
        """Transform kernel output.  Default: identity."""
        return y

    # Forward (subclasses with non-standard returns, e.g. VarMeanFwdOp,
    # must override)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reduce op on *x* along the configured dim."""
        scalar_out = self._maybe_scalar(x)
        if scalar_out is not None:
            return scalar_out
        noop_out = self._maybe_noop(x)
        if noop_out is not None:
            return noop_out
        x, orig_shape, dim_info, kernel = self._prepare_input(x)
        x, ctx = self._pre_kernel(x)
        y = kernel(x)
        y = self._post_kernel(y, ctx)
        return self._reshape_output(y, orig_shape, dim_info)

    # Empty-dim no-op short-circuit

    def _noop_output_dtype(self) -> Optional[torch.dtype]:
        """Manifest-declared output dtype for the dtype-altering short-circuits.

        Consulted by both the empty-dim no-op path (``_maybe_noop``) and
        the scalar 0-D path (``_scalar_forward``) so the manifest output
        dtype contract is honored without dispatching to the kernel.
        Subclasses with a fixed output dtype (e.g. All/Any -> bool,
        CountNonzero -> int64) MUST override. The default ``None`` means
        "preserve input dtype".
        """
        return None

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate device, dtype, and rank of the forward input.

        Shared by ``_prepare_input`` and the ``dim=[]`` noop short-circuit
        so both paths enforce the same forward contract.

        NPU adaptation (O1): device check uses ``backend.is_device_tensor(x)``
        instead of ``x.is_cuda``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # Scalar (0-D) input fast path

    def _validate_scalar_dim(self) -> None:
        """Validate that ``self.dim`` is an accepted form for a 0-D input.

        PyTorch accepts ``None``, ``0``, ``-1``, ``()``, and ``[]`` on a
        0-D tensor, plus singleton list/tuple forms (``[0]``, ``(0,)``,
        ``[-1]``, ``(-1,)``). Integers outside ``{0, -1}`` raise
        ``IndexError``. Multi-entry sequences whose canonical dims
        collide (``0`` and ``-1`` both alias axis ``0`` on a 0-D tensor)
        raise ``RuntimeError`` to match PyTorch's
        ``"dim 0 appears multiple times in the list of dims"``.
        """
        dim = self.dim
        if dim is None:
            return
        if isinstance(dim, int):
            if dim not in (0, -1):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
                )
            return
        if isinstance(dim, (list, tuple)):
            seen: set = set()
            for d in dim:
                if d not in (0, -1):
                    raise IndexError(
                        f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                    )
                canon = 0  # 0 and -1 alias the same axis on a 0-D tensor.
                if canon in seen:
                    raise RuntimeError(f"dim {canon} appears multiple times in the list of dims")
                seen.add(canon)
            return

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward result for a 0-D input natively.

        Single-element reductions are degenerate: every arithmetic family
        collapses to the input value, the logical families collapse to
        ``x != 0`` cast to the manifest output dtype, and the Welford
        family follows a closed form in ``correction``. This method
        computes the closed-form result directly so the kernel path
        (undefined for ``N == 1``) is bypassed without delegating to
        PyTorch's reduction ops.

        Arithmetic reductions (``sum``, ``mean``, ``amin``, ``amax``,
        ``prod``) over one element return the element itself. Logical /
        count ops override ``_noop_output_dtype`` so this default applies
        the ``x != 0`` predicate and casts to the declared output dtype.
        Welford ops (``std``, ``var``, ``var_mean``) override this hook
        because their result depends on ``correction``.
        """
        out_dtype = self._noop_output_dtype()
        if out_dtype is None:
            return x.clone()
        return (x != 0).to(out_dtype)

    def _maybe_scalar(self, x: torch.Tensor):
        """Short-circuit a 0-D input to the native scalar forward.

        Returns the scalar-path output when ``x.ndim == 0``; returns
        ``None`` otherwise so the caller proceeds with the kernel path.
        The roofline state is bound to ``(1, 1)`` so ``eval_roofline()``
        after a scalar forward stays well-defined.
        """
        if x.ndim != 0:
            return None
        if self._op_kind not in _SCALAR_REDUCE_KINDS:
            # Subclasses without a defined 0-D contract (e.g. argmax/argmin/
            # l1/l2/inf) fall through to the kernel path, which raises the
            # pre-existing ``ValueError("Input tensor must be at least 1D")``.
            return None
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"x must be a {backend.name} tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        self._validate_scalar_dim()
        self._last_roofline_mn = (1, 1)
        return self._scalar_forward(x)

    def _maybe_noop(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return *x* (cast to the manifest output dtype) when ``dim`` is
        an empty list/tuple and the op's ``_empty_dim_policy`` is
        ``"noop"``; return ``None`` otherwise so the caller proceeds with
        the normal kernel path.

        Runs the same input validation as ``_prepare_input`` (device /
        dtype / ndim) and binds ``_last_roofline_mn`` before
        short-circuiting, so the noop path still honors the public forward
        contract -- bad inputs raise, and ``eval_roofline()`` works after a
        noop forward.
        """
        if self._empty_dim_policy != "noop":
            return None
        if not isinstance(self.dim, (list, tuple)) or len(self.dim) != 0:
            return None
        self._validate_input_tensor(x)
        # Bind roofline state. The noop performs no reduction but still
        # reads every input element and writes an equal-shape result
        # (cast to bool for All/Any, the only ops whose ``_empty_dim_policy``
        # is ``"noop"``; other reduce ops, including ``CountNonzero``, keep
        # ``"full"`` and never enter this branch). Model this as a
        # degenerate reduction over an axis of length 1: M = numel, N = 1.
        # Under the existing per-op-kind
        # formulas this yields mem_bytes proportional to numel * elem_bytes
        # for the read plus the output term, instead of collapsing to
        # zero, which would under-count the actual data-movement cost.
        self._last_roofline_mn = (x.numel(), 1)
        out_dtype = self._noop_output_dtype()
        if out_dtype is None:
            return x
        return x.to(out_dtype)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        op_kind = self._op_kind

        if op_kind == "mean":
            flops = M * (N + 1)
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "std":
            flops = 5 * M * N + M
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "var":
            flops = 5 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "var_mean":
            flops = 5 * M * N
            mem_bytes = (M * N + 2 * M) * elem_bytes
        elif op_kind in {"argmax", "argmin"}:
            flops = M * N
            mem_bytes = M * N * elem_bytes + M * 8
        elif op_kind in {"all", "any"}:
            flops = M * N
            mem_bytes = M * N * elem_bytes + M
        elif op_kind == "count_nonzero":
            flops = 2 * M * N
            mem_bytes = M * N * elem_bytes + M * 8
        elif op_kind == "l1":
            flops = 2 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "l2":
            flops = 2 * M * N + M
            mem_bytes = (M * N + M) * elem_bytes
        elif op_kind == "inf":
            flops = 2 * M * N
            mem_bytes = (M * N + M) * elem_bytes
        else:
            flops = M * N
            mem_bytes = (M * N + M) * elem_bytes

        return flops, mem_bytes

    # Kernel cache

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed.

        NPU adaptation (O3): the kernel constructor takes
        ``(M, N, op_kind, dtype, config=None, device_index=None)`` — no
        ``tune`` argument.
        """
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M,
                N,
                self._op_kind,
                self.dtype,
                **self._build_kernel_kwargs(),
            )
            # Lift the per-shape msprof anchor (Kernel-level declaration,
            # e.g. multi-kernel dispatch pointing at its dominant kernel)
            # onto the Op instance: tier-2 resolution in
            # tileops.benchmark.msprof reads ``functor.msprof_kernel_name``
            # (this Op exposes kernels via ``_kernel_cache``, not ``.kernel``).
            _kname = getattr(self._kernel_cache[key], "msprof_kernel_name", None)
            if isinstance(_kname, str) and _kname:
                self.msprof_kernel_name = _kname
        return self._kernel_cache[key]

    # Input preparation (validate → transpose → reshape → pad)

    def _prepare_input(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Size, object, object]:
        """Validate, derive M/N, transpose, reshape to 2D, optionally pad.

        Returns ``(x_2d, orig_shape, dim_info, kernel)`` where
        *dim_info* is either an ``int`` (single-dim) or ``list[int]``
        (multi-dim).

        When ``_kernel_handles_padding`` is ``True``, the raw ``(M, N)``
        tensor is passed through -- the kernel handles alignment internally
        via masked loads.  Otherwise, host-side ``F.pad`` is applied for
        backward compatibility with kernels that expect ``(M, N_padded)``.
        """
        self._validate_input_tensor(x)

        orig_shape = x.shape

        # --- multi-dim path (includes dim=None for full reduction) ---
        if isinstance(self.dim, (list, tuple)) or self.dim is None:
            dims = normalize_dim(
                self.dim,
                x.ndim,
                empty_dim_policy=self._empty_dim_policy,
            )
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            self._last_roofline_mn = (M, N)
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N)
            if not self._kernel_handles_padding:
                N_padded = align_up(N, DEFAULT_ALIGNMENT)
                if N_padded != N:
                    pv = self._pad_value()
                    pad = (0, N_padded - N)
                    x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)
            return x, orig_shape, dims, kernel

        # --- single-dim path ---
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)
        self._last_roofline_mn = (M, N)

        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        kernel = self._get_or_create_kernel(M, N)

        if not self._kernel_handles_padding:
            N_padded = align_up(N, DEFAULT_ALIGNMENT)
            if N_padded != N:
                pv = self._pad_value()
                pad = (0, N_padded - N)
                x = F.pad(x, pad) if pv == 0.0 else F.pad(x, pad, value=pv)

        return x, orig_shape, dim, kernel

    # Output reshape

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim_info: Union[int, List[int]],
    ) -> torch.Tensor:
        """Reshape (M,) kernel output to match keepdim setting.

        *dim_info* is either an ``int`` (single-dim) or ``list[int]``
        (multi-dim).
        """
        if isinstance(dim_info, list):
            return restore_multidim_shape(y, orig_shape, dim_info, self.keepdim)

        dim = dim_info
        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            return y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            return y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)
