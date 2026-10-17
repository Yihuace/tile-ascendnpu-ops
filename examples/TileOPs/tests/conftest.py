import pytest
import torch

from tileops.device import get_device_backend
from tileops.testing.test_base import get_check_result


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    backend = get_device_backend()
    backend.manual_seed_all(1235)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    cr = get_check_result()
    if cr.op_name:
        item.user_properties.append(("op", cr.op_name))
        if cr.op_module:
            item.user_properties.append(("op_module", cr.op_module))
        if cr.max_abs_err is not None:
            item.user_properties.append(("max_abs_err", f"{cr.max_abs_err:.2e}"))
        cr.op_name = None
        cr.op_module = None
        cr.max_abs_err = None


# ---------------------------------------------------------------------------
# Toolchain-bound expected failures (CG-2026-0003)
# ---------------------------------------------------------------------------
# Some cases fail for *toolchain* reasons, not op bugs.  Marking them ``xfail``
# ONLY on the affected toolchain keeps the suite green while keeping the defect
# visible; a future fixed toolchain turns them into XPASS (auto-alerting us to
# drop the marker).  Not an unconditional skip -- the marker self-expires.
#
# CG-2026-0003: bishengir-compile pass ``ConvertHFusionToHIVM`` calls
#   utils::getAnnotateOpWithAttr(op.getResult(0), ...)
# on buffer-form IR.  ``hivm.hir.mmadL1 ins(...) outs(%m : memref<...>)`` carries
# 0 results, so ``getResult(0)`` is out of bounds -> SIGSEGV (some shapes spin
# in a CPU loop instead of crashing).  Affects Cube IR only; Vector ops are fine.
# Mainline deleted this walk but the fix was NOT back-ported to the release
# branch shipped with CANN 9.0.0 (bishengir-compile 1.1.0 / 428ab8fdab46).
# Evidence: docs/cann9_adaptation.md sections 9.3 / 11.1
#           (knowledge base entry CG-2026-0003).

_CG_0003_TOOLCHAIN = "9.0.0"

_CG_0003_REASON = (
    "CG-2026-0003: bishengir-compile ConvertHFusionToHIVM getResult(0) OOB on "
    "buffer-form hivm.hir.mmadL1 (Cube IR) -> SIGSEGV; CANN 9.0.0 toolchain "
    "defect, not an op bug. See docs/cann9_adaptation.md 9.3/11.1"
)

_CG_0003_NODE_SUFFIXES = (
    "test_conv1d.py::test_conv1d_same_padding_even_kernel_matches_torch[bias]",
    "test_conv1d_bias.py::test_conv1d_bias[smoke-tcn-k3-s1-fp16]",
    "test_conv1d_bias.py::test_conv1d_bias[smoke-tcn-k3-s1-bf16]",
    "test_conv1d_bias.py::test_conv1d_bias[full-seanet-residual-k3-s1-fp16]",
    "test_conv1d_bias.py::test_conv1d_bias[full-groups2-k3-fp16]",
    "test_conv1d_bias.py::test_conv1d_bias_same_padding_even_kernel_matches_torch",
)


def _cann_version():
    """CANN toolkit version string, or ``None`` when it cannot be determined."""
    try:
        from tileops.reporting.setup_info import _cann_version as _cv

        return _cv()
    except Exception:
        return None


def pytest_collection_modifyitems(config, items):
    """Mark the CG-2026-0003 cases xfail, but only on the affected toolchain."""
    if _cann_version() != _CG_0003_TOOLCHAIN:
        return
    for item in items:
        if any(item.nodeid.endswith(suffix) for suffix in _CG_0003_NODE_SUFFIXES):
            item.add_marker(pytest.mark.xfail(reason=_CG_0003_REASON, strict=False))
