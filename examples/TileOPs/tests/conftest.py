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
