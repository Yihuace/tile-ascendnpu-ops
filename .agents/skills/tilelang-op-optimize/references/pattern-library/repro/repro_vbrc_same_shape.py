# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""TRAP-vbrc-same-shape-empty-broadcast: T.vbrc with identical src/dst shapes
fails MLIR verification ('empty broadcast dims array').First verified: 2026-09-28, tilelang 0.1.2 (dev root build 2026-09-24) + CANN 8.5.0 / Ascend910B2C (argmax-_argreduce_kernel-stage4-20260928).
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
import tilelang
import tilelang.language as T
import torch_npu  # noqa: F401


def build():
    @tilelang.jit(out_idx=[], target="npuir")
    def _func():
        @T.prim_func
        def main(a: T.Tensor((1, 512), "float32"), b: T.Tensor((1, 512), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, _):
                src = T.alloc_fragment((1, 512), "float32")
                dst = T.alloc_fragment((1, 512), "float32")
                T.copy(a, src)
                T.vbrc(src, dst)  # same shape -> empty broadcast dims
                T.copy(dst, b)

        return main

    return _func()


def main():
    import tempfile
    import subprocess
    import sys
    import os as _os

    # The canonical diagnostic goes to stderr (MLIR pipeline), so capture it.
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as f:
        errpath = f.name
    r = subprocess.run(
        [sys.executable, __file__, "--inner"],
        capture_output=True,
        text=True,
        cwd=_os.path.dirname(__file__),
    )
    err = ""
    if _os.path.exists(errpath):
        with open(errpath, encoding="utf-8") as errfile:
            err = errfile.read()
    _os.unlink(errpath)
    diag = r.stderr + err
    assert r.returncode != 0, "expected compile failure did not trigger"
    assert "empty broadcast dims" in diag, f"unexpected diagnostic: {diag[:200]}"
    print("ASSERT PASS: TRAP-vbrc-same-shape-empty-broadcast reproduced")
    print(
        "  stderr diagnostic:",
        [l for l in diag.splitlines() if "empty broadcast" in l][0][:100],
    )


def inner():
    try:
        build()
        raise AssertionError("expected MLIR verify failure did not trigger")
    except Exception:  # noqa: BLE001
        raise


if __name__ == "__main__":
    import sys

    if "--inner" in sys.argv:
        inner()
    else:
        main()
