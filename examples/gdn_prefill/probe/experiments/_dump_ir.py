"""Run a probe while preserving the transient tilelang temp dirs (for IR inspection).

Usage: python3 _dump_ir.py <probe.py>
Preserved copies land in /tmp/opencode/ir_dump/dump_N/.
"""

import itertools
import os
import runpy
import shutil
import sys
import tempfile

OUT = "/tmp/opencode/ir_dump"
os.makedirs(OUT, exist_ok=True)
_counter = itertools.count()
_orig = tempfile.TemporaryDirectory


class _KeepDir(_orig):
    def __exit__(self, exc_type, exc, tb):
        try:
            dst = os.path.join(OUT, f"dump_{next(_counter)}")
            shutil.copytree(self.name, dst, dirs_exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dump] copy failed: {exc}")
        return super().__exit__(exc_type, exc, tb)


tempfile.TemporaryDirectory = _KeepDir

probe = sys.argv[1]
runpy.run_path(probe, run_name="__main__")
