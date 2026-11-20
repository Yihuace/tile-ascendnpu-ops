#!/usr/bin/env python3
"""Check which npuir toolchain binaries tilelang actually resolves.

Usage:
    source /path/to/cann/set_env.sh
    export PATH=/path/to/custom/bin:$PATH
    python3 team/knowledge/tools/check_toolchain_path.py \
        --expect-prefix /path/to/custom

Exit codes:
    0  all resolved binaries are under --expect-prefix (or no prefix given)
    1  tilelang resolved a binary outside --expect-prefix
    2  tilelang import/resolution failed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BINARIES = ("bishengir-compile", "bishengir-opt", "bisheng")
ENV_KEYS = ("ASCEND_HOME_PATH", "ASCEND_HOME", "TILELANG_NPU_COMPILER_PATH")


def which_all(name: str) -> list[str]:
    """All executable matches of name on PATH, in PATH order."""
    found: list[str] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            found.append(str(candidate))
    return found


def resolve_with_tilelang() -> dict[str, str]:
    from tilelang.utils.npu_utils import (
        get_bisheng_path,
        get_npucompiler_opt_path,
        get_npucompiler_path,
    )

    return {
        "bishengir-compile": str(get_npucompiler_path()),
        "bishengir-opt": str(get_npucompiler_opt_path()),
        "bisheng": str(get_bisheng_path()),
    }


def under_prefix(path: str, prefix: Path) -> bool:
    candidate = Path(path).resolve()
    try:
        return candidate.is_relative_to(prefix)
    except AttributeError:  # Python < 3.9 fallback
        return prefix == candidate or prefix in candidate.parents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-prefix",
        default="",
        help="optional directory that all resolved binaries must be under",
    )
    args = parser.parse_args()

    print("== environment ==")
    for key in ENV_KEYS:
        print(f"{key} = {os.environ.get(key, '')}")
    print(f"PATH = {os.environ.get('PATH', '')}")

    print("\n== which -a ==")
    for name in BINARIES:
        matches = which_all(name)
        print(f"{name}:")
        if matches:
            for path in matches:
                print(f"  {path}")
        else:
            print("  <not found>")

    print("\n== tilelang resolved ==")
    try:
        resolved = resolve_with_tilelang()
    except Exception as exc:  # noqa: BLE001 - diagnostics: report any failure
        print(f"tilelang resolution failed: {exc!r}")
        return 2

    for name in BINARIES:
        print(f"{name}: {resolved[name]}")

    if not args.expect_prefix:
        print("\nOK (no --expect-prefix given; only printed paths)")
        return 0

    prefix = Path(args.expect_prefix).resolve()
    mismatched = [
        name for name, path in resolved.items() if not under_prefix(path, prefix)
    ]
    if mismatched:
        print(f"\nMISMATCH: {', '.join(mismatched)} not under {prefix}")
        return 1

    print(f"\nOK: all resolved binaries are under {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
