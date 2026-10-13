"""Discover and parse optional OpenCode session timing Markdown artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

_ARTIFACT_NAME = "SESSION_TIMING_ANALYSIS.md"
_BLOCK = re.compile(r"<!--\s*TILEOPS_SESSION_TIMING_V1\s*(\{.*?\})\s*-->", re.DOTALL)


def _snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _operator_slugs(root: Path, operator_catalog: list[dict[str, Any]]) -> dict[str, str]:
    slugs: dict[str, str] = {}
    for meta_path in root.glob("tileops/kernels/**/.migration_meta.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = metadata.get("op_name")
        slug = metadata.get("op_slug")
        if isinstance(name, str) and isinstance(slug, str):
            slugs[name] = slug

    for entry in operator_catalog:
        name = entry.get("name")
        kernel = entry.get("kernel")
        if not isinstance(name, str) or name in slugs or not isinstance(kernel, str):
            continue
        parent = PurePosixPath(kernel.replace("\\", "/")).parent.name
        if parent:
            slugs[name] = parent
    return slugs


def _finite_nonnegative(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def parse_session_timing(path: str | Path, operator: str) -> dict[str, Any]:
    """Parse one validated timing block, returning an N/A record on failure."""
    source = Path(path)
    base = {
        "operator": operator,
        "source": str(source),
        "status": "missing",
        "total_duration_s": None,
        "breakdown": [],
    }
    if not source.is_file():
        return {**base, "reason": f"{_ARTIFACT_NAME} not found"}

    try:
        text = source.read_text(encoding="utf-8")
        match = _BLOCK.search(text)
        if match is None:
            raise ValueError("TILEOPS_SESSION_TIMING_V1 block not found")
        payload = json.loads(match.group(1))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported session timing schema")
        if payload.get("operator") != operator:
            raise ValueError(
                f"operator mismatch: expected {operator!r}, got {payload.get('operator')!r}"
            )
        total = _finite_nonnegative(
            payload.get("total_duration_s"), "total_duration_s", nullable=True
        )
        raw_breakdown = payload.get("breakdown")
        if not isinstance(raw_breakdown, list):
            raise ValueError("breakdown must be a list")
        breakdown = []
        for index, item in enumerate(raw_breakdown):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError(f"breakdown[{index}] must have a string name")
            name = item["name"].strip()
            if not name:
                raise ValueError(f"breakdown[{index}].name must not be empty")
            detail = item.get("detail", "")
            if not isinstance(detail, str):
                raise ValueError(f"breakdown[{index}].detail must be a string")
            breakdown.append(
                {
                    "name": name,
                    "duration_s": _finite_nonnegative(
                        item.get("duration_s"), f"breakdown[{index}].duration_s"
                    ),
                    "ratio_percent": _finite_nonnegative(
                        item.get("ratio_percent"), f"breakdown[{index}].ratio_percent"
                    ),
                    "detail": detail,
                }
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {**base, "status": "invalid", "reason": str(exc)}

    return {
        **base,
        "status": "available" if total is not None else "unavailable",
        "total_duration_s": total,
        "breakdown": breakdown,
        "reason": None if total is not None else "total duration is N/A",
    }


def attach_session_timing(
    run: dict[str, Any],
    *,
    root: str | Path,
    operator_catalog: list[dict[str, Any]],
    enabled: bool,
) -> dict[str, Any]:
    """Attach per-operator timing records when the opt-in report feature is enabled."""
    if not enabled:
        run.pop("session_timing", None)
        return run

    project_root = Path(root).resolve()
    examples_root = project_root.parent
    slugs = _operator_slugs(project_root, operator_catalog)
    records = {}
    for operator in run.get("operators", []):
        name = operator.get("operator")
        if not isinstance(name, str):
            continue
        slug = slugs.get(name, _snake_case(re.sub(r"Op$", "", name)))
        records[name] = parse_session_timing(examples_root / slug / _ARTIFACT_NAME, name)
    run["session_timing"] = {"enabled": True, "operators": records}
    return run


__all__ = ["attach_session_timing", "parse_session_timing"]
