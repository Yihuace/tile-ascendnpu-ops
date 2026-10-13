#!/usr/bin/env python3
"""Generate {GPU_REPO}/.migration_source_map.json for the src-layout upstream.

Machine-readable version of the migration task book §5 table. Fields:
  op -> {op_class, op, kernel, kernel_map, test, bench, workload}
"""
import ast
import json
import sys
from pathlib import Path

import yaml

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/TileOPs-main").resolve()
SRC = REPO / "src" / "tileops"
MANIFEST_DIR = SRC / "manifest" / "spec"


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def op_class_kernel_types():
    """Map op class name -> (file_rel, {slug: KernelClassName})."""
    out = {}
    for f in (SRC / "ops").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.ClassDef):
                continue
            for s in n.body:
                tgt = None
                if isinstance(s, ast.Assign) and len(s.targets) == 1:
                    tgt = getattr(s.targets[0], "id", None)
                elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                    tgt = s.target.id  # e.g. ``kernel_types: ClassVar[...] = {...}``
                if tgt != "kernel_types" or not isinstance(s.value, ast.Dict):
                    continue
                kmap = {}
                for k, v in zip(s.value.keys, s.value.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Name):
                        kmap[k.value] = v.id
                if kmap:
                    out[n.name] = (rel(f), kmap)
    return out


def kernel_class_files():
    """Map kernel class name -> kernel file rel."""
    out = {}
    for f in (SRC / "kernels").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                out.setdefault(n.name, rel(f))
    return out


def manifest_entries():
    merged = {}
    for p in sorted(MANIFEST_DIR.glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        for name, entry in data.items():
            merged[name] = (rel(p), entry)
    return merged


def slug_of(op_name: str) -> str:
    # CamelCase -> snake_case
    s = []
    for i, c in enumerate(op_name):
        if c.isupper() and i and (not op_name[i - 1].isupper()):
            s.append("_")
        s.append(c.lower())
    return "".join(s)


def main():
    okt = op_class_kernel_types()
    kcf = kernel_class_files()
    entries = manifest_entries()

    smap = {}
    for op_name, (mfile, entry) in sorted(entries.items()):
        if not isinstance(entry, dict):
            continue
        if op_name not in okt:
            continue
        op_file, kmap = okt[op_name]
        kernel_file = None
        for kclass in kmap.values():
            if kclass in kcf:
                kernel_file = kcf[kclass]
                break
        slug = slug_of(op_name)
        test = f"tests/ops/test_{slug}.py"
        bench = f"benchmarks/ops/bench_{slug}.py"
        smap[op_name] = {
            "op_class": op_name,
            "op": op_file,
            "kernel": kernel_file,
            "kernel_map": kmap,
            "manifest": mfile,
            "test": test if (REPO / test).exists() else None,
            "bench": bench if (REPO / bench).exists() else None,
            "workload": entry.get("workloads"),
        }

    out = REPO / ".migration_source_map.json"
    out.write_text(json.dumps(smap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(smap)} ops")
    for k in ("HardsigmoidFwdOp", "HardswishFwdOp", "ClampScalarFwdOp", "Conv1dFwdOp", "GLARecurrentFwdOp", "GDNFwdOp"):
        print(k, "->", json.dumps(smap.get(k), ensure_ascii=False))


if __name__ == "__main__":
    main()
