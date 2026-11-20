"""Extract a self-contained TileLang kernel implementation for a TileOPs op.

Reads the op's manifest entry to locate the declared kernel source file and
kernel class, then recursively inlines every same-module and intra-package
dependency (base classes, ``@tilelang.jit``/``@T.prim_func`` builder
functions, constants, helper functions, sub-kernels) into a single standalone
Python file.

External imports (``tilelang``, ``torch``, ``typing``, ``functools`` ...) are
preserved verbatim. Internal imports (``from tileops... import`` / relative
``from .x import``) are replaced by the inlined definitions so the emitted
file has zero dependency on the ``tileops`` package.

Usage:
    python scripts/extract_tl_kernel.py --op-name MishFwdOp \\
        --gpu-repo-root /path/to/TileOPs [--out OUT.py]

If ``--out`` is omitted the result is written to ``extracted_<OpName>.py`` in
the current working directory.
"""

from __future__ import annotations

import argparse
import ast
import copy
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Manifest loading (parse YAML directly; never import the tileops package)
# --------------------------------------------------------------------------- #


def load_manifest_entry(op_name: str, repo_root: Path) -> dict[str, Any]:
    """Return the merged manifest entry for *op_name*.

    Scans every ``tileops/manifest/*.yaml`` file and merges them, mirroring
    :func:`tileops.manifest.load_manifest` without importing the package.
    """
    manifest_dir = repo_root / "tileops" / "manifest"
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"manifest directory not found: {manifest_dir}")
    merged: dict[str, Any] = {}
    for path in sorted(manifest_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        ops = yaml.safe_load(text) or {}
        if not isinstance(ops, dict):
            raise ValueError(f"{path.name}: top-level YAML must be a mapping")
        for name, entry in ops.items():
            if name in merged:
                raise ValueError(f"duplicate op {name!r} across manifest files")
            merged[name] = entry
    if op_name not in merged:
        raise KeyError(f"op '{op_name}' not found in manifest under {manifest_dir}")
    entry = merged[op_name]
    if not isinstance(entry, dict) or "source" not in entry:
        raise ValueError(f"op '{op_name}' has no 'source' section in manifest")
    return entry


# --------------------------------------------------------------------------- #
# Target kernel class name resolution
# --------------------------------------------------------------------------- #


def resolve_kernel_class_names(entry: dict[str, Any], op_name: str, repo_root: Path) -> list[str]:
    """Return the kernel class name(s) declared for this op.

    Resolution order (prefers the op's own ``default_kernel_map`` because it
    may declare more kernels than the manifest's ``source.kernel_map``):

      1. ``default_kernel_map`` property in ``source.op`` (AST-parsed),
         walking the class MRO and resolving ``self._kernel_class`` style
         indirect references.
      2. ``kernel_cls`` class attribute in ``source.op`` (AST-parsed).
      3. ``source.kernel_map`` values (manifest-declared, fallback).
    """
    source = entry["source"]

    op_rel = source.get("op")
    if not op_rel:
        # Fallback to manifest kernel_map if no op source file.
        kmap = source.get("kernel_map")
        if isinstance(kmap, dict) and kmap:
            names = [v for v in kmap.values() if isinstance(v, str)]
            if names:
                return names
        raise ValueError(f"op '{op_name}': no 'source.op' and no 'source.kernel_map'")

    repo_root = repo_root.resolve()
    op_file = repo_root / op_rel
    if not op_file.exists():
        raise FileNotFoundError(f"op source file not found: {op_file}")

    loader = _Loader(repo_root)
    mro = _walk_mro(loader, op_file, op_name)
    if not mro:
        raise ValueError(f"class {op_name} not found in {op_file}")

    # 1. default_kernel_map (walk MRO, resolve self._kernel_class).
    method_loc = _find_method_in_mro(mro, "default_kernel_map")
    if method_loc is not None:
        _m_file, m_node = method_loc
        names = _names_from_default_kernel_map(m_node, mro)
        if names:
            return names

    # 2. kernel_cls class attribute.
    for _f, cls in mro:
        attr = _class_attr_name_value(cls, "kernel_cls")
        if attr:
            return [attr]

    # 3. Manifest kernel_map (fallback).
    kmap = source.get("kernel_map")
    if isinstance(kmap, dict) and kmap:
        names = [v for v in kmap.values() if isinstance(v, str)]
        if names:
            return names

    raise ValueError(
        f"could not determine kernel class for '{op_name}': define "
        f"default_kernel_map/kernel_cls in {op_file} or add 'source.kernel_map' "
        f"to the manifest"
    )


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _names_from_returned_dict(method: ast.FunctionDef) -> list[str]:
    """Collect ``ast.Name`` ids from the values of a returned dict literal."""
    names: list[str] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for v in node.value.values:
                if isinstance(v, ast.Name) and v.id not in names:
                    names.append(v.id)
    return names


def _resolve_self_attr_in_mro(mro: list[tuple[Path, ast.ClassDef]], attr_name: str) -> str | None:
    """Resolve ``self.<attr_name>`` to a class attribute value via the MRO.

    Looks for ``<attr_name> = <Name>`` class-level assignments in the MRO
    (most-derived first) and returns the Name's id. Used to resolve
    ``self._kernel_class`` → the actual kernel class name.
    """
    for _file, cls in mro:
        attr = _class_attr_name_value(cls, attr_name)
        if attr:
            return attr
    return None


def _names_from_default_kernel_map(
    method: ast.FunctionDef, mro: list[tuple[Path, ast.ClassDef]]
) -> list[str]:
    """Extract kernel class names from a ``default_kernel_map`` method.

    Handles two patterns:
      1. ``return {"k": KernelClass, ...}`` — literal dict with Name values.
      2. ``return {self._kernel_key: self._kernel_class}`` — indirect via
         ``self.*`` class attributes; resolves ``_kernel_class`` through the
         MRO.
    """
    for node in ast.walk(method):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        names: list[str] = []
        for v in node.value.values:
            if isinstance(v, ast.Name) and v.id not in names:
                names.append(v.id)
            elif (
                isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
                and v.value.id == "self"
            ):
                # self._kernel_class → resolve via MRO class attribute.
                resolved = _resolve_self_attr_in_mro(mro, v.attr)
                if resolved and resolved not in names:
                    names.append(resolved)
        return names
    return []


def _class_attr_name_value(cls: ast.ClassDef, attr_name: str) -> str | None:
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == attr_name
                    and isinstance(node.value, ast.Name)
                ):
                    return node.value.id
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == attr_name
            and isinstance(node.value, ast.Name)
        ):
            return node.value.id
    return None


# --------------------------------------------------------------------------- #
# Per-file symbol table
# --------------------------------------------------------------------------- #


@dataclass
class ImportInfo:
    bound_name: str
    original_name: str
    is_external: bool
    source_file: Path | None
    segment: str


@dataclass
class SymbolTable:
    path: Path
    source: str
    tree: ast.Module
    defs: dict[str, ast.AST] = field(default_factory=dict)
    imports: dict[str, ImportInfo] = field(default_factory=dict)
    # class_name -> {method_name: FunctionDef}; lets us look up e.g.
    # MishFwdKernel.op_func which is not a top-level def.
    class_methods: dict[str, dict[str, ast.FunctionDef]] = field(default_factory=dict)


class _Loader:
    """Parses Python files into :class:`SymbolTable` objects and resolves
    internal import source files."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._cache: dict[Path, SymbolTable] = {}

    def load(self, path: Path) -> SymbolTable:
        path = path.resolve()
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(f"kernel source file not found: {path}")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        st = SymbolTable(path=path, source=source, tree=tree)
        self._index(st)
        self._cache[path] = st
        return st

    def _index(self, st: SymbolTable) -> None:
        for node in st.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                st.defs[node.name] = node
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        st.defs[tgt.id] = node
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    st.defs[node.target.id] = node
            elif isinstance(node, ast.Import):
                self._index_import(st, node)
            elif isinstance(node, ast.ImportFrom):
                self._index_import_from(st, node)
            # Index class methods so we can resolve e.g. MishFwdKernel.op_func.
            if isinstance(node, ast.ClassDef):
                methods: dict[str, ast.FunctionDef] = {}
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[item.name] = item
                st.class_methods[node.name] = methods

    def _index_import(self, st: SymbolTable, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            module = alias.name
            is_external = not module.startswith("tileops")
            src = None if is_external else self._resolve_abs(module)
            st.imports[bound] = ImportInfo(
                bound_name=bound,
                original_name=bound,
                is_external=is_external,
                source_file=src,
                segment=ast.get_source_segment(st.source, node),
            )

    def _index_import_from(self, st: SymbolTable, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            if node.level and node.level > 0:
                is_external = False
                src = self._resolve_relative(st.path, node)
            else:
                module = node.module or ""
                is_external = not module.startswith("tileops")
                src = None if is_external else self._resolve_abs(module)
            st.imports[bound] = ImportInfo(
                bound_name=bound,
                original_name=alias.name,
                is_external=is_external,
                source_file=src,
                segment=ast.get_source_segment(st.source, node),
            )

    def _resolve_abs(self, module: str) -> Path | None:
        parts = module.split(".")
        py = self.repo_root.joinpath(*parts).with_suffix(".py")
        if py.exists():
            return py
        init = self.repo_root.joinpath(*parts) / "__init__.py"
        if init.exists():
            return init
        return None

    def _resolve_relative(self, file_path: Path, node: ast.ImportFrom) -> Path | None:
        rel = file_path.relative_to(self.repo_root)
        parts = list(rel.with_suffix("").parts)
        is_init = rel.name == "__init__.py"
        containing = parts if is_init else parts[:-1]
        drop = (node.level - 1) if node.level and node.level > 1 else 0
        base = containing[: len(containing) - drop] if drop > 0 else list(containing)
        if node.module:
            base = base + node.module.split(".")
        py = self.repo_root.joinpath(*base).with_suffix(".py")
        if py.exists():
            return py
        init = self.repo_root.joinpath(*base) / "__init__.py"
        if init.exists():
            return init
        return None


# --------------------------------------------------------------------------- #
# Reference collection
# --------------------------------------------------------------------------- #


def _collect_refs(node: ast.AST) -> set[str]:
    """Return the set of ``ast.Name`` ids referenced anywhere under *node*."""
    refs: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            root = n
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                refs.add(root.id)
    return refs


# --------------------------------------------------------------------------- #
# MRO walking + kernel-member resolution (Pattern A)
# --------------------------------------------------------------------------- #
#
# When a kernel class inherits its TileLang kernel from a base (e.g.
# ``MishFwdKernel`` → ``UnaryKernel`` which sets ``self.kernel =
# self._build_kernel(self.strategy)``), the actual TileLang kernel is the
# *value* of the ``self.kernel`` member, not a directly-referenced module
# function. Pattern A statically resolves that member:
#
#   1. Walk the kernel class MRO (across files, following imports).
#   2. Find the ``self.kernel = <expr>`` assignment.
#   3. If it calls ``self.<method>(...)`` (e.g. ``self._build_kernel``),
#      parse that method's ``if strategy == "X": return <builder>(...)``
#      branches, pick the branch matching the class's ``DEFAULT_STRATEGY``,
#      and extract that builder (a ``@tilelang.jit`` function).
#   4. Extract the ``op_func`` static method on the kernel class — the
#      per-op pointwise math passed into the builder.
#
# When no ``self.kernel`` assignment exists (e.g. ``GatedDeltaNetFwdKernel``
# calls ``@tilelang.jit`` builders directly in ``forward``), Pattern A
# returns None and the extractor falls back to Pattern B (discover all
# reachable ``@tilelang.jit`` functions).


def _resolve_name_location(loader: _Loader, file: Path, name: str) -> tuple[Path, ast.AST] | None:
    """Return (file, node) where *name* is defined, following imports."""
    st = loader.load(file)
    if name in st.defs:
        return (st.path, st.defs[name])
    if name in st.imports:
        imp = st.imports[name]
        if imp.source_file is not None:
            return _resolve_name_location(loader, imp.source_file, imp.original_name)
    return None


def _walk_mro(
    loader: _Loader, kernel_file: Path, class_name: str
) -> list[tuple[Path, ast.ClassDef]]:
    """Return [(file, ClassDef)] for the class and its bases, MRO order."""
    seen: set[tuple[Path, str]] = set()
    result: list[tuple[Path, ast.ClassDef]] = []
    stack: list[tuple[Path, str]] = [(kernel_file, class_name)]
    while stack:
        file, name = stack.pop(0)
        key = (file.resolve(), name)
        if key in seen:
            continue
        seen.add(key)
        loc = _resolve_name_location(loader, file, name)
        if loc is None or not isinstance(loc[1], ast.ClassDef):
            continue
        loc_file, cls_node = loc
        result.append((loc_file, cls_node))
        for base in cls_node.bases:
            if isinstance(base, ast.Name):
                stack.append((loc_file, base.id))
    return result


def _find_self_kernel_assignment(
    mro: list[tuple[Path, ast.ClassDef]],
) -> tuple[Path, ast.FunctionDef, ast.Assign] | None:
    """Find ``self.kernel = <expr>`` in any method across the MRO."""
    for file, cls in mro:
        for item in cls.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(item):
                if not isinstance(n, ast.Assign) or len(n.targets) != 1:
                    continue
                tgt = n.targets[0]
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr == "kernel"
                ):
                    return (file, item, n)
    return None


def _find_method_in_mro(
    mro: list[tuple[Path, ast.ClassDef]], name: str
) -> tuple[Path, ast.FunctionDef] | None:
    for file, cls in mro:
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                return (file, item)
    return None


def _extract_strategy_str(test: ast.expr) -> str | None:
    """Return the string literal from ``<strategy> == "X"`` (or reversed)."""
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        for left, right in ((test.left, test.comparators[0]), (test.comparators[0], test.left)):
            if not (isinstance(left, ast.Constant) and isinstance(left.value, str)):
                continue
            ok = (isinstance(right, ast.Name) and right.id == "strategy") or (
                isinstance(right, ast.Attribute)
                and right.attr == "strategy"
                and isinstance(right.value, ast.Name)
                and right.value.id == "self"
            )
            if ok:
                return left.value
    return None


def _parse_strategy_branches(method: ast.FunctionDef) -> dict[str, str]:
    """Parse ``if strategy == "X": return <builder>(...)`` → {strategy: builder}."""
    branches: dict[str, str] = {}
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        strategy = _extract_strategy_str(node.test)
        if strategy is None:
            continue
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Return)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
            ):
                branches[strategy] = stmt.value.func.id
    return branches


def _direct_return_calls(method: ast.FunctionDef) -> list[str]:
    """Return function names from bare ``return <name>(...)`` statements."""
    names: list[str] = []
    for n in ast.walk(method):
        if (
            isinstance(n, ast.Return)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
        ):
            names.append(n.value.func.id)
    return list(dict.fromkeys(names))


def _resolve_default_strategy(mro: list[tuple[Path, ast.ClassDef]]) -> str | None:
    """Find the ``DEFAULT_STRATEGY = "..."`` class attr in MRO (most-derived)."""
    for _file, cls in mro:
        for item in cls.body:
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            tgt = item.targets[0]
            if (
                isinstance(tgt, ast.Name)
                and tgt.id == "DEFAULT_STRATEGY"
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                return item.value.value
    return None


def _find_op_func_in_mro(
    mro: list[tuple[Path, ast.ClassDef]],
) -> tuple[Path, ast.FunctionDef] | None:
    """Find the ``op_func`` static method on the kernel class or its bases."""
    for file, cls in mro:
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "op_func":
                return (file, item)
    return None


# --------------------------------------------------------------------------- #
# op_func inlining (merge builder + op_func into one function)
# --------------------------------------------------------------------------- #


def _op_name_to_snake(op_name: str) -> str:
    """Convert an op name to a snake_case kernel name.

    ``MishFwdOp`` → ``mish_fwd``, ``GatedDeltaNetFwdOp`` → ``gated_deltanet_fwd``.
    The trailing ``Op`` suffix is stripped, then PascalCase is converted to
    snake_case.
    """
    name = op_name[:-2] if op_name.endswith("Op") else op_name
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _contains_call_to(node: ast.AST, name: str) -> bool:
    """Return True if *node* contains a ``name(...)`` call."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name:
            return True
    return False


def _inline_op_func(
    builder: ast.FunctionDef,
    op_func: ast.FunctionDef,
    merged_name: str,
) -> ast.FunctionDef:
    """Inline *op_func*'s body into *builder*, rename, and return the merged AST.

    The ``op_func`` parameter is removed from the builder's signature. Every
    ``op_func(arg)`` call inside the builder is replaced by the return
    expression of *op_func* with its parameter substituted by *arg*. Any
    local-variable declarations in *op_func* (statements before the return)
    are hoisted into the builder just before the first call site.

    The resulting AST is suitable for :func:`ast.unparse`.
    """
    # Parse op_func: param name, local declarations, return expression.
    params = [a.arg for a in op_func.args.args]
    if not params:
        raise ValueError("op_func must have at least one parameter")
    param_name = params[0]
    body = list(op_func.body)
    return_stmt = body[-1]
    if not isinstance(return_stmt, ast.Return) or return_stmt.value is None:
        raise ValueError("op_func must end with a return statement")
    local_decls = body[:-1]
    return_expr = return_stmt.value

    # Clone the builder so we don't mutate the original AST.
    merged = copy.deepcopy(builder)

    # Remove the op_func parameter from the builder's signature.
    merged.args.args = [a for a in merged.args.args if a.arg != "op_func"]

    # Rename.
    merged.name = merged_name

    # Recursive statement-list processor: replaces op_func(arg) calls and
    # hoists local declarations before the first call site.
    hoisted = {"done": False}

    def _substitute_param(expr: ast.AST, arg: ast.AST) -> ast.AST:
        """Clone *expr* and replace every Name(param_name) with a clone of *arg*."""
        cloned = copy.deepcopy(expr)

        class _Subst(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == param_name:
                    return copy.deepcopy(arg)
                return node

        return _Subst().visit(cloned)

    def _replace_calls(node: ast.AST) -> ast.AST:
        """Replace op_func(arg) calls with the inlined return expression."""

        class _Replacer(ast.NodeTransformer):
            def visit_Call(self, c: ast.Call) -> ast.AST:
                self.generic_visit(c)
                if isinstance(c.func, ast.Name) and c.func.id == "op_func" and len(c.args) >= 1:
                    return ast.copy_location(_substitute_param(return_expr, c.args[0]), c)
                return c

        return _Replacer().visit(node)

    def _process_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            if _contains_call_to(stmt, "op_func"):
                # Hoist local declarations once, before the first call site.
                if not hoisted["done"] and local_decls:
                    for decl in local_decls:
                        result.append(copy.deepcopy(decl))
                    hoisted["done"] = True
                stmt = _replace_calls(stmt)
            # Recurse into nested statement lists.
            for field_name in ("body", "orelse", "finalbody"):
                val = getattr(stmt, field_name, None)
                if isinstance(val, list):
                    setattr(stmt, field_name, _process_stmts(val))
            if hasattr(stmt, "handlers") and isinstance(stmt.handlers, list):
                for handler in stmt.handlers:
                    if isinstance(getattr(handler, "body", None), list):
                        handler.body = _process_stmts(handler.body)
            result.append(stmt)
        return result

    merged.body = _process_stmts(merged.body)
    ast.fix_missing_locations(merged)
    return merged


@dataclass
class EmitTarget:
    """A definition to emit in the extracted file."""

    file: Path
    node: ast.AST
    kind: str  # "tl_kernel" (a @tilelang.jit builder) or "op_func" (per-op math)
    # When set, the node has been AST-transformed (e.g. op_func inlined) and
    # must be emitted via ast.unparse instead of source-segment extraction.
    merged_ast: ast.AST | None = None


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


def _has_tilelang_jit(node: ast.AST) -> bool:
    """Return True if *node* contains a function decorated with ``@tilelang.jit``.

    A TileLang kernel function is a builder whose body defines an inner
    function decorated with ``tilelang.jit`` (e.g. ``_h_recurrence_tl``
    returns ``_func`` which is ``@tilelang.jit``-decorated). This helper
    walks the entire subtree so both the outer builder and a directly
    decorated top-level function are recognised.
    """
    for n in ast.walk(node):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in n.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "tilelang"
                and target.attr == "jit"
            ):
                return True
    return False


# Base class names that are considered "root" (not intermediate). A kernel
# class that inherits ONLY from these is treated as Pattern B.
_ROOT_BASE_NAMES = frozenset({"Kernel", "ABC", "object"})


def _has_intermediate_base(loader: _Loader, kernel_file: Path, kernel_class_name: str) -> bool:
    """Return True if the kernel class inherits from a non-root base class.

    Pattern A (op_func inlining) is only used when the kernel class has an
    intermediate base — e.g. ``MishFwdKernel(FloatUnaryKernel)`` inherits
    from ``FloatUnaryKernel``, which is not a root name. A class like
    ``GatedDeltaNetFwdKernel(Kernel)`` inherits directly from ``Kernel`` and
    is handled by Pattern B.
    """
    st = loader.load(kernel_file)
    cls_node = st.defs.get(kernel_class_name)
    if not isinstance(cls_node, ast.ClassDef):
        return False
    for base in cls_node.bases:
        if isinstance(base, ast.Name) and base.id not in _ROOT_BASE_NAMES:
            return True
    return False


def _is_module_const(node: ast.AST) -> bool:
    """Return True for a top-level assignment (module-level constant)."""
    return isinstance(node, (ast.Assign, ast.AnnAssign))


class Extractor:
    """Extract the TileLang kernel implementation for an op.

    Two resolution patterns are supported:

    * **Pattern A (kernel member)** — the kernel class inherits its TileLang
      kernel from a base that sets ``self.kernel = self._build_kernel(...)``
      (e.g. ``MishFwdKernel``). The extractor statically resolves the
      ``self.kernel`` member to the specific ``@tilelang.jit`` builder
      selected by ``DEFAULT_STRATEGY``, and emits that builder plus the
      per-op ``op_func`` static method passed into it.

    * **Pattern B (direct call)** — the kernel class's ``forward`` calls
      ``@tilelang.jit`` builders directly (e.g. ``GatedDeltaNetFwdKernel``
      calls ``_h_recurrence_tl`` / ``_output_o_tl`` /
      ``fused_prepare_compute_w_u_tl``). The extractor discovers all
      reachable definitions and emits those containing ``@tilelang.jit``.

    In both cases the output is a minimal, self-contained file: the
    TileLang kernel function(s), the module-level constants they reference,
    and the external imports (tilelang / torch / stdlib) they need. The
    ``Kernel`` subclass, wrapper functions, ``register_fake`` hooks and
    other helpers are intentionally *not* emitted.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.loader = _Loader(repo_root)
        # Discovery results (insertion-ordered for deterministic output).
        self.discovered: dict[tuple[Path, str], ast.AST] = {}
        self.discovered_order: list[tuple[Path, str]] = []
        self._visiting: set[tuple[Path, str]] = set()
        self.warnings: list[str] = []
        self.pattern: str = ""  # "A" or "B", for reporting.
        self.targets: list[EmitTarget] = []  # emitted targets, for reporting.

    # -- entry point ------------------------------------------------------ #

    def extract(self, kernel_file: Path, kernel_class_names: list[str], op_name: str) -> str:
        self.targets = self._resolve_targets(kernel_file, kernel_class_names, op_name)
        targets = self.targets

        # Always discover the original builder (loads its file into cache so
        # imports/constants are available), even when op_func has been inlined
        # into a merged AST. The merged AST is only used for emission.
        for t in targets:
            if t.kind == "tl_kernel":
                self._discover(t.file, t.node.name)

        # Collect names referenced by every emitted target. For merged
        # targets, use the merged AST (it has the inlined op_func body and
        # may reference constants/imports not in the original builder).
        needed_refs: set[str] = set()
        for t in targets:
            ref_node = t.merged_ast if t.merged_ast is not None else t.node
            needed_refs |= _collect_refs(ref_node)
        const_keys = self._collect_constants(needed_refs)

        # Emitted names: use the merged function name when merged, else the
        # original node name.
        emitted_names: set[str] = set()
        for t in targets:
            if t.merged_ast is not None:
                emitted_names.add(t.merged_ast.name)
            else:
                emitted_names.add(t.node.name)
        emitted_names |= {k[1] for k in const_keys}
        import_texts, _ = self._collect_imports(needed_refs | emitted_names)

        return self._assemble(targets, const_keys, import_texts)

    # -- target resolution ------------------------------------------------ #

    def _resolve_targets(
        self, kernel_file: Path, kernel_class_names: list[str], op_name: str
    ) -> list[EmitTarget]:
        # Pattern A (op_func inlining) only applies when the kernel class
        # inherits from an intermediate base class (e.g. MishFwdKernel →
        # FloatUnaryKernel → UnaryKernel → Kernel). A class that inherits
        # directly from Kernel (e.g. GatedDeltaNetFwdKernel) always uses
        # Pattern B.
        #
        # Multi-kernel ops (e.g. Conv2dFwdOp with Conv2dKernel +
        # Conv2d1x1Kernel + GroupConv2dKernel) always use Pattern B: each
        # kernel class is discovered independently and its @tilelang.jit
        # builders are collected.
        if len(kernel_class_names) == 1 and _has_intermediate_base(
            self.loader, kernel_file, kernel_class_names[0]
        ):
            pa = self._pattern_a(kernel_file, kernel_class_names[0], op_name)
            if pa is not None:
                self.pattern = "A"
                return pa
        self.pattern = "B"
        return self._pattern_b(kernel_file, kernel_class_names)

    def _pattern_a(
        self, kernel_file: Path, kernel_class_name: str, op_name: str
    ) -> list[EmitTarget] | None:
        """Resolve the ``self.kernel`` member to its builder + op_func, then
        inline op_func into the builder to produce a single merged function."""
        mro = _walk_mro(self.loader, kernel_file, kernel_class_name)
        if not mro:
            return None
        assignment = _find_self_kernel_assignment(mro)
        if assignment is None:
            return None
        _file, _method, assign_node = assignment
        rhs = assign_node.value
        builder_names = self._resolve_builder_names(mro, rhs)
        if not builder_names:
            return None

        tl_targets: list[EmitTarget] = []
        for name in builder_names:
            loc = _resolve_name_location(self.loader, kernel_file, name)
            if loc is None:
                self.warnings.append(f"Pattern A: could not locate builder '{name}'; skipping.")
                continue
            loc_file, loc_node = loc
            if isinstance(loc_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _has_tilelang_jit(
                loc_node
            ):
                tl_targets.append(EmitTarget(loc_file, loc_node, "tl_kernel"))

        op_func_loc = _find_op_func_in_mro(mro)

        if not tl_targets:
            return None

        # If we have both a builder and an op_func, inline op_func into the
        # builder to produce a single merged function named after the op.
        if op_func_loc is not None and len(tl_targets) == 1:
            builder_node = tl_targets[0].node
            op_func_node = op_func_loc[1]
            merged_name = f"{_op_name_to_snake(op_name)}_kernel"
            merged_ast = _inline_op_func(builder_node, op_func_node, merged_name)
            # Keep the original builder node (for discovery/imports) and set
            # merged_ast for emission.
            return [
                EmitTarget(
                    tl_targets[0].file,
                    builder_node,
                    "tl_kernel",
                    merged_ast=merged_ast,
                )
            ]

        # Fall back: emit builder(s) and op_func separately.
        targets = list(tl_targets)
        if op_func_loc is not None:
            targets.append(EmitTarget(op_func_loc[0], op_func_loc[1], "op_func"))
        return targets

    def _resolve_builder_names(
        self, mro: list[tuple[Path, ast.ClassDef]], rhs: ast.expr
    ) -> list[str]:
        """Map the RHS of ``self.kernel = <rhs>`` to builder function names."""
        # self.<method>(...)  e.g. self._build_kernel(self.strategy)
        if (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Attribute)
            and isinstance(rhs.func.value, ast.Name)
            and rhs.func.value.id == "self"
        ):
            method_name = rhs.func.attr
            method_loc = _find_method_in_mro(mro, method_name)
            if method_loc is None:
                return []
            _m_file, m_node = method_loc
            branches = _parse_strategy_branches(m_node)
            if branches:
                strategy = _resolve_default_strategy(mro)
                if strategy and strategy in branches:
                    return [branches[strategy]]
                # Strategy not statically resolvable: keep all branches.
                return list(dict.fromkeys(branches.values()))
            return _direct_return_calls(m_node)
        # Direct call <func_name>(...)
        if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
            return [rhs.func.id]
        return []

    def _pattern_b(self, kernel_file: Path, kernel_class_names: list[str]) -> list[EmitTarget]:
        """Pattern B: discover from each kernel class, keep ``@tilelang.jit`` funcs.

        Handles both single-kernel ops (e.g. GatedDeltaNetFwdKernel) and
        multi-kernel ops (e.g. Conv2dFwdOp with Conv2dKernel +
        Conv2d1x1Kernel + GroupConv2dKernel). Each kernel class is discovered
        independently; all reachable ``@tilelang.jit`` builders are
        collected, de-duplicated by (file, name).
        """
        for name in kernel_class_names:
            self._discover(kernel_file, name)
        targets: list[EmitTarget] = []
        seen: set[tuple[Path, str]] = set()
        for key in self.discovered_order:
            node = self.discovered[key]
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _has_tilelang_jit(node)
                and key not in seen
            ):
                seen.add(key)
                targets.append(EmitTarget(key[0], node, "tl_kernel"))
        return targets

    # -- dependency discovery (shared) ------------------------------------ #

    def _discover(self, file: Path, name: str) -> None:
        key = (file.resolve(), name)
        if key in self.discovered or key in self._visiting:
            return
        st = self.loader.load(file)
        if name in st.defs:
            self._visiting.add(key)
            local = set(st.defs.keys()) | set(st.imports.keys())
            for ref in sorted(_collect_refs(st.defs[name]) & local):
                self._discover(file, ref)
            self._visiting.discard(key)
            if key not in self.discovered:
                self.discovered[key] = st.defs[name]
                self.discovered_order.append(key)
        elif name in st.imports:
            imp = st.imports[name]
            if imp.is_external:
                return
            if imp.source_file is not None:
                self._discover(imp.source_file, imp.original_name)
            else:
                self.warnings.append(
                    f"could not resolve internal import '{name}' in {file}; "
                    "leaving reference unresolved"
                )
        # else: builtin / parameter / local — ignore silently.

    def _collect_constants(self, needed: set[str]) -> list[tuple[Path, str]]:
        """Return discovery-order keys of module-level constants whose name is
        in *needed*, expanding *needed* transitively as new constants are
        added."""
        const_keys: list[tuple[Path, str]] = []
        seen_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for key in self.discovered_order:
                node = self.discovered[key]
                name = key[1]
                if name in seen_names or name not in needed:
                    continue
                if not _is_module_const(node):
                    continue
                seen_names.add(name)
                const_keys.append(key)
                needed |= _collect_refs(node)
                changed = True
        return const_keys

    def _collect_imports(self, needed: set[str]) -> tuple[list[str], list[tuple[Path, str]]]:
        """Return (ordered import texts, source keys) for external imports
        whose bound name is in *needed*."""
        texts: list[str] = []
        seen_text: set[str] = set()
        keys: list[tuple[Path, str]] = []
        for (file, _), imp in self._iter_imports():
            if imp.bound_name not in needed or not imp.is_external:
                continue
            text = imp.segment.strip()
            if text in seen_text:
                continue
            seen_text.add(text)
            texts.append(text)
            keys.append((file, imp.bound_name))
        return texts, keys

    def _iter_imports(self):
        for key in self.discovered_order:
            st = self.loader._cache[key[0]]
            for _bound, imp in st.imports.items():
                yield key, imp

    # -- assembly --------------------------------------------------------- #

    def _emit_node(self, target: EmitTarget) -> str:
        # Merged AST (op_func inlined into builder): emit via ast.unparse.
        if target.merged_ast is not None:
            return ast.unparse(target.merged_ast)
        st = self.loader.load(target.file)
        if target.kind == "op_func":
            # Extract the method's full source lines manually. Unlike
            # ast.get_source_segment (which starts the first line at
            # col_offset and so breaks textwrap.dedent for indented class
            # methods), this preserves the real leading whitespace so dedent
            # can normalise the method to module level.
            node = target.node
            lines = st.source.splitlines()
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            seg_lines = [
                ln
                for ln in seg.splitlines()
                if not ln.lstrip().startswith(("@staticmethod", "@classmethod"))
            ]
            return textwrap.dedent("\n".join(seg_lines))
        return ast.get_source_segment(st.source, target.node)

    def _assemble(
        self,
        targets: list[EmitTarget],
        const_keys: list[tuple[Path, str]],
        import_texts: list[str],
    ) -> str:
        tl_targets = [t for t in targets if t.kind == "tl_kernel"]
        op_targets = [t for t in targets if t.kind == "op_func"]
        # Use the merged function name when merged_ast is set.
        tl_names = [
            (t.merged_ast.name if t.merged_ast is not None else t.node.name) for t in tl_targets
        ]
        op_names = [t.node.name for t in op_targets]
        parts: list[str] = [self._header(tl_names, op_names), ""]
        if import_texts:
            parts.append("# --- external imports (preserved) ---")
            parts.extend(import_texts)
            parts.append("")
        if const_keys:
            parts.append("# --- module-level constants referenced by the kernels ---")
            for key in const_keys:
                st = self.loader._cache[key[0]]
                parts.append(ast.get_source_segment(st.source, self.discovered[key]))
            parts.append("")
        if tl_targets:
            parts.append("# --- TileLang kernel functions (@tilelang.jit) ---")
            for t in tl_targets:
                parts.append(self._emit_node(t))
            parts.append("")
        if op_targets:
            parts.append("# --- per-op pointwise math (op_func passed into the builder) ---")
            for t in op_targets:
                parts.append(self._emit_node(t))
            parts.append("")
        all_names = tl_names + op_names
        all_list = ", ".join(f'"{n}"' for n in all_names)
        parts.append(f"__all__ = [{all_list}]")
        return "\n".join(parts) + "\n"

    def _header(self, tl_names: list[str], op_names: list[str]) -> str:
        desc = f"TileLang kernels: {', '.join(tl_names)}" if tl_names else ""
        if op_names:
            desc += (
                f" | op_func: {', '.join(op_names)}" if desc else f"op_func: {', '.join(op_names)}"
            )
        return (
            '"""Extracted TileLang kernel implementation.\n\n'
            f"{desc}\n"
            f"Generated by extract_tl_kernel.py (pattern {self.pattern}) "
            f"from repo root: {self.repo_root}\n"
            '"""'
        )

    # -- reporting helpers ------------------------------------------------ #

    def contributing_files(self) -> list[str]:
        return sorted({str(p.relative_to(self.repo_root)) for p in self.loader._cache})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a self-contained TileLang kernel for a TileOPs op."
    )
    parser.add_argument("--op-name", required=True, help="Manifest op key (e.g. MishFwdOp).")
    parser.add_argument(
        "--gpu-repo-root", required=True, type=Path, help="Path to the TileOPs repo root."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .py path, or a directory to write into "
        "(default: extracted_<OpName>.py in cwd). "
        "When a directory is given, the filename is "
        "auto-generated as _<op_slug>_kernels.py.",
    )
    args = parser.parse_args(argv)

    repo_root = args.gpu_repo_root.resolve()
    if not repo_root.exists():
        parser.error(f"--gpu-repo-root does not exist: {repo_root}")

    entry = load_manifest_entry(args.op_name, repo_root)
    source = entry["source"]
    kernel_rel = source.get("kernel")
    if not kernel_rel:
        parser.error(f"op '{args.op_name}' has no 'source.kernel' field")
    kernel_file = repo_root / kernel_rel
    if not kernel_file.exists():
        parser.error(f"source.kernel file not found: {kernel_file}")

    target_names = resolve_kernel_class_names(entry, args.op_name, repo_root)

    extractor = Extractor(repo_root)
    output = extractor.extract(kernel_file, target_names, args.op_name)

    for w in extractor.warnings:
        print(f"[warn] {w}", file=sys.stderr)

    out_path = args.out or (Path.cwd() / f"extracted_{_slug(args.op_name)}.py")
    # If --out is a directory (or a non-.py path that doesn't yet exist),
    # auto-generate the filename inside it.
    if out_path.is_dir() or (not out_path.exists() and out_path.suffix != ".py"):
        out_path = out_path / f"_{_op_name_to_snake(args.op_name)}_kernels.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")

    # Report which source files contributed to the extraction.
    contributing = extractor.contributing_files()
    tl_count = sum(1 for t in extractor.targets if t.kind == "tl_kernel")
    op_count = sum(1 for t in extractor.targets if t.kind == "op_func")
    print(
        f"[ok] kernel class(es): {', '.join(target_names)} (resolution pattern {extractor.pattern})"
    )
    print(
        f"[ok] extracted {tl_count} TileLang kernel function(s) "
        f"+ {op_count} op_func(s) from {len(contributing)} source file(s):"
    )
    for f in contributing:
        print(f"       - {f}")
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
