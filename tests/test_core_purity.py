"""Architectural guard: `core/` must never import Streamlit.

The core package holds framework-agnostic logic shared by the Streamlit
app, the MCP server, and any future cross-domain adapter. If any core
module reaches for Streamlit, the abstraction rots silently — this test
fails loudly instead.

Checks at both AST level (catches `import streamlit`, `from streamlit
import ...`, and aliased forms) and string level (catches dynamic imports
like `__import__("streamlit")` or `importlib.import_module("streamlit")`).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "core"


def _core_py_files() -> list[Path]:
    return sorted(p for p in CORE_DIR.rglob("*.py") if p.is_file())


class _StreamlitImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                self.hits.append(f"line {node.lineno}: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and (node.module == "streamlit" or node.module.startswith("streamlit.")):
            self.hits.append(f"line {node.lineno}: from {node.module} import ...")
        self.generic_visit(node)


@pytest.mark.parametrize("path", _core_py_files(), ids=lambda p: p.name)
def test_core_module_has_no_streamlit_import(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source, filename=str(path))
    visitor = _StreamlitImportVisitor()
    visitor.visit(tree)
    assert not visitor.hits, (
        f"{path.relative_to(CORE_DIR.parent)} imports Streamlit (forbidden in core/):\n  "
        + "\n  ".join(visitor.hits)
    )

    # Catch dynamic-import escape hatches too.
    lowered = source.lower()
    for sentinel in (
        '__import__("streamlit"',
        "__import__('streamlit'",
        'import_module("streamlit"',
        "import_module('streamlit'",
    ):
        assert sentinel not in lowered, (
            f"{path.relative_to(CORE_DIR.parent)} uses dynamic streamlit import: {sentinel!r}"
        )


def test_core_package_has_at_least_one_module() -> None:
    """Guard against silently passing the purity test on an empty core/."""
    files = _core_py_files()
    # Expect __init__.py + types.py + portfolio_ops.py at minimum.
    assert len(files) >= 3, f"core/ has only {len(files)} .py files: {files}"
