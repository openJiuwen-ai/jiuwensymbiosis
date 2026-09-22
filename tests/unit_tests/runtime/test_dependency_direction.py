"""Core runtime must remain usable without a GUI installation."""

import ast
from pathlib import Path


def test_core_does_not_import_gui_except_legacy_launcher():
    root = Path(__file__).resolve().parents[3] / "jiuwensymbiosis"
    violations = []
    for path in root.rglob("*.py"):
        if path == root / "gui" / "__main__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            if any(module.split(".")[0] in {"jiuwensymbiosis_gui", "nicegui"} for module in modules):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations


def test_runtime_does_not_import_calibration():
    root = Path(__file__).resolve().parents[3] / "jiuwensymbiosis" / "runtime"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("jiuwensymbiosis.calibration"), path
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("jiuwensymbiosis.calibration") for alias in node.names), path
