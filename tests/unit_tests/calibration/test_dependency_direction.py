# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Import-boundary contracts for the in-tree calibration subsystem."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_ROOT = _REPO_ROOT / "jiuwensymbiosis"
_CALIBRATION_PREFIX = "jiuwensymbiosis.calibration"
# The calibration package is an independent subsystem, so only its own subtree may
# import its facade. GUI consumers live outside the core package and have their own
# suite under tests/gui/.
_CALIBRATION_CONSUMERS = ("calibration",)


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            imported.extend((node.lineno, alias.name) for alias in node.names)
    return imported


def _run_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(value for value in (str(_REPO_ROOT), existing) if value)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_core_source_does_not_import_calibration_facade() -> None:
    """Core runtime modules depend on protocols, not the calibration package."""
    offenders: list[str] = []
    for path in _CORE_ROOT.rglob("*.py"):
        relative = path.relative_to(_CORE_ROOT)
        if any(part in _CALIBRATION_CONSUMERS for part in relative.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.ImportFrom):
                imported = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == _CALIBRATION_PREFIX or alias.name.startswith(f"{_CALIBRATION_PREFIX}."):
                        offenders.append(f"{relative}:{node.lineno}: {alias.name}")
            if imported and (imported == _CALIBRATION_PREFIX or imported.startswith(f"{_CALIBRATION_PREFIX}.")):
                offenders.append(f"{relative}:{node.lineno}: {imported}")
    assert not offenders, "core imports calibration facade:\n" + "\n".join(offenders)


def test_importing_core_and_adapters_does_not_load_calibration() -> None:
    """Package and adapter public imports stay lazy with respect to calibration."""
    probe = """
import sys
import jiuwensymbiosis
import jiuwensymbiosis.adapters.piper
import jiuwensymbiosis.adapters.so101
offenders = sorted(name for name in sys.modules if name == 'jiuwensymbiosis.calibration' or name.startswith('jiuwensymbiosis.calibration.'))
assert not offenders, offenders
"""
    result = _run_probe(probe)
    assert result.returncode == 0, result.stdout + result.stderr


def test_calibration_facade_does_not_load_concrete_adapters() -> None:
    """The domain facade can be imported without importing a vendor adapter."""
    probe = """
import sys
import jiuwensymbiosis.calibration
offenders = sorted(name for name in sys.modules if name.startswith('jiuwensymbiosis.adapters.'))
assert not offenders, offenders
"""
    result = _run_probe(probe)
    assert result.returncode == 0, result.stdout + result.stderr


def test_calibration_internal_layers_keep_one_way_dependencies() -> None:
    """Domain/artifacts/workflows must not reach into an outer layer."""
    forbidden = {
        "domain": (
            f"{_CALIBRATION_PREFIX}.artifacts",
            f"{_CALIBRATION_PREFIX}.workflows",
            f"{_CALIBRATION_PREFIX}.integration",
            f"{_CALIBRATION_PREFIX}.adapters",
        ),
        "artifacts": (
            f"{_CALIBRATION_PREFIX}.workflows",
            f"{_CALIBRATION_PREFIX}.integration",
            f"{_CALIBRATION_PREFIX}.adapters",
        ),
        "workflows": (
            f"{_CALIBRATION_PREFIX}.integration",
            f"{_CALIBRATION_PREFIX}.adapters",
        ),
    }
    offenders: list[str] = []
    calibration_root = _CORE_ROOT / "calibration"
    for layer, prefixes in forbidden.items():
        for path in (calibration_root / layer).rglob("*.py"):
            for lineno, imported in _imported_modules(path):
                if any(imported == prefix or imported.startswith(f"{prefix}.") for prefix in prefixes):
                    offenders.append(f"{path.relative_to(_CORE_ROOT)}:{lineno}: {imported}")
    assert not offenders, "calibration layer dependency inversion:\n" + "\n".join(offenders)
