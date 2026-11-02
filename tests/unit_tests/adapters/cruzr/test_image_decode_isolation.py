# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The modules a ``/usr/bin/python3`` worker loads by path must stay import-isolated.

The Cruzr waist-camera worker (``ros2_camera_worker.py``) runs under
``/usr/bin/python3``, where ``openjiuwen`` is NOT installed. Importing any
``jiuwensymbiosis.<submodule>`` there runs the package root ``__init__`` which
eagerly imports the agent stack -> ``openjiuwen`` -> ``ModuleNotFoundError``.

So a worker loads such a module by file path (``spec_from_file_location``),
bypassing the package ``__init__``. That only works while the module pulls in
nothing from ``jiuwensymbiosis`` (numpy + stdlib only). Two modules are loaded
this way today — ``ros2/image_decode`` (camera worker) and
``motion/diff_drive`` (wheel worker) — and these tests lock the contract in
for both.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.motion import diff_drive
from jiuwensymbiosis.ros2 import image_decode

MODULE_PATH = Path(image_decode.__file__)
# Every module a bare-interpreter worker loads by path. Adding one here is what
# keeps the next author from importing the agent stack into it by accident.
STANDALONE_MODULES = [Path(image_decode.__file__), Path(diff_drive.__file__)]


def _load_by_path(path: Path = MODULE_PATH):
    """Load a module the way the worker does — file path, no package init."""
    spec = importlib.util.spec_from_file_location(f"standalone_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", STANDALONE_MODULES, ids=lambda p: p.name)
def test_module_imports_nothing_from_jiuwensymbiosis(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)
    offenders = [name for name in imported if name.split(".")[0] == "jiuwensymbiosis"]
    assert not offenders, (
        f"{path.name} must stay dependency-free so the /usr/bin/python3 worker "
        f"can load it standalone; found package imports: {offenders}"
    )


@pytest.mark.parametrize("path", STANDALONE_MODULES, ids=lambda p: p.name)
def test_standalone_load_succeeds(path):
    assert _load_by_path(path) is not None


def test_wheel_worker_bootstrap_path_points_at_diff_drive():
    """The worker's ``parents[N]`` walk must still land on diff_drive.py.

    It is a hand-written relative path, so a directory move silently breaks it at
    robot start-up rather than at import time — exactly what happened when the
    workers moved under ``adapters/cruzr/ros2/``.
    """
    worker = Path(__file__).resolve().parents[4] / "jiuwensymbiosis/adapters/cruzr/ros2/wheel_worker.py"
    resolved = worker.resolve().parents[3] / "motion" / "diff_drive.py"
    assert resolved == Path(diff_drive.__file__).resolve()


def test_standalone_load_decodes_rgb8():
    mod = _load_by_path()
    h, w = 1, 2
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    msg = SimpleNamespace(height=h, width=w, encoding="rgb8", step=w * 3, data=rgb.tobytes())
    out = mod.decode_image_msg(msg)
    np.testing.assert_array_equal(out, rgb)
