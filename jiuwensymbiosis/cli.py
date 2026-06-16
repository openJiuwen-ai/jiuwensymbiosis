# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lightweight CLI entry points registered as ``console_scripts`` in pyproject.toml."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _run_example(script: str) -> None:
    script_path = _EXAMPLES_DIR / script
    if not script_path.is_file():
        sys.stderr.write(f"Example script not found: {script_path}\n")
        raise SystemExit(1)
    runpy.run_path(str(script_path), run_name="__main__")


def piper_pick_demo() -> None:
    _run_example("piper_pick_demo.py")


def piper_watch_pick_place() -> None:
    _run_example("piper_watch_pick_place.py")
