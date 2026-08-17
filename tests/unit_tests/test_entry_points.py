# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Every shipped adapter is reachable from every entry point.

The runner and the GUI each keep their own list of bodies, and they drifted:
SO-101 shipped without a runner entry, Cruzr without a GUI entry — so a body was
runnable one way and invisible the other. This pins both lists to the set of
adapters that actually ship a default config.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ("piper", "so101", "cruzr")


@pytest.fixture(scope="module")
def runner():
    """``examples/run_task.py`` loaded as a module (it is normally run via runpy)."""
    spec = importlib.util.spec_from_file_location("_run_task", REPO_ROOT / "examples" / "run_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("adapter", SHIPPED)
def test_shipped_adapter_has_a_default_config(adapter):
    cfg = REPO_ROOT / "configs" / adapter / f"{adapter}.yaml"
    assert yaml.safe_load(cfg.read_text(encoding="utf-8"))["adapter"] == adapter


@pytest.mark.parametrize("adapter", SHIPPED)
def test_runner_registers_shipped_adapter(runner, adapter):
    assert adapter in runner._robot_session_builders()


@pytest.mark.parametrize("adapter", SHIPPED)
def test_gui_lists_shipped_adapter(adapter):
    from jiuwensymbiosis.gui import registry

    body = registry.get_body(adapter)
    assert body.key == adapter
    assert body.config_path() == REPO_ROOT / "configs" / adapter / f"{adapter}.yaml"
