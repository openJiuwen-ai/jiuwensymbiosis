# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""scripts/gui_launcher.py 引导器的纯逻辑单测。

引导器在 ``scripts/``(非包),按文件路径加载;只测不依赖显示的消息构造。
"""

from __future__ import annotations

from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import module_from_spec
from pathlib import Path

_GUI_LAUNCHER = Path(__file__).resolve().parents[3] / "scripts" / "gui_launcher.py"


def _load_launcher():
    """把 scripts/gui_launcher.py 按文件路径加载成模块(它在 scripts/,不是可 import 的包)。"""
    file_loader = SourceFileLoader("jw_gui_launcher", str(_GUI_LAUNCHER))
    spec = ModuleSpec("jw_gui_launcher", file_loader, origin=str(_GUI_LAUNCHER))
    spec.has_location = True  # 让 module_from_spec 设好 __file__,供脚本内 os.path.abspath(__file__)
    launcher = module_from_spec(spec)
    file_loader.exec_module(launcher)
    return launcher


def test_launcher_file_exists():
    assert _GUI_LAUNCHER.is_file()


def test_import_failure_message_is_actionable():
    mod = _load_launcher()
    msg = mod._import_failure_message(ModuleNotFoundError("No module named 'jiuwensymbiosis'"))
    assert "conda activate" in msg
    assert ".[gui]" in msg
    assert "jiuwensymbiosis" in msg  # 透出异常摘要
