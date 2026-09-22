# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Launcher startup helpers and the legacy ``python -m`` forwarding entry."""

from __future__ import annotations

import sys
from types import ModuleType

from jiuwensymbiosis_gui import startup


def test_preferred_cjk_font_picks_first_available():
    available = {"fixed", "Noto Sans CJK SC", "SimHei"}
    assert startup._preferred_cjk_font(available) == "Noto Sans CJK SC"


def test_preferred_cjk_font_respects_priority():
    # 只装了较低优先级的:文泉驿优先于 SimHei
    assert startup._preferred_cjk_font({"SimHei", "WenQuanYi Micro Hei"}) == "WenQuanYi Micro Hei"


def test_preferred_cjk_font_falls_back_to_x11_core():
    # 无 Xft 的 Tk 只认得核心位图字体
    assert startup._preferred_cjk_font({"fixed", "gothic", "song ti"}) == "song ti"


def test_preferred_cjk_font_none_when_absent():
    assert startup._preferred_cjk_font({"fixed", "helvetica", "courier"}) is None


def test_nearest_native_px_picks_closest():
    # song ti 在目标 26 附近的原生位图字号
    assert startup._nearest_native_px(26, [16, 24, 25, 36]) == 25


def test_nearest_native_px_prefers_smaller_on_tie():
    # 距离相等时取较小的字号,避免弹窗过大
    assert startup._nearest_native_px(30, [24, 36]) == 24


def test_nearest_native_px_falls_back_to_target_when_no_bitmap():
    # TrueType/Xft:任意字号都平滑,没有"原生尺寸"约束,直接用目标像素
    assert startup._nearest_native_px(26, []) == 26


def test_old_module_entry_forwards_to_unified_launcher(monkeypatch):
    calls: list[str] = []
    launcher_module = ModuleType("jiuwensymbiosis_gui.__main__")
    launcher_module.main = lambda: calls.append("called") or 23
    monkeypatch.setitem(sys.modules, launcher_module.__name__, launcher_module)

    from jiuwensymbiosis.gui import __main__ as old_entry

    assert old_entry.main() == 23
    assert calls == ["called"]
