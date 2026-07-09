# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""GUI 入口(``python -m jiuwensymbiosis.gui``)的纯逻辑单测。

只测不依赖显示的部分:兜底错误弹窗的中文字体挑选。弹窗本身需要 X11/Tk,
留给手动/集成验证。"""

from __future__ import annotations

from jiuwensymbiosis.gui import __main__ as gui_main


def test_preferred_cjk_font_picks_first_available():
    available = {"fixed", "Noto Sans CJK SC", "SimHei"}
    assert gui_main._preferred_cjk_font(available) == "Noto Sans CJK SC"


def test_preferred_cjk_font_respects_priority():
    # 只装了较低优先级的:文泉驿优先于 SimHei
    assert gui_main._preferred_cjk_font({"SimHei", "WenQuanYi Micro Hei"}) == "WenQuanYi Micro Hei"


def test_preferred_cjk_font_falls_back_to_x11_core():
    # 无 Xft 的 Tk 只认得核心位图字体
    assert gui_main._preferred_cjk_font({"fixed", "gothic", "song ti"}) == "song ti"


def test_preferred_cjk_font_none_when_absent():
    assert gui_main._preferred_cjk_font({"fixed", "helvetica", "courier"}) is None


def test_nearest_native_px_picks_closest():
    # song ti 在目标 26 附近的原生位图字号
    assert gui_main._nearest_native_px(26, [16, 24, 25, 36]) == 25


def test_nearest_native_px_prefers_smaller_on_tie():
    # 距离相等时取较小的字号,避免弹窗过大
    assert gui_main._nearest_native_px(30, [24, 36]) == 24


def test_nearest_native_px_falls_back_to_target_when_no_bitmap():
    # TrueType/Xft:任意字号都平滑,没有"原生尺寸"约束,直接用目标像素
    assert gui_main._nearest_native_px(26, []) == 26


def test_startup_failure_message_is_actionable():
    msg = gui_main._startup_failure_message(ModuleNotFoundError("No module named 'nicegui'"))
    assert ".[gui]" in msg  # 指引装 GUI 依赖
    assert "conda activate" in msg  # 指引换环境
    assert "nicegui" in msg  # 透出异常摘要,便于定位
