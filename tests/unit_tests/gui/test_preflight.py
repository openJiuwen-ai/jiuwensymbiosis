# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""preflight:NiceGUI 预检与缺失指引(纯逻辑,monkeypatch 探测结果)。

preflight 用 ``find_spec`` 只探测、不导入 nicegui,故本模块在只装 [dev] 的 CI 里也能跑。
"""

from __future__ import annotations

from jiuwensymbiosis.gui import preflight


def test_no_message_when_nicegui_present(monkeypatch):
    monkeypatch.setattr(preflight, "nicegui_installed", lambda: True)
    assert preflight.preflight_message() is None


def test_message_when_nicegui_missing_gives_pip_install_hint(monkeypatch):
    monkeypatch.setattr(preflight, "nicegui_installed", lambda: False)
    message = preflight.preflight_message()
    assert message is not None
    assert "NiceGUI" in message
    assert 'pip install -e ".[gui]"' in message
