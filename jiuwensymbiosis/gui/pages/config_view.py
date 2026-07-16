# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""配置页(NiceGUI 版):按类别分组的常用表单 + 原始 YAML 兜底(双向同步)。

dict 为单一真源(``ConfigModel``)。表单控件按 ``FieldSpec.path`` 绑定到点分路径;
「原始 YAML」标签可整体编辑其余字段,点「应用 YAML」回填并重建表单。模型相关字段在
模拟模式下置灰(模拟用离线模型,无需真实端点)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from jiuwensymbiosis.gui.config_model import FIELD_GROUPS, GROUP_ORDER, ConfigModel, FieldSpec

__all__ = ["ConfigView"]

_MODEL_GROUP = "模型"
_YAML_TAB = "原始 YAML"


class ConfigView:
    """任务配置编辑页。``on_run`` 用当前配置运行,``on_back`` 返回主页。"""

    def __init__(self, *, on_run: Callable[[], None], on_back: Callable[[], None]) -> None:
        self._model = ConfigModel()
        self._mock = True
        self._yaml: Any = None
        with ui.column().classes("w-full gap-2"):
            self._title = ui.label("").classes("text-lg font-bold")
            self._form_host = ui.column().classes("w-full")
            self._warn = ui.label("").classes("text-orange-600 text-sm")
            with ui.row().classes("w-full items-center gap-2"):
                ui.button("← 返回主页", on_click=lambda: on_back()).props("flat")
                ui.space()
                ui.button("▶ 用当前配置运行", on_click=lambda: on_run()).props("color=primary")

    # ------------------------------------------------------------------ API
    def load(self, title: str, model: ConfigModel, *, mock: bool) -> None:
        """载入某任务的配置模型并重建表单。"""
        self._model = model
        self._mock = mock
        self._title.set_text(f"配置:{title}")
        self._build_form()
        self._refresh_warnings()

    # ------------------------------------------------------------------ 表单
    def _build_form(self) -> None:
        self._form_host.clear()
        groups = [g for g in GROUP_ORDER if any(s.group == g for s in FIELD_GROUPS)]
        with self._form_host:
            with ui.tabs().classes("w-full") as tabs:
                for group in groups:
                    ui.tab(group)
                ui.tab(_YAML_TAB)
            first = groups[0] if groups else _YAML_TAB
            with ui.tab_panels(tabs, value=first, on_change=self._on_tab).classes("w-full"):
                for group in groups:
                    with ui.tab_panel(group):
                        self._build_group(group)
                with ui.tab_panel(_YAML_TAB):
                    self._yaml = (
                        ui.textarea(value=self._model.to_yaml()).classes("w-full font-mono").props("outlined rows=20")
                    )
                    ui.button("✔ 应用 YAML 到表单", on_click=self._apply_yaml)

    def _build_group(self, group: str) -> None:
        disabled_group = self._mock and group == _MODEL_GROUP
        for spec in [s for s in FIELD_GROUPS if s.group == group]:
            control = self._make_control(spec)
            if disabled_group or (self._mock and spec.disable_in_mock):
                control.disable()
            if spec.help:
                control.tooltip(spec.help)
        if disabled_group:
            ui.label("模拟模式使用离线模型,无需配置真实端点。").classes("text-gray-500 text-sm")

    def _make_control(self, spec: FieldSpec) -> Any:
        value = self._model.field_value(spec)
        path = spec.path
        if spec.kind == "bool":
            if spec.on_value is not None or spec.off_value is not None:
                sw = ui.switch(spec.label, value=value == spec.on_value)
                sw.on_value_change(
                    lambda e, p=path, on=spec.on_value, off=spec.off_value: self._set(p, on if e.value else off)
                )
                return sw
            plain_sw = ui.switch(spec.label, value=bool(value))
            plain_sw.on_value_change(lambda e, p=path: self._set(p, bool(e.value)))
            return plain_sw
        if spec.kind == "int":
            start_i = int(value) if isinstance(value, int | float) else (spec.min_value or 0)
            num_i = ui.number(spec.label, value=start_i, min=spec.min_value, precision=0, step=1)
            num_i.on_value_change(lambda e, p=path: self._set(p, int(e.value) if e.value is not None else 0))
            return num_i.classes("w-64")
        if spec.kind == "float":
            start_f = float(value) if isinstance(value, int | float) else 0.0
            num_f = ui.number(
                spec.label, value=start_f, min=spec.min_value, max=spec.max_value, step=spec.step, precision=3
            )
            num_f.on_value_change(lambda e, p=path: self._set(p, float(e.value) if e.value is not None else 0.0))
            return num_f.classes("w-64")
        if spec.kind == "choice":
            options = dict(spec.choices)
            sel = ui.select(options, label=spec.label, value=value if value in options else None)
            sel.on_value_change(lambda e, p=path: self._set(p, e.value))
            return sel.classes("w-64")
        if spec.kind == "text":
            ta = (
                ui.textarea(spec.label, value="" if value is None else str(value))
                .classes("w-full")
                .props("outlined rows=10")
            )
            ta.on_value_change(lambda e, p=path: self._set(p, e.value))
            return ta
        inp = ui.input(spec.label, value="" if value is None else str(value)).classes("w-full")
        inp.on_value_change(lambda e, p=path: self._set(p, e.value))
        return inp

    def _set(self, path: str, value: Any) -> None:
        self._model.set(path, value)
        self._refresh_warnings()

    def _refresh_warnings(self) -> None:
        self._warn.set_text("  ".join(f"⚠ {w}" for w in self._model.validate()))

    # ------------------------------------------------------------------ YAML 同步
    def _on_tab(self, e: Any) -> None:
        if e.value == _YAML_TAB and self._yaml is not None:
            self._yaml.set_value(self._model.to_yaml())

    def _apply_yaml(self) -> None:
        try:
            self._model.replace_from_yaml(self._yaml.value)
        except ValueError as exc:
            ui.notify(f"YAML 无效:{exc}", type="negative")
            return
        self._build_form()
        self._refresh_warnings()
