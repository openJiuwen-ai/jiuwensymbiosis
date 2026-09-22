# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""配置页(NiceGUI 版):按类别分组的常用表单 + 原始 YAML 兜底(双向同步)。

dict 为单一真源(``ConfigModel``)。表单控件按 ``FieldSpec.path`` 绑定到点分路径;
「原始 YAML」标签可整体编辑其余字段,点「应用」(或 Ctrl+S)回填并重建表单,点「下载」
经浏览器另存,点「存储为可选配置」写进本体的配置目录(``configs/<本体>/``,写进去即成为
主页「配置文件」下拉的候选)。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.config_model import GROUP_ORDER, ConfigModel, FieldSpec, field_groups_for_config

__all__ = ["ConfigView"]

_YAML_TAB = "原始 YAML"


class ConfigView:
    """任务配置编辑页。``on_run`` 用当前配置运行,``on_back`` 返回主页。"""

    def __init__(
        self,
        *,
        on_run: Callable[[], None],
        on_back: Callable[[], None],
        on_config_saved: Callable[[], None] = lambda: None,
    ) -> None:
        self._on_config_saved = on_config_saved
        self._model = ConfigModel()
        self._fields: tuple[FieldSpec, ...] = ()
        self._yaml: Any = None
        self._body_key = ""
        self._form_tabs: Any = None
        self._controls: dict[str, Any] = {}
        with ui.column().classes("w-full gap-2"):
            self._title = ui.label("").classes("text-lg font-bold")
            self._form_host = ui.column().classes("w-full")
            self._warn = ui.label("").classes("text-orange-600 text-sm")
            with ui.row().classes("w-full items-center gap-2"):
                ui.button("← 返回主页", on_click=lambda: on_back()).props("flat")
                ui.space()
                ui.button("▶ 用当前配置运行", on_click=lambda: on_run()).props("color=primary")
            self._save_dialog, self._save_name, self._save_hint = self._build_save_dialog()

    # ------------------------------------------------------------------ API
    def load(self, title: str, model: ConfigModel, *, body_key: str) -> None:
        """载入某(本体, 任务)的配置模型并重建表单;字段按本体切换「机器人参数」组。"""
        self._model = model
        self._body_key = body_key
        self._fields = field_groups_for_config(body_key, model)
        self._title.set_text(f"配置:{title}")
        self._build_form()
        self._refresh_warnings()

    def reveal_field(self, path: str) -> bool:
        """切到某字段所在的分组并聚焦它;该字段不在本体表单里(只能走原始 YAML)则返回 False。"""
        spec = next((s for s in self._fields if s.path == path), None)
        if spec is None or self._form_tabs is None:
            return False
        self._form_tabs.set_value(spec.group)
        control = self._controls.get(path)
        if control is not None:
            # 标签面板只挂载当前那一页,刚切过去的这一帧控件还不在 DOM 里,聚焦得等它挂上。
            ui.timer(0.2, lambda: control.run_method("focus"), once=True)
        return True

    # ------------------------------------------------------------------ 表单
    def _build_form(self, *, active: str | None = None) -> None:
        """重建表单;``active`` 指定重建后停留的标签(缺省回到第一个分组)。"""
        self._form_host.clear()
        self._controls.clear()
        groups = [g for g in GROUP_ORDER if any(s.group == g for s in self._fields)]
        with self._form_host:
            with ui.tabs().classes("w-full") as tabs:
                for group in groups:
                    ui.tab(group)
                ui.tab(_YAML_TAB)
            self._form_tabs = tabs
            first = active or (groups[0] if groups else _YAML_TAB)
            with ui.tab_panels(tabs, value=first, on_change=self._on_tab).classes("w-full"):
                for group in groups:
                    with ui.tab_panel(group):
                        self._build_group(group)
                with ui.tab_panel(_YAML_TAB):
                    self._yaml = (
                        ui.textarea(value=self._model.to_yaml()).classes("w-full font-mono").props("outlined rows=20")
                    )
                    # .prevent 拦掉浏览器自带的 Ctrl+S(保存网页)对话框。
                    self._yaml.on("keydown.ctrl.s.prevent", lambda _e: self._apply_yaml())
                    with ui.row().classes("items-center gap-2"):
                        ui.button("应用", on_click=self._apply_yaml)
                        ui.button("下载", on_click=self._download_yaml)
                        ui.button("存储为可选配置", on_click=self._open_save)

    def _build_group(self, group: str) -> None:
        for spec in [s for s in self._fields if s.group == group]:
            control = self._make_control(spec)
            self._controls[spec.path] = control
            if spec.help:
                control.tooltip(spec.help)

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
        self._build_form(active=_YAML_TAB)  # 应用后仍停在原始 YAML,便于接着改
        self._refresh_warnings()
        ui.notify("已应用到表单", type="positive", timeout=1500)

    # ------------------------------------------------------------------ 下载 / 存储
    def _download_yaml(self) -> None:
        """把编辑框里的 YAML(不是表单里那份)交给浏览器下载;非法 YAML 不下载。"""
        text = self._yaml.value
        if not self._valid_yaml(text):
            return
        ui.download.content(text, f"{self._body_key or 'config'}.yaml", media_type="application/x-yaml")

    def _build_save_dialog(self) -> tuple[Any, Any, Any]:
        with ui.dialog() as dialog, ui.card().classes("w-[32rem]"):
            ui.label("存储为可选配置").classes("text-lg font-bold")
            name_input = ui.input("配置名").classes("w-full")
            hint = ui.label("").classes("text-red-600 text-xs")
            name_input.on_value_change(lambda _e: self._refresh_overwrite_hint())
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=dialog.close).props("flat")
                ui.button("保存", on_click=self._save_as_option).props("color=primary")
        return dialog, name_input, hint

    def _open_save(self) -> None:
        """默认名用 ``<本体>.local``(``*.local.yaml`` 已被 gitignore),不去撞随包配置。"""
        self._save_name.set_value(f"{self._body_key}.local")
        self._refresh_overwrite_hint()
        self._save_dialog.open()

    def _refresh_overwrite_hint(self) -> None:
        try:
            path = registry.body_config_path(self._body_key, self._save_name.value)
        except ValueError:
            self._save_hint.set_text("")
            return
        self._save_hint.set_text(f"{path.name} 已存在,保存将覆盖它" if path.is_file() else "")

    def _save_as_option(self) -> None:
        """把编辑框里的 YAML 存进本体配置目录,存完通知主页刷新「配置文件」下拉。"""
        text = self._yaml.value
        if not self._valid_yaml(text):
            return
        try:
            path = registry.save_body_config(self._body_key, self._save_name.value, text)
        except (ValueError, OSError) as exc:
            ui.notify(f"保存失败:{exc}", type="negative")
            return
        self._save_dialog.close()
        self._on_config_saved()
        ui.notify(f"已保存:{path}", type="positive", timeout=2500)

    @staticmethod
    def _valid_yaml(text: str) -> bool:
        """校验待导出的 YAML,非法时提示并返回 False(不把打不开的配置存出去)。"""
        try:
            ConfigModel.from_yaml_text(text)
        except ValueError as exc:
            ui.notify(f"YAML 无效:{exc}", type="negative")
            return False
        return True
