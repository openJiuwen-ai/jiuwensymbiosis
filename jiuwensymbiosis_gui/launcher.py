"""Discover lightweight plugin descriptions and launch only the selected app."""

from __future__ import annotations

import argparse
import importlib
import logging
import re
from collections.abc import Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from jiuwensymbiosis_gui.plugin import GUI_API_VERSION, GuiSpec, LaunchOptions, StartupError

__all__ = ["discover_guis", "main"]


def _load_target(target: str) -> Any:
    module, separator, name = target.partition(":")
    if not separator:
        raise StartupError(f"无效的插件入口: {target}")
    if not module or not name or "." in name:
        raise StartupError(f"无效的插件入口: {target}")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise StartupError(f"插件入口不是可调用函数: {target}")
    return value


def _validate(spec: Any, key: str) -> GuiSpec:
    if not isinstance(spec, GuiSpec) or spec.key != key or not re.fullmatch(r"[a-z][a-z0-9_-]*", key):
        raise StartupError(f"GUI 描述无效或 key 与注册名不一致: {key}")
    if spec.api_version != GUI_API_VERSION:
        raise StartupError(f"GUI {key} 的插件 API 版本 {spec.api_version} 不兼容（需要 {GUI_API_VERSION}）")
    if not spec.display_name or ":" not in spec.launch_target:
        raise StartupError(f"GUI 描述缺少显示名或启动入口: {key}")
    return spec


def discover_guis() -> tuple[dict[str, GuiSpec], dict[str, str]]:
    """Return valid descriptions and per-plugin errors without loading app targets."""
    from jiuwensymbiosis_gui.workbench.plugin import describe

    specs = {"workbench": _validate(describe(), "workbench")}
    errors: dict[str, str] = {}
    seen = {"workbench"}
    for entry in sorted(entry_points(group="jiuwensymbiosis.gui"), key=lambda value: value.name):
        if entry.name in seen:
            raise StartupError(f"重复的 GUI key: {entry.name}")
        seen.add(entry.name)
        try:
            specs[entry.name] = _validate(entry.load()(), entry.name)
        except Exception as exc:
            errors[entry.name] = f"{type(exc).__name__}: {exc}"
    return specs, errors


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse common options, preserve a light help/list path, then launch."""
    parser = argparse.ArgumentParser(prog="jiuwensymbiosis-gui", description="选择并启动 GUI 插件")
    parser.add_argument("--gui", default="workbench")
    parser.add_argument("--list-guis", action="store_true")
    parser.add_argument("--config", type=_path)
    parser.add_argument("--workspace", type=_path)
    parser.add_argument("--gui-config", type=_path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        specs, errors = discover_guis()
        if args.list_guis:
            for key, spec in sorted(specs.items()):
                print(f"{key}\t{spec.display_name}\tAPI {spec.api_version}")
            for key, error in sorted(errors.items()):
                print(f"{key}\t不可用: {error}")
            return 0
        if args.gui in errors:
            raise StartupError(f"GUI {args.gui} 不可用: {errors[args.gui]}")
        if args.gui not in specs:
            raise StartupError(f"未知 GUI: {args.gui}；已安装: {', '.join(sorted(specs))}")
        if not 1 <= args.port <= 65535:
            raise StartupError("端口必须在 1..65535 之间")
        for path in (args.config, args.gui_config):
            if path is not None and not path.is_file():
                raise StartupError(f"配置文件不存在: {path}")
        options = LaunchOptions(
            args.gui, args.config, args.workspace, args.host, args.port, not args.no_browser, args.gui_config
        )
        from jiuwensymbiosis.utils.proxy import clear_proxy_env

        clear_proxy_env()
        return int(_load_target(specs[args.gui].launch_target)(options))
    except Exception as exc:
        from jiuwensymbiosis_gui.startup import _show_error_dialog

        message = f"GUI {args.gui} 启动失败: {exc}"
        logging.getLogger(__name__).error(message, exc_info=not isinstance(exc, StartupError))
        if not args.no_browser and not args.list_guis:
            _show_error_dialog("无法启动 Jiuwen Symbiosis", message)
        return 1
