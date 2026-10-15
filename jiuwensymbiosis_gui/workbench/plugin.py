"""Lazy workbench registration and dependency preflight."""

from __future__ import annotations

from jiuwensymbiosis_gui.plugin import GuiSpec, LaunchOptions, StartupError


def describe() -> GuiSpec:
    return GuiSpec("workbench", "Jiuwen Symbiosis 工作台", 1, "jiuwensymbiosis_gui.workbench.plugin:main")


def main(options: LaunchOptions) -> int:
    from jiuwensymbiosis_gui.workbench.preflight import preflight_message

    message = preflight_message()
    if message:
        raise StartupError(message)
    if options.gui_config_path is not None:
        raise StartupError("workbench 当前没有专属 GUI 配置项，请使用 --config 指定运行配置")
    from jiuwensymbiosis_gui.workbench.app import run

    return run(options=options)
