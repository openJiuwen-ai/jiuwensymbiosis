"""Framework-independent, versioned GUI launch contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

__all__ = ["GUI_API_VERSION", "GuiSpec", "LaunchOptions", "StartupError"]

GUI_API_VERSION = 1


class StartupError(ValueError):
    """An actionable startup problem safe to display without a traceback."""


@dataclass(frozen=True)
class GuiSpec:
    key: str
    display_name: str
    api_version: int
    launch_target: str


@dataclass(frozen=True)
class LaunchOptions:
    gui: str = "workbench"
    config_path: Path | None = None
    workspace: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8770
    open_browser: bool = True
    gui_config_path: Path | None = None

    def arguments(self, *, restart: bool = False) -> list[str]:
        """Reproduce normalized launch options without using a shell."""
        options = replace(self, open_browser=False) if restart else self
        args = ["--gui", options.gui, "--host", options.host, "--port", str(options.port)]
        for flag, path in (
            ("--config", options.config_path),
            ("--workspace", options.workspace),
            ("--gui-config", options.gui_config_path),
        ):
            if path is not None:
                args.extend([flag, str(path)])
        if not options.open_browser:
            args.append("--no-browser")
        return args

    def identity(self) -> dict[str, str | int | None]:
        """Application identity excludes the one-shot browser-opening preference."""
        return {
            "gui": self.gui,
            "host": self.host,
            "port": self.port,
            "config": str(self.config_path) if self.config_path else None,
            "workspace": str(self.workspace) if self.workspace else None,
            "gui_config": str(self.gui_config_path) if self.gui_config_path else None,
        }
