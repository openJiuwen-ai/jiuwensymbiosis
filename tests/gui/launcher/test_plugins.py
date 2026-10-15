"""Plugin selection, dependency isolation and restart argument contracts."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui import launcher
from jiuwensymbiosis_gui.plugin import GuiSpec, LaunchOptions, StartupError


def _entry(key, spec):
    return SimpleNamespace(name=key, load=lambda: lambda: spec)


def test_list_never_imports_framework_or_core():
    code = """
import sys
from jiuwensymbiosis_gui.launcher import main
assert main(["--list-guis"]) == 0
assert "nicegui" not in sys.modules
assert "jiuwensymbiosis" not in sys.modules
assert "jiuwensymbiosis_gui.workbench.app" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "workbench" in result.stdout


def test_metadata_listing_does_not_load_targets(monkeypatch):
    spec = GuiSpec("example", "Example", 1, "missing_ui_dependency.app:main")
    monkeypatch.setattr(launcher, "entry_points", lambda **kw: [_entry("example", spec)])
    specs, errors = launcher.discover_guis()
    assert specs["example"] == spec
    assert errors == {}


def test_duplicate_key_is_rejected(monkeypatch):
    monkeypatch.setattr(launcher, "entry_points", lambda **kw: [_entry("workbench", None)])
    with pytest.raises(StartupError, match="重复"):
        launcher.discover_guis()


def test_incompatible_plugin_does_not_hide_other_plugins(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "entry_points",
        lambda **kw: [
            _entry("broken", GuiSpec("broken", "Broken", 999, "broken:main")),
            _entry("example", GuiSpec("example", "Example", 1, "example:main")),
        ],
    )
    specs, errors = launcher.discover_guis()
    assert set(specs) == {"workbench", "example"}
    assert "版本" in errors["broken"]


def test_selected_plugin_receives_normalized_options(monkeypatch, tmp_path):
    config = tmp_path / "robot.yaml"
    config.write_text("adapter: piper\n")
    spec = GuiSpec("example", "Example", 1, "example:main")
    monkeypatch.setattr(launcher, "discover_guis", lambda: ({"example": spec}, {}))
    calls = []
    monkeypatch.setattr(launcher, "_load_target", lambda target: lambda options: calls.append((target, options)) or 0)
    assert launcher.main(["--gui", "example", "--config", str(config), "--port", "8811", "--no-browser"]) == 0
    target, options = calls[0]
    assert target == "example:main"
    assert options.config_path == config.resolve()
    assert options.gui == "example" and options.port == 8811 and not options.open_browser


def test_unknown_gui_never_falls_back(monkeypatch):
    monkeypatch.setattr(launcher, "entry_points", lambda **kw: [])
    monkeypatch.setattr(launcher, "_load_target", lambda target: pytest.fail("must not launch"))
    assert launcher.main(["--gui", "missing", "--no-browser"]) == 1


def test_restart_preserves_all_options(tmp_path):
    options = LaunchOptions(
        "example", tmp_path / "robot.yaml", tmp_path / "space", "127.0.0.2", 8899, True, tmp_path / "ui.yaml"
    )
    args = options.arguments(restart=True)
    assert args == [
        "--gui",
        "example",
        "--host",
        "127.0.0.2",
        "--port",
        "8899",
        "--config",
        str(tmp_path / "robot.yaml"),
        "--workspace",
        str(tmp_path / "space"),
        "--gui-config",
        str(tmp_path / "ui.yaml"),
        "--no-browser",
    ]
    assert options.open_browser
