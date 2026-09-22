"""Install the example entry point and exercise the public plugin/runtime seams."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def installed_example_plugin(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "plugin-source"
    shutil.copytree(root / "examples" / "gui_plugin", source)
    target = tmp_path / "installed-plugin"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(target),
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"local plugin install failed:\n{result.stdout}\n{result.stderr}"
    return target, root


def test_installed_entry_point_is_discovered_and_selected_outside_repository(
    installed_example_plugin: tuple[Path, Path], tmp_path: Path
) -> None:
    target, root = installed_example_plugin
    gui_config = target / "jiuwensymbiosis_example_gui" / "configs" / "self-check.json"
    workspace = tmp_path / "self-check-workspace"
    resource_directory = tmp_path / "resource-locks"
    script = tmp_path / "run_installed_plugin.py"
    script.write_text(
        """from importlib.metadata import entry_points
import sys

from jiuwensymbiosis_gui.launcher import discover_guis, main

entries = {entry.name: entry for entry in entry_points(group=\"jiuwensymbiosis.gui\")}
assert \"example\" in entries, entries
specs, errors = discover_guis()
assert \"example\" in specs and not errors, (specs, errors)
assert specs[\"example\"].launch_target == \"jiuwensymbiosis_example_gui.plugin:main\"
assert main([\"--gui\", \"example\", \"--gui-config\", sys.argv[1],
             \"--workspace\", sys.argv[2], \"--no-browser\"]) == 0
assert \"nicegui\" not in sys.modules
assert \"jiuwensymbiosis_gui.workbench.app\" not in sys.modules
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(target), str(root)))
    env["JIUWENSYMBIOSIS_RUNTIME_DIR"] = str(resource_directory)
    result = subprocess.run(
        [sys.executable, str(script), str(gui_config), str(workspace)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"external plugin launch failed:\n{result.stdout}\n{result.stderr}"
    assert '"gui": "example"' in result.stdout
    assert '"runtime_closed": true' in result.stdout


def test_runtime_public_task_flow_uses_mock_session_without_private_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jiuwensymbiosis.adapters._common.builder import make_builder
    from jiuwensymbiosis.env.mock import MockArmEnv
    from jiuwensymbiosis.runtime import Runtime, worker
    from tests.mocks.mock_api import MockApi

    adapter_name = "installed_plugin_smoke"
    module_name = f"jiuwensymbiosis.adapters.{adapter_name}"

    @dataclass(frozen=True)
    class MockConfig:
        name: str = "installed-plugin-smoke"

        @classmethod
        def from_dict(cls, data: dict) -> MockConfig:
            return cls(name=str(data.get("name", "installed-plugin-smoke")))

        @classmethod
        def from_yaml(cls, _path: str | Path) -> MockConfig:
            return cls()

    class MockEnv(MockArmEnv):
        def __init__(self, _config: MockConfig) -> None:
            super().__init__()

    module = ModuleType(module_name)
    builder = make_builder(
        MockConfig,
        MockEnv,
        MockApi,
        resource_keys=lambda _config: ("device:installed-plugin-smoke",),
    )
    setattr(module, f"build_{adapter_name}_session", builder)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.chdir(tmp_path)

    config = tmp_path / "robot.yaml"
    config.write_text(
        "adapter: installed_plugin_smoke\nname: installed-plugin-smoke\nagent:\n  exec_mode: stepagent\n",
        encoding="utf-8",
    )
    runtime = Runtime(tmp_path / "workspace", resource_directory=tmp_path / "resource-locks")
    monkeypatch.setattr(
        worker,
        "run_robot_task",
        lambda *_args, **_kwargs: {"result_type": "success", "output": "mock task completed"},
    )
    try:
        binding = runtime.prepare_binding(config)
        accepted = runtime.submit_task(binding.binding_id, "installed-plugin-request", "test task")
        final = runtime.wait_for_job(accepted["job_id"], timeout=10)

        assert final["phase"] == "succeeded"
        assert final["cleanup"]["released"]
        assert final["result"]["result"] == {"result_type": "success", "output": "mock task completed"}
        event_kinds = {event["kind"] for event in runtime.read_events(accepted["job_id"], limit=128)["events"]}
        assert {"accepted", "run_started", "run_finished"} <= event_kinds
        assert not runtime.get_runtime_state()["busy"]
    finally:
        runtime.close(timeout=10)
        runtime.store.close()


def test_example_server_and_task_share_one_console_handler(monkeypatch, tmp_path, capsys):
    import logging
    from unittest.mock import Mock

    import jiuwensymbiosis.runtime as runtime_module
    from jiuwensymbiosis.utils.logging import configure_logging
    from jiuwensymbiosis_gui.plugin import LaunchOptions

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "gui_plugin"))
    from jiuwensymbiosis_example_gui import plugin, webapp

    runtime = Mock()
    runtime.close.return_value = {"closed": True}

    def serve(*_args, **_kwargs):
        # Agent startup repeats framework logging configuration after GUI startup.
        configure_logging()
        logging.getLogger("jiuwensymbiosis.task").warning("single-console-message")

    root = logging.getLogger()
    original_level = root.level
    with monkeypatch.context() as patcher:
        patcher.setattr(root, "handlers", [])
        patcher.setattr(runtime_module, "Runtime", Mock(return_value=runtime))
        patcher.setattr(webapp, "serve", serve)
        try:
            assert plugin.main(LaunchOptions(config_path=tmp_path / "robot.yaml")) == 0
            # Capture the actual stream: caplog counts records, not duplicate emits.
            assert capsys.readouterr().err.count("single-console-message") == 1
        finally:
            for handler in root.handlers:
                handler.close()
            root.setLevel(original_level)
