# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""layout:拖入 YAML 配置的判定、应用与「存为可选配置」。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui.workbench import registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.layout import Layout

_BASE_YAML = "env:\n  cfg:\n    low_level:\n      port: /dev/base\n"
_DROPPED_YAML = "env:\n  cfg:\n    low_level:\n      port: /dev/dropped\n"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """把 ``configs/`` 引到临时目录后装配整页,免得用例写进仓库配置。"""
    monkeypatch.setattr(registry, "configs_dir", lambda: tmp_path)
    body_key = registry.list_bodies()[0].key
    body_dir = registry.get_body(body_key).config_path().parent
    body_dir.mkdir(parents=True, exist_ok=True)
    registry.get_body(body_key).config_path().write_text(_BASE_YAML, encoding="utf-8")
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "locks"))
    return Layout(AppState(workspace=str(tmp_path / "workspace")))


def _drop(layout: Layout, name: str, text: str) -> None:
    layout._on_yaml_dropped(SimpleNamespace(args=[{"name": name, "text": text}]))


def test_external_launch_config_survives_save_refresh_but_not_body_change(tmp_path, monkeypatch):
    from jiuwensymbiosis_gui.workbench.pages.home_view import HomeView

    monkeypatch.setattr(registry, "configs_dir", lambda: tmp_path / "configs")
    source = tmp_path / "external.yaml"
    source.write_text(_BASE_YAML, encoding="utf-8")
    state = AppState(workspace=str(tmp_path / "workspace"))
    state.current_config_file = str(source)
    home = HomeView(state, on_run=lambda _: None, on_config=lambda _: None)
    source.write_text(_DROPPED_YAML, encoding="utf-8")
    home.reload_configs()
    assert state.current_config_file == str(source)
    assert home._config_file.value == str(source)
    home._on_body_change()
    assert state.current_config_file is None
    assert str(source) not in home._config_file.options


def test_restart_does_not_spawn_until_application_cleanup_is_confirmed(layout, monkeypatch):
    from unittest.mock import Mock

    replacement = Mock()
    monkeypatch.setattr("jiuwensymbiosis_gui.workbench.app.spawn_replacement", replacement)
    monkeypatch.setattr(layout._state.host, "close", lambda: {"closed": False})
    layout._do_restart()
    replacement.assert_not_called()


def test_refreshed_client_can_request_shutdown_of_existing_maintenance(layout):
    from unittest.mock import Mock

    tool = Mock()
    tool.is_running.return_value = True
    tool.can_dispose.return_value = False
    layout._state.host.start_tool(tool)
    layout._confirm_quit()
    assert layout._quit_dialog.value is True
    layout._do_quit()
    tool.stop.assert_called()
    assert layout._state.host.is_closing
    assert not layout._bye_dialog.value


def test_jumping_to_a_config_field_loads_the_form_first(layout):
    """切标签不会重建表单(``_on_nav`` 认标签名,``_goto`` 传的是 Tab 对象),
    不先 load 就会误报「表单里没有这个字段」—— 而字段没填正是跳过来的原因。"""
    layout._open_config_field("env.cfg.low_level.camera_serial")

    assert layout._tabs.value is layout._config_tab
    assert layout._config._form_tabs.value == "机器人参数"


def test_a_run_will_not_start_while_a_tool_still_holds_the_hardware(layout):
    """标定的自动采集中途停不下来:硬开跑会变成两边同时占相机、同时对机械臂下指令。"""
    layout._tools.release_hardware = lambda **_kw: False
    started: list[str] = []
    layout._run.attach = lambda engine: started.append("attached")

    layout._start_run(layout._state.current_task)

    assert started == []
    assert layout._state.engine is None


class TestReturnToStart:
    @staticmethod
    def _armed(layout) -> list[list[float]]:
        """给 layout 一份可用快照与一个记录回位请求的假引擎。"""
        layout._state.current_body = registry.list_bodies()[0].key
        layout._state.remember_start_pose(layout._state.current_body, [1.0, 2.0])
        returned: list[list[float]] = []
        layout._state.engine = SimpleNamespace(
            is_running=lambda: False,
            start_return_to=returned.append,
        )
        return returned

    def test_a_valid_snapshot_opens_the_confirmation(self, layout):
        self._armed(layout)

        layout._confirm_return_to_start()

        assert layout._return_dialog.value is True

    def test_switching_body_invalidates_the_snapshot(self, layout):
        self._armed(layout)
        layout._state.current_body = "some-other-body"

        layout._confirm_return_to_start()

        assert layout._return_dialog.value is False

    def test_a_running_task_blocks_the_confirmation(self, layout):
        self._armed(layout)
        layout._state.engine = SimpleNamespace(is_running=lambda: True)

        layout._confirm_return_to_start()

        assert layout._return_dialog.value is False

    def test_a_tool_holding_the_arm_blocks_the_move(self, layout):
        """回位是真机自主运动,和松力矩摆位抢机械臂会同时对它下指令。"""
        returned = self._armed(layout)
        layout._tools.release_hardware = lambda **_kw: False

        layout._do_return_to_start()

        assert returned == []

    def test_confirmed_move_targets_the_recorded_joints(self, layout):
        returned = self._armed(layout)
        layout._tools.release_hardware = lambda **_kw: True

        layout._do_return_to_start()

        assert returned == [[1.0, 2.0]]

    def test_a_snapshot_needs_a_body(self, layout):
        layout._state.current_body = None
        layout._remember_start_pose([1.0, 2.0])

        assert layout._state.start_pose_joints() is None


def test_drop_of_body_config_opens_dialog_defaulting_to_apply_only(layout):
    _drop(layout, "arm.local.yaml", _DROPPED_YAML)
    assert layout._drop_dialog.value is True
    assert layout._drop_name.text == "arm.local.yaml"
    assert layout._drop_choice.value == "apply"


def test_drop_of_non_body_yaml_is_rejected(layout):
    _drop(layout, "tasks.yaml", "tasks: []\n")
    assert layout._drop_dialog.value is False


def test_apply_only_updates_config_without_writing_file(layout):
    _drop(layout, "arm.local.yaml", _DROPPED_YAML)
    layout._confirm_drop()

    state = layout._state
    assert state.current_config().get("env.cfg.low_level.port") == "/dev/dropped"
    assert not registry.body_config_path(state.current_body, "arm.local").exists()


def test_saving_dropped_config_lands_in_body_dir_and_dropdown(layout):
    _drop(layout, "arm.local.yaml", _DROPPED_YAML)
    layout._drop_choice.set_value("save")
    layout._confirm_drop()

    state = layout._state
    saved = registry.body_config_path(state.current_body, "arm.local")
    assert saved.read_text(encoding="utf-8") == _DROPPED_YAML
    assert str(saved) in layout._home._config_file.options  # 主页下拉即刻多出这份配置
    assert state.current_config().get("env.cfg.low_level.port") == "/dev/dropped"


class TestRerunTakesTheCurrentConfig:
    """「重新执行」跑的是配置页此刻的配置,不是上次开跑时那份快照。"""

    def _finished_run(self, layout: Layout):
        """装出一次跑完的运行:引擎记着本体/任务,界面随后改了配置。"""
        layout._tools.release_hardware = lambda **_kw: True
        layout._run.attach = lambda engine: None
        layout._start_run(layout._state.current_task)
        return layout._state.engine

    def test_edits_made_after_the_run_reach_the_rerun(self, layout):
        engine = self._finished_run(layout)
        assert engine is not None
        layout._state.current_config().set("env.cfg.low_level.port", "/dev/edited")

        layout._rerun()

        fresh = layout._state.engine
        assert fresh is not engine
        assert fresh._config.get("env.cfg.low_level.port") == "/dev/edited"

    def test_the_rerun_keeps_the_body_and_task_that_ran(self, layout):
        engine = self._finished_run(layout)
        body_key, task_key = engine.body_key, engine.task_key
        # 界面切走本体后重跑:重跑的仍是刚才那个本体/任务。
        layout._state.current_body = "some-other-body"

        layout._rerun()

        fresh = layout._state.engine
        assert fresh.body_key == body_key and fresh.task_key == task_key

    def test_switching_config_keeps_new_snapshot_and_source_together(self, layout, tmp_path):
        from jiuwensymbiosis.runtime import prepare_binding

        previous = self._finished_run(layout)
        source = tmp_path / "new-config" / "robot.yaml"
        source.parent.mkdir()
        source.write_text("adapter: piper\ncan_port: can_new\ncalib_path: calibration.json\n", encoding="utf-8")
        layout._state.current_config_file = str(source)
        layout._rerun()

        fresh = layout._state.engine
        assert fresh.config_source == source
        assert fresh.config_source != previous.config_source
        binding = prepare_binding(fresh.config_source, config_snapshot=fresh._config.data, include_sidecars=False)
        assert binding.config_data()["calib_path"] == str(source.parent / "calibration.json")
