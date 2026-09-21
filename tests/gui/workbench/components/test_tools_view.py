# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""tools_view:工具挂载、面板切换与硬件释放。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis_gui.workbench import board_print, registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.calibration_engine import CalibrationEngine
from jiuwensymbiosis_gui.workbench.pages import calibration_view
from jiuwensymbiosis_gui.workbench.pages.calibration_view import _DOUBLE_TAP_S
from jiuwensymbiosis_gui.workbench.pages.tools_view import _TOOLS, ToolsView


@pytest.fixture(autouse=True)
def board_print_available(monkeypatch):
    """把标定板依赖探测与绘图桩掉:两者都在 ``[calib]`` extra 里,本文件测的是向导装配。

    缺 OpenCV 时 ``refresh()`` 会挂出依赖提示条并提前返回,硬件行、量取提示与预览都不会
    被填。预览桩取 ``BoardParams`` 的 repr,所以板参数变了预览就变、只改实测值则不变。
    绘图本身由 ``test_board_print.py`` 覆盖。
    """
    monkeypatch.setattr(calibration_view, "board_print_dependency_hint", lambda: None)
    monkeypatch.setattr(board_print, "render_preview_data_uri", lambda params, **_kw: f"preview:{params}")


@pytest.fixture(autouse=True)
def isolate_runtime_dir(tmp_path, monkeypatch):
    """Keep local resource locks and runtime state inside pytest's temporary directory."""
    monkeypatch.setenv("JIUWENSYMBIOSIS_RUNTIME_DIR", str(tmp_path / "runtime"))


@pytest.fixture
def tools(tmp_path):
    return ToolsView(AppState(workspace=str(tmp_path)))


def _descendants(element):
    for child in element.default_slot.children:
        yield child
        yield from _descendants(child)


def _teach_stage(wizard) -> str:
    return wizard._teach_stage()


def _rigidity(*, std: float, worst: float, rot: float) -> dict:
    """引擎 ``_quality_payload`` 产出的刚性字段(照它的形状给全,别只给断言要用的那几个)。"""
    return {
        "invariant_frame": "flange_target",
        "translation_mm": worst,
        "translation_std_mm": std,
        "rotation_deg": rot,
    }


def _warn_lines(wizard) -> list[str]:
    """保存框里当前挂着的几条提示,按屏幕上的先后。"""
    return [child.text for child in wizard._save_warns.default_slot.children if getattr(child, "text", None)]


def _key(name: str, *, repeat: bool = False):
    """一次 keydown 事件(``ui.keyboard`` 交给 on_key 的形状)。"""
    return SimpleNamespace(
        action=SimpleNamespace(keydown=True, keyup=False, repeat=repeat),
        key=SimpleNamespace(
            name=name,
            space=name == " ",
            escape=name == "Escape",
            enter=name == "Enter",
        ),
    )


def _space(*, repeat: bool = False):
    return _key(" ", repeat=repeat)


def _teaching_engine(**extra):
    """一个"正在示教"的假引擎;``extra`` 覆盖要断言的那个方法。"""
    calls: list[str] = []
    engine = SimpleNamespace(
        drain=list,
        is_running=lambda: True,
        confirm_teaching=lambda: calls.append("confirm"),
        record_waypoint=lambda: calls.append("record"),
        pause_teaching=lambda: calls.append("pause"),
        resume_teaching=lambda: calls.append("resume"),
        abort_teaching=lambda: calls.append("abort"),
        finish_teaching=lambda: calls.append("finish"),
        **extra,
    )
    return engine, calls


def test_both_tools_get_their_own_panel(tools):
    assert {key for key, _ in _TOOLS} == set(tools._panels)
    assert tools._calibration is not None


def test_only_the_selected_panel_is_visible(tools):
    assert tools._panels["perception"].visible
    assert not tools._panels["calibration"].visible

    tools._select_tool("calibration")

    assert tools._panels["calibration"].visible
    assert not tools._panels["perception"].visible


def test_switching_tools_releases_the_one_being_left(tools, monkeypatch):
    """相机与机械臂总线是单占用的:切走时必须先放手,否则下一个工具连不上。"""
    stopped: list[str] = []
    monkeypatch.setattr(tools, "stop_preview", lambda **_kw: stopped.append("perception"))
    monkeypatch.setattr(tools._calibration, "stop", lambda **_kw: stopped.append("calibration"))

    tools._select_tool("calibration")
    assert stopped == ["perception"]

    tools._select_tool("perception")
    assert stopped == ["perception", "calibration"]


def test_reselecting_the_current_tool_does_not_stop_it(tools, monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr(tools, "stop_preview", lambda **_kw: stopped.append("perception"))
    tools._select_tool("perception")
    assert stopped == []


def test_release_hardware_covers_every_tool(tools, monkeypatch):
    """离开工具页 / 重启 / 开始运行都走这里,必须覆盖两个工具而不只是预览。"""
    stopped: list[str] = []
    monkeypatch.setattr(tools, "stop_preview", lambda **_kw: stopped.append("perception"))
    monkeypatch.setattr(tools._calibration, "stop", lambda **_kw: stopped.append("calibration") or True)

    assert tools.release_hardware()
    assert stopped == ["perception", "calibration"]


def test_release_hardware_reports_a_tool_that_would_not_let_go(tools, monkeypatch):
    """标定的自动采集中途停不下来。调用方要据此别去开相机,而不是以为已经归还了。"""
    monkeypatch.setattr(tools._calibration, "stop", lambda **_kw: False)

    assert not tools.release_hardware()


class TestCalibrationPreconditions:
    def test_blocks_until_a_body_is_chosen(self, tools):
        calibration = tools._calibration
        assert calibration._blocker.visible
        assert "选择一个本体" in calibration._blocker.text

    def test_body_without_a_task_still_resolves_config(self, tools):
        """标定与任务无关 —— 只选了本体也要能读出该本体的硬件信息。"""
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()

        assert not calibration._blocker.visible
        assert calibration._hw_rows.default_slot.children, "应列出相机位置等硬件信息"

    def test_camera_mount_is_stated_in_plain_words_never_as_a_question_mark(self, tools):
        """相机安装方式是机型固有属性(config 里是 Literal),配置没写也要说得出来。"""
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration._low_level = lambda _body: {}  # 模拟运行配置没写 camera_mount
        calibration.refresh()

        texts = [child.text for child in _descendants(calibration._hw_rows) if getattr(child, "text", None)]
        assert any("不随机械臂移动" in text for text in texts), texts
        assert not any("?" in text for text in texts), texts

    def test_measure_hint_follows_the_longer_side(self, tools):
        """量一整排再除比量单格准,格数要跟着板参数走。"""
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        assert "7 个格子" in calibration._measure_hint.text

        calibration._in_sy.set_value(4)
        calibration._in_sx.set_value(9)

        assert "9 个格子" in calibration._measure_hint.text

    def test_changing_a_value_refreshes_without_leaving_the_field(self, tools):
        """点上下箭头调数不会让输入框失焦 —— 回调必须绑在值上,不能只绑 blur。"""
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        before = calibration._board_preview.source

        calibration._in_square.set_value(26.0)

        assert calibration._board_spec().square_size_mm == 26.0
        assert calibration._board_preview.source != before, "预览应随参数立即重画"


class TestMissingCameraSerial:
    """随包配置里序列号是占位符,所以这条提示是新用户必经的一步。"""

    @pytest.fixture
    def opened(self):
        return []

    @pytest.fixture
    def wizard(self, tmp_path, opened):
        tools = ToolsView(AppState(workspace=str(tmp_path)), on_open_config=opened.append)
        tools._state.current_body = "so101"
        tools._calibration.refresh()
        return tools._calibration

    def test_the_config_word_jumps_straight_to_that_field(self, wizard, opened):
        """只说「去配置页填」等于让人自己在几个分组里翻,提示要能点到那一栏。"""
        link = next(c for c in _descendants(wizard._hw_rows) if getattr(c, "text", "") == "「配置」")
        next(iter(link._event_listeners.values())).handler(None)

        assert opened == ["env.cfg.low_level.camera_serial"]

    def test_a_filled_serial_needs_no_link(self, wizard):
        wizard._low_level = lambda _body: {"camera_mount": "eye_to_hand", "camera_serial": "1234"}
        wizard.refresh()

        texts = [c.text for c in _descendants(wizard._hw_rows) if getattr(c, "text", None)]
        assert "1234" in texts
        assert "「配置」" not in texts


class TestMeasuredSize:
    """实测边长只喂给标定,不能倒灌回画图。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_measuring_does_not_change_what_gets_printed(self, wizard):
        """量完发现打印缩了,改的是标定用的数;图和 PDF 必须还按设定值出,否则越修越偏。"""
        drawn_before = wizard._board_preview.source

        wizard._in_measured.set_value(19.2)

        assert wizard._board_spec().square_size_mm == 20.0
        assert wizard._board_preview.source == drawn_before

    def test_measured_size_scales_the_marker_proportionally(self, wizard):
        """打印缩放是等比的:方格缩多少 marker 就缩多少,比例不能变。"""
        wizard._in_measured.set_value(19.0)
        board = wizard._measured_board()

        assert board.square_size_mm == 19.0
        assert board.marker_size_mm == pytest.approx(15.0 * 19.0 / 20.0)
        spec = wizard._board_spec()
        assert board.marker_size_mm / board.square_size_mm == pytest.approx(spec.marker_size_mm / spec.square_size_mm)

    def test_shrunk_measurement_never_trips_the_marker_check(self, wizard):
        """只缩 square 不缩 marker 会误报「marker 不能大于方格」—— 等比缩放后不该发生。"""
        wizard._in_square.set_value(20.0)
        wizard._in_marker.set_value(19.0)  # 合法但很接近
        wizard._in_measured.set_value(12.0)  # 打印缩得很厉害

        wizard._measured_board().validate()

    def test_default_marker_survives_an_invalid_board(self, wizard):
        """参数报错时也要更新「默认」标记,不能卡在上一次的状态。"""
        wizard._in_measured.set_value(19.2)
        assert "（默认）" not in wizard._in_measured._props["label"]

        wizard._in_marker.set_value(99.0)  # marker > square,板参数非法
        wizard._in_measured.set_value(wizard._in_square.value)

        assert "（默认）" in wizard._in_measured._props["label"]


class TestEngineFreshness:
    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_capture_uses_the_current_form_values(self, wizard, monkeypatch):
        """拍照次数在示教之后才轮到用户填,复用示教时的 setup 会拿老参数静默跑。"""
        monkeypatch.setattr(CalibrationEngine, "start_capture", lambda _self, **_kw: None)
        wizard._new_engine()
        assert wizard._setup.n_stations == 20

        wizard._in_stations.set_value(30)
        wizard._start_capture()

        assert wizard._setup.n_stations == 30

    def test_capture_carries_the_measured_board_not_the_drawn_one(self, wizard):
        wizard._in_measured.set_value(19.0)
        wizard._new_engine()

        assert wizard._setup.board.square_size_mm == 19.0

    def test_changing_the_nominal_size_resets_the_measured_one(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()

        calibration._in_square.set_value(24.0)

        assert calibration._in_measured.value == 24.0

    def test_safety_checkboxes_gate_the_teaching_step(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        assert not calibration._btn_to_teach.enabled, "未确认安全事项前不得进入示教"

        calibration._ck_board.value = True
        calibration._ck_estop.value = True
        calibration._refresh_prepare_gate()

        assert calibration._btn_to_teach.enabled


class TestWizardNavigation:
    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        calibration._ck_board.value = True
        calibration._ck_estop.value = True
        calibration._refresh_prepare_gate()
        return calibration

    def test_can_step_back_to_an_earlier_step(self, wizard):
        """走到后面才发现板参数填错时,不该只能从头再来。"""
        wizard._goto("teach")
        assert wizard._step == "teach"

        wizard._goto_back("prepare")

        assert wizard._step == "prepare"

    def test_cannot_jump_forward_past_a_gate(self, wizard):
        """步骤条只能往回点;往前必须走各步自己的出口,那里带着前置校验。"""
        wizard._goto_back("capture")
        assert wizard._step == "prepare"

        wizard._goto_back("prepare")
        assert wizard._step == "prepare"

    def test_returning_to_teaching_clears_the_previous_round(self, wizard):
        wizard._goto("teach")
        wizard._waypoints = 5
        wizard._wp_label.set_text("5")
        wizard._teaching_done = True

        wizard._goto("teach")

        assert wizard._waypoints == 0
        assert wizard._wp_label.text == "0"
        assert not wizard._teaching_done
        assert _teach_stage(wizard) == "idle", "重新示教要收回上一轮的指引与按钮"


class TestTeachStages:
    """示教这一步一次只摆当前该做的动作,不把四个按钮平铺着(其中两个还是灰的)。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_exactly_one_action_block_is_on_screen(self, wizard):
        assert [key for key, panel in wizard._teach_stages.items() if panel.visible] == ["idle"]

    def test_starting_takes_the_start_button_away(self, wizard, monkeypatch):
        """连接期间没有任何可点的东西,留着「开始示教」只会被重复点。"""
        monkeypatch.setattr(
            wizard, "_new_engine", lambda: SimpleNamespace(start_teaching=lambda: None, can_dispose=lambda: True)
        )

        wizard._start_teaching()

        assert _teach_stage(wizard) == "busy"

    def test_only_the_torque_release_button_is_red(self, wizard):
        """一屏里只留一个危险色;红色到处都是,它就不再是警告。"""
        assert wizard._btn_teach_confirm._props.get("color") == "negative"

    def test_every_teaching_button_looks_the_same(self, wizard):
        """示教中的动作是同一类,就用同一种样式;主次靠排列,不靠给每个按钮换个长相。"""
        buttons = (wizard._btn_record, wizard._btn_pause, wizard._btn_resume, wizard._btn_teach_done)
        assert all(button._props.get("flat") for button in buttons)
        assert not any(button._props.get("color") == "negative" for button in buttons)


class TestTeachingPrompt:
    """示教指引按设备实现的端口给,不按机型;确认门放行前不开放记录。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_no_prompt_before_the_device_reports_its_mode(self, wizard):
        """连接前不知道松不松力矩,所以不能先摆一条可能是假话的警告。"""
        assert _teach_stage(wizard) == "idle"

    def test_torque_release_body_gets_the_falling_warning(self, wizard):
        wizard._on_teach_mode({"mode": "manual_guidance"})

        assert _teach_stage(wizard) == "armed"
        assert "失去力矩" in wizard._teach_hint.text

    def test_external_teach_body_is_not_told_the_arm_will_drop(self, wizard):
        wizard._on_teach_mode({"mode": "external_snapshot"})

        assert _teach_stage(wizard) == "armed"
        assert "失去力矩" not in wizard._teach_hint.text
        assert "示教器" in wizard._teach_hint.text

    def test_recording_opens_only_after_the_gate_is_confirmed(self, wizard):
        released: list[bool] = []
        wizard._engine = SimpleNamespace(confirm_teaching=lambda: released.append(True), drain=list)
        wizard._on_teach_mode({"mode": "manual_guidance"})
        assert _teach_stage(wizard) == "armed", "确认之前不该出现「记录当前姿态」"

        wizard._confirm_teaching()

        assert released == [True], "确认按钮必须真的放行工作线程,而不只是改界面"
        assert _teach_stage(wizard) == "teaching"

    def test_space_tapped_twice_confirms_without_the_mouse(self, wizard):
        """两只手都在托机械臂时够不着鼠标 —— 键盘要能盲按确认。"""
        released: list[bool] = []
        wizard._engine = SimpleNamespace(
            confirm_teaching=lambda: released.append(True), drain=list, is_running=lambda: True
        )
        wizard._on_teach_mode({"mode": "manual_guidance"})

        wizard._on_key(_space())
        assert released == [], "一下不算数:这一下的后果是机械臂失力矩下坠"

        wizard._on_key(_space())
        assert released == [True]
        assert _teach_stage(wizard) == "teaching"

    def test_two_taps_far_apart_are_not_a_double_tap(self, wizard):
        released: list[bool] = []
        wizard._engine = SimpleNamespace(
            confirm_teaching=lambda: released.append(True), drain=list, is_running=lambda: True
        )
        wizard._on_teach_mode({"mode": "manual_guidance"})
        wizard._on_key(_space())
        wizard._space_at -= _DOUBLE_TAP_S + 1.0  # 把第一下挪到判定窗口之外

        wizard._on_key(_space())

        assert released == []

    def test_space_does_nothing_while_no_gate_is_waiting(self, wizard):
        """确认门没亮时空格必须是死键:它在别的步骤里只是普通按键。"""
        released: list[bool] = []
        wizard._engine = SimpleNamespace(
            confirm_teaching=lambda: released.append(True), drain=list, is_running=lambda: True
        )

        wizard._on_key(_space())
        wizard._on_key(_space())

        assert released == []

    def test_held_down_space_is_not_a_double_tap(self, wizard):
        released: list[bool] = []
        wizard._engine = SimpleNamespace(
            confirm_teaching=lambda: released.append(True), drain=list, is_running=lambda: True
        )
        wizard._on_teach_mode({"mode": "manual_guidance"})

        wizard._on_key(_space())
        wizard._on_key(_space(repeat=True))

        assert released == []

    def test_prompt_is_reset_after_the_round_ends(self, wizard):
        wizard._on_teach_mode({"mode": "manual_guidance"})
        wizard._on_teaching_done({"count": 12, "mode": "manual_guidance"})

        assert _teach_stage(wizard) == "idle", "示教结束要收回指引与按钮"
        assert "力矩已恢复" in wizard._teach_status.text

    def test_external_teach_body_is_not_told_torque_was_restored(self, wizard):
        wizard._on_teaching_done({"count": 12, "mode": "external_snapshot"})

        assert "力矩" not in wizard._teach_status.text


class TestTeachShortcuts:
    """示教全程双手在托机械臂,四个动作都要能盲按;空格固定表示"改变机械臂的软硬"。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    @pytest.fixture
    def teaching(self, wizard):
        """已确认、正在示教的向导;返回 (向导, 引擎调用记录)。"""
        engine, calls = _teaching_engine()
        wizard._engine = engine
        wizard._on_teach_mode({"mode": "manual_guidance", "can_pause": True})
        wizard._confirm_teaching()
        calls.clear()
        return wizard, calls

    def test_enter_records_a_pose(self, teaching):
        wizard, calls = teaching
        wizard._on_key(_key("Enter"))
        assert calls == ["record"]

    def test_s_finishes_and_moves_on(self, teaching):
        wizard, calls = teaching
        wizard._waypoints = 12  # 够下限,不触发「再点一次」的警告
        wizard._on_key(_key("s"))
        assert calls == ["finish"]

    def test_one_space_pauses_because_holding_the_arm_is_the_safe_direction(self, teaching):
        wizard, calls = teaching
        wizard._on_key(_space())
        assert calls == ["pause"]

    def test_resuming_needs_two_taps_because_the_arm_goes_limp_again(self, teaching):
        wizard, calls = teaching
        wizard._on_paused({"paused": True})
        assert _teach_stage(wizard) == "paused"

        wizard._on_key(_space())
        assert calls == [], "一下不算数:继续意味着力矩再次掉下去"

        wizard._on_key(_space())
        assert calls == ["resume"]

    def test_the_ui_waits_for_the_engine_before_claiming_the_arm_is_held(self, teaching):
        """先改界面再动力矩,等于叫人在还软着的时候松手。"""
        wizard, _calls = teaching
        wizard._on_key(_space())
        assert _teach_stage(wizard) == "teaching"

        wizard._on_paused({"paused": True})
        assert _teach_stage(wizard) == "paused"

    def test_esc_asks_again_before_discarding_recorded_poses(self, teaching):
        wizard, calls = teaching
        wizard._waypoints = 9

        wizard._on_key(_key("Escape"))
        assert calls == [], "一下 Esc 不该丢掉九个姿态"

        wizard._on_key(_key("Escape"))
        assert calls == ["abort"]

    def test_esc_aborts_at_once_when_nothing_was_recorded(self, teaching):
        wizard, calls = teaching
        wizard._on_key(_key("Escape"))
        assert calls == ["abort"]

    def test_aborting_clears_the_round_instead_of_advancing(self, teaching):
        wizard, _calls = teaching
        wizard._waypoints = 5
        wizard._wp_label.set_text("5")

        wizard._on_aborted({"reason": "用户中止示教"})

        assert _teach_stage(wizard) == "idle"
        assert wizard._wp_label.text == "0"
        assert wizard._step != "capture", "中止没有产出归档,不能像正常结束那样进采集步骤"

    def test_a_body_that_cannot_pause_gets_no_pause_button(self, wizard):
        wizard._on_teach_mode({"mode": "external_snapshot", "can_pause": False})
        assert not wizard._btn_pause.visible

    def test_keys_are_dead_once_the_worker_is_gone(self, teaching):
        """线程已经退出时按键只会往一个没人读的队列里塞命令。"""
        wizard, calls = teaching
        wizard._engine.is_running = lambda: False

        wizard._on_key(_key("Enter"))
        wizard._on_key(_key("Escape"))

        assert calls == []


class TestPausedRecording:
    """暂停后机械臂被托住不动,这一个姿态可以记,但只能记一次。"""

    @pytest.fixture
    def paused(self, tools):
        tools._state.current_body = "so101"
        wizard = tools._calibration
        wizard.refresh()
        engine, calls = _teaching_engine()
        wizard._engine = engine
        wizard._on_teach_mode({"mode": "manual_guidance", "can_pause": True})
        wizard._confirm_teaching()
        wizard._on_paused({"paused": True})
        calls.clear()
        return wizard, calls

    def test_the_held_pose_can_be_recorded_once(self, paused):
        wizard, calls = paused
        wizard._on_key(_key("Enter"))
        assert calls == ["record"]
        assert not wizard._btn_record.enabled

    def test_a_second_record_is_refused_because_it_would_be_a_duplicate(self, paused):
        wizard, calls = paused
        wizard._on_key(_key("Enter"))
        wizard._on_key(_key("Enter"))
        assert calls == ["record"]

    def test_resuming_opens_recording_again(self, paused):
        wizard, calls = paused
        wizard._on_key(_key("Enter"))

        wizard._on_paused({"paused": False})

        assert wizard._btn_record.enabled
        wizard._on_key(_key("Enter"))
        assert calls == ["record", "record"]

    def test_pausing_warns_before_the_arm_goes_limp_again(self, paused):
        """继续示教和最初的确认是同一件事(力矩会掉),就该给同一条警告。"""
        wizard, _calls = paused
        assert wizard._pause_hint.visible
        assert "失去力矩" in wizard._pause_hint.text

    def test_the_status_line_says_it_is_paused_instead_of_adding_a_banner(self, paused):
        wizard, _calls = paused
        assert "暂停" in wizard._teach_status.text


class TestSavingTheResult:
    """标定文件常常是用户手工指过去、正在用的那一份,只能由用户点了保存才写。"""

    @pytest.fixture
    def wizard(self, tools, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "configs_dir", lambda: tmp_path / "configs")
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    @pytest.fixture
    def solved(self, wizard, tmp_path):
        """一份求解完成、还没落盘的结果;暂存文件已在工作区就位。"""
        pending = tmp_path / "pending" / "so101_eye_to_hand.json"
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text('{"schema_version": 2}', encoding="utf-8")
        pending.with_name("so101_eye_to_hand.stations.npz").write_bytes(b"stations")
        wizard._on_capture_done({"candidate": False, "artifact_path": str(pending), "quality": {}})
        return wizard

    def test_solving_writes_nothing_into_the_config_tree(self, wizard, tmp_path):
        """求解产物落在工作区暂存目录 —— configs/ 底下那份可能正被运行配置指着用。"""
        wizard._new_engine()
        assert registry.configs_dir() not in wizard._setup.out_path.parents

    def test_a_free_path_offers_only_save(self, solved, tmp_path):
        solved._in_save_path.set_value(str(tmp_path / "fresh.json"))

        assert solved._btn_save.enabled
        assert not solved._btn_save_backup.enabled
        assert not any("覆盖" in line for line in _warn_lines(solved))

    def test_an_occupied_path_warns_and_offers_the_backup(self, solved, tmp_path):
        occupied = tmp_path / "mine.json"
        occupied.write_text("我手工指过去的那一份", encoding="utf-8")

        solved._in_save_path.set_value(str(occupied))

        assert solved._btn_save.enabled
        assert solved._btn_save_backup.enabled
        assert any("覆盖" in line for line in _warn_lines(solved))

    def test_both_save_buttons_look_alike(self, solved):
        """两颗都是"保存",一实心一扁平会让人以为它们不是一类动作。"""
        assert solved._btn_save._props.get("color") == solved._btn_save_backup._props.get("color")
        assert bool(solved._btn_save._props.get("flat")) == bool(solved._btn_save_backup._props.get("flat"))

    def test_the_not_yet_written_warning_comes_before_the_overwrite_one(self, solved, tmp_path):
        """两条都是"动手之前要知道",范围大的先说。"""
        occupied = tmp_path / "mine.json"
        occupied.write_text("原文件", encoding="utf-8")

        solved._in_save_path.set_value(str(occupied))

        lines = _warn_lines(solved)
        assert len(lines) == 2
        assert "还没写盘" in lines[0]
        assert "覆盖" in lines[1]

    def test_saving_clears_the_not_yet_written_warning(self, solved, tmp_path):
        solved._in_save_path.set_value(str(tmp_path / "fresh.json"))

        solved._save_result(backup=False)

        assert not any("还没写盘" in line for line in _warn_lines(solved))

    def test_nothing_is_written_until_save_is_clicked(self, solved, tmp_path):
        target = tmp_path / "mine.json"
        target.write_text("原文件", encoding="utf-8")

        solved._in_save_path.set_value(str(target))

        assert target.read_text(encoding="utf-8") == "原文件"
        assert not solved._btn_apply.enabled, "没保存就没有可启用的文件"

    def test_backup_before_save_keeps_the_old_file(self, solved, tmp_path):
        target = tmp_path / "mine.json"
        target.write_text("原文件", encoding="utf-8")
        solved._in_save_path.set_value(str(target))

        solved._save_result(backup=True)

        assert '"schema_version": 2' in target.read_text(encoding="utf-8")
        backups = list(tmp_path.glob("mine.json.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "原文件"

    def test_plain_save_overwrites_as_the_warning_says(self, solved, tmp_path):
        target = tmp_path / "mine.json"
        target.write_text("原文件", encoding="utf-8")
        solved._in_save_path.set_value(str(target))

        solved._save_result(backup=False)

        assert '"schema_version": 2' in target.read_text(encoding="utf-8")
        assert not list(tmp_path.glob("mine.json.*.bak"))

    def test_the_station_archive_travels_with_it(self, solved, tmp_path):
        """站点归档是唯一能重新求解的原始数据,留在暂存目录等于丢掉。"""
        solved._in_save_path.set_value(str(tmp_path / "mine.json"))

        solved._save_result(backup=False)

        assert (tmp_path / "mine.stations.npz").read_bytes() == b"stations"

    def test_applying_points_at_what_was_saved_not_the_staged_copy(self, solved, tmp_path):
        target = tmp_path / "mine.json"
        solved._in_save_path.set_value(str(target))

        solved._save_result(backup=False)

        assert solved._btn_apply.enabled
        assert solved._saved_path == target

    def test_a_rejected_calibration_offers_no_save_at_all(self, wizard):
        wizard._on_capture_done({"candidate": True, "artifact_path": None, "quality": {}})

        assert not wizard._save_box.visible
        assert not wizard._btn_apply.enabled


class TestCapturePage:
    """采集这一步只有两个状态:没开跑(参数 + 开始)和跑起来了(进度)。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_the_photo_count_is_explained_against_what_was_just_taught(self, wizard):
        """光给一个数字没人知道该填多少 —— 要和他刚示教的姿态数对上。"""
        wizard._waypoints = 16
        wizard._refresh_capture_note()

        assert "16 个姿态" in wizard._capture_note.text
        assert "20 个位置" in wizard._capture_note.text

    def test_progress_replaces_the_form_while_the_arm_moves(self, wizard, monkeypatch):
        monkeypatch.setattr(CalibrationEngine, "start_capture", lambda _self, **_kw: None)

        wizard._start_capture()

        assert wizard._capture_stages["running"].visible
        assert not wizard._capture_stages["idle"].visible

    def test_progress_says_where_it_is_and_whether_the_board_is_seen(self, wizard):
        """人站在旁边等,他要的是"到第几个了、认出来没有",不是一屏日志。"""
        wizard._on_station({"index": 7, "total": 20, "ok": True})

        assert "7 / 20" in wizard._progress_label.text
        assert "1 张" in wizard._progress_label.text
        assert wizard._capture_progress.value == pytest.approx(0.35)

    def test_a_second_capture_counts_from_zero(self, wizard, monkeypatch):
        """两次采集彼此独立(第二轮重走轨迹、重新求解),进度也不能接着上一轮数。"""
        monkeypatch.setattr(CalibrationEngine, "start_capture", lambda _self, **_kw: None)
        wizard._on_station({"index": 1, "total": 20, "ok": True})
        wizard._on_station({"index": 2, "total": 20, "ok": True})

        wizard._start_capture()
        wizard._on_station({"index": 1, "total": 20, "ok": True})

        assert "认出标定板 1 张" in wizard._progress_label.text

    def test_undetected_stations_do_not_count_as_seen(self, wizard):
        wizard._on_station({"index": 1, "total": 20, "ok": False})
        wizard._on_station({"index": 2, "total": 20, "ok": True})

        assert "1 张" in wizard._progress_label.text

    def test_finishing_puts_the_form_back(self, wizard):
        wizard._set_capture_stage("running")
        wizard._on_capture_done({"candidate": True, "artifact_path": None, "quality": {}})

        assert wizard._capture_stages["idle"].visible


class TestRejectedResultPage:
    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    @staticmethod
    def _expansions(wizard) -> list[str]:
        return [
            child.props.get("label", "")
            for child in _descendants(wizard._result_detail)
            if child.tag == "q-expansion-item"
        ]

    def test_the_two_detail_sections_have_different_titles(self, wizard):
        """两块同名折叠区并列时,点开一个的人不会想到下面还有一个。"""
        wizard._on_capture_done(
            {
                "candidate": True,
                "artifact_path": "/tmp/x.candidate.json",
                "quality": {"reprojection_px": {"mean": 3.1, "max": 4.0}},
                "reasons": ["reproj mean 3.10px > 2.0px"],
                "failed_checks": ["reproj"],
            }
        )

        titles = self._expansions(wizard)
        assert len(titles) == len(set(titles)), titles

    def test_no_empty_detail_section_when_nothing_was_solved(self, wizard):
        """站点太少时求解压根没跑,摆一个展开之后是空的折叠区只会浪费一次点击。"""
        wizard._on_capture_done({"candidate": True, "artifact_path": None, "quality": {}, "reasons": []})

        assert "技术详情" not in self._expansions(wizard)

    def test_the_dead_apply_button_is_gone(self, wizard):
        """结果不能用时「启用」永远不会亮,不该摆在那里让人点。"""
        wizard._on_capture_done({"candidate": True, "artifact_path": None, "quality": {}})

        assert not wizard._apply_row.visible
        assert not wizard._save_box.visible


class TestRemedyList:
    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    def test_one_advice_per_line_even_when_several_checks_map_to_it(self, wizard):
        """target_consistency 的平移与旋转是两个检查项,共用一条建议 —— 列两遍只是噪声。"""
        wizard._on_capture_done(
            {
                "candidate": True,
                "artifact_path": None,
                "quality": {},
                "failed_checks": ["observability_duplicates", "target_consistency_trans", "target_consistency_rot"],
                "reasons": ["1 duplicate station pair(s)"],
            }
        )

        advice = [
            child.text for child in _descendants(wizard._result_detail) if getattr(child, "text", "").startswith("• ")
        ]
        assert len(advice) == len(set(advice)), advice
        assert len(advice) == 2


class TestMetricsMatchTheGate:
    """面板显示的合格线必须就是判 accept/reject 的那条,否则会出现"良好但不通过"。"""

    @pytest.fixture
    def wizard(self, tools):
        tools._state.current_body = "so101"
        calibration = tools._calibration
        calibration.refresh()
        return calibration

    @staticmethod
    def _rows(wizard) -> list[str]:
        return [child.text for child in _descendants(wizard._result_detail) if getattr(child, "text", None)]

    def test_a_value_over_the_gate_is_never_called_good(self, wizard):
        """3.06 的 std 刚过 3.0 的线;旧版按 max 与 5.0 判,会标成「✓ 良好」。"""
        wizard._on_capture_done(
            {
                "candidate": True,
                "artifact_path": None,
                "quality": {"rigidity": _rigidity(std=3.06, worst=4.2, rot=2.03)},
                "limits": {"translation_std_mm": 3.0, "rotation_spread_deg": 2.0},
                "failed_checks": ["target_consistency_trans"],
            }
        )

        rows = self._rows(wizard)
        assert any("3.06 mm（限 3）" in row for row in rows), rows
        assert any("2.03 °（限 2）" in row for row in rows), rows
        assert "✓ 良好" not in rows

    def test_the_gate_free_mode_shows_no_rigidity_limit(self, wizard):
        """eye-in-hand 没有刚性门,就不该摆一条它并不适用的合格线。"""
        wizard._on_capture_done(
            {
                "candidate": False,
                "artifact_path": "/tmp/x.json",
                "quality": {"rigidity": _rigidity(std=3.06, worst=4.2, rot=2.03)},
                "limits": {"reproj_good_px": 1.0, "reproj_warn_px": 2.0},
            }
        )

        assert not any("限" in row for row in self._rows(wizard))
