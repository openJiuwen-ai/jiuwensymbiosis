# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""run_status:结束结果 → 徽章/叙述判定(纯逻辑,不实例化任何 UI 控件)。"""

from __future__ import annotations

from jiuwensymbiosis.gui import run_status


def test_incomplete_message_explains_max_iterations_in_chinese():
    msg = run_status.incomplete_message("Max iterations reached without completion")
    assert "未完成" in msg
    assert "最大步数" in msg
    assert "Max iterations" not in msg  # 不把英文兜底串原样丢给用户


def test_incomplete_message_passthrough_for_other_text():
    assert run_status.incomplete_message("some other reason") == "未完成:some other reason"


def test_incomplete_message_empty():
    assert run_status.incomplete_message("") == "未完成:agent 结束但未确认任务已完成。"


def test_outcome_success_is_green_status():
    outcome = run_status.outcome_from_result({"ok": True, "result": {"result_type": "answer", "output": "好了"}})
    assert outcome.status == "成功"
    assert outcome.is_failure is False
    assert "完成:好了" in outcome.narration


def test_outcome_success_without_summary_drops_none_suffix():
    outcome = run_status.outcome_from_result({"ok": True, "result": {"result_type": "answer", "output": None}})
    assert outcome.status == "成功"
    assert outcome.narration == "完成"
    assert "None" not in outcome.narration


def test_outcome_max_iterations_is_incomplete_not_success():
    outcome = run_status.outcome_from_result(
        {"ok": True, "result": {"result_type": "error", "output": "Max iterations reached without completion"}}
    )
    assert outcome.status == "未完成"
    assert run_status.STATUS_COLORS["未完成"] != run_status.STATUS_COLORS["成功"]  # 非绿色
    assert "最大步数" in outcome.narration


def test_outcome_stopped():
    outcome = run_status.outcome_from_result(
        {"ok": True, "result": {"result_type": "stopped", "output": "用户已停止运行"}}
    )
    assert outcome.status == "已停止"
    assert "已停止" in outcome.narration


def test_outcome_failure_flags_diagnosis():
    outcome = run_status.outcome_from_result({"ok": False, "error": "boom"})
    assert outcome.status == "失败"
    assert outcome.is_failure is True


def test_outcome_fast_path_step_failure_is_not_success():
    # fast 结果形状：外层 ok=True(没抛异常)，但内层 steps 有 failed → 不能显示成功。
    result = {
        "ok": True,
        "result": {
            "ok": False,
            "steps_done": 8,
            "steps": [
                {"i": 6, "op": "goto_xyzr", "ok": True},
                {"i": 7, "op": "get_grasp_info_simple", "ok": False, "reason": "no_valid_depth"},
            ],
        },
    }
    outcome = run_status.outcome_from_result(result)
    assert outcome.status == "未完成"  # 非绿色「成功」
    assert outcome.narration == "未完成"  # 相机下方只留简短「未完成」,不糊长英文
    # 详细原因(失败动作 + reason)放进 detail,由界面转交「错误诊断」;不引用对不上时间线的步序号。
    assert "第 7 步" not in outcome.detail
    assert "识别并定位物体" in outcome.detail and "no_valid_depth" in outcome.detail


def test_outcome_fast_path_track_detect_failure_names_action_and_reason_in_detail():
    result = {
        "ok": True,
        "result": {
            "ok": False,
            "steps_done": 1,
            "steps": [{"i": 0, "op": "track_detect", "ok": False, "reason": "target 'red block' not detected"}],
        },
    }
    outcome = run_status.outcome_from_result(result)
    assert outcome.status == "未完成" and outcome.narration == "未完成"
    assert "识别并定位物体" in outcome.detail  # 友好动作名
    assert "red block" in outcome.detail  # reason 里的物体名对用户可见


def test_outcome_fast_path_success_is_green():
    result = {"ok": True, "result": {"ok": True, "steps_done": 13, "steps": [{"i": 0, "op": "home", "ok": True}]}}
    outcome = run_status.outcome_from_result(result)
    assert outcome.status == "成功" and outcome.narration == "完成"
