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
