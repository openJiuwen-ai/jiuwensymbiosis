# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""运行状态与结果文案(纯逻辑,无 Qt / 无 nicegui)。

把「运行结束结果 → 状态徽章 + 一句话叙述」的判定从视图里剥出来独立单测。关键约束:
agent 循环结束但未真正完成任务(``result_type=="error"``,最典型是达到最大步数)必须
显示为**非绿色的「未完成」**,而不是成功。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["STATUS_COLORS", "Outcome", "incomplete_message", "outcome_from_result"]

# 状态徽章配色(与运行页徽章一致)。「未完成」用橙红,明确区别于绿色「成功」。
STATUS_COLORS: dict[str, str] = {
    "准备中": "#888",
    "运行中": "#2a6bd0",
    "成功": "#1a9a4a",
    "失败": "#c0392b",
    "已停止": "#c67a00",
    "未完成": "#d35400",
}


@dataclass(frozen=True)
class Outcome:
    """一次运行结束后的界面呈现:状态标签、叙述文案、是否需要展开错误诊断。"""

    status: str
    narration: str
    is_failure: bool


def incomplete_message(summary: str) -> str:
    """把 agent 的「未完成」兜底结果译成一句面向用户的中文说明。"""
    if "Max iterations reached without completion" in summary:
        return (
            "未完成:达到最大步数仍未完成任务。可在「执行方式 → 最大步数」调高上限;"
            "若是反复识别不到物体,多为相机/检测异常,请先确认视觉是否正常。"
        )
    return f"未完成:{summary}" if summary else "未完成:agent 结束但未确认任务已完成。"


def outcome_from_result(result: dict[str, Any]) -> Outcome:
    """从 ``run_finished`` 结果推导状态徽章与叙述。

    失败(``ok`` 为假)交由调用方另跑 ``diagnose`` 渲染诊断页,故 ``is_failure=True``。
    """
    if not result.get("ok"):
        return Outcome("失败", "运行失败,已在下方「错误诊断」给出原因与处理建议。", True)
    payload = result.get("result")
    # fast 路径的结果形状是 {"ok", "steps_done", "steps":[...], ...},没有 result_type;
    # 外层 RunEngine 的 ok 只表示"没抛异常",真正成败看这里的内层 ok——否则一步 failed
    # 也会落到最后的"成功"分支。
    if isinstance(payload, dict) and "steps_done" in payload:
        if payload.get("ok"):
            return Outcome("成功", "完成", False)
        failed = next((s for s in payload.get("steps", []) if isinstance(s, dict) and not s.get("ok")), None)
        if failed is not None:
            reason = str(failed.get("reason", "")).strip()
            head = f"第 {failed.get('i')} 步（{failed.get('op', '')}）失败"
            return Outcome("未完成", f"未完成:{head}：{reason}" if reason else f"未完成:{head}。", False)
        return Outcome("未完成", "未完成:动作序列未跑完。", False)
    rtype = payload.get("result_type") if isinstance(payload, dict) else ""
    summary = payload.get("output") if isinstance(payload, dict) else payload
    if rtype == "stopped":
        return Outcome("已停止", f"已停止:{summary}", False)
    if rtype == "error":
        return Outcome("未完成", incomplete_message(str(summary)), False)
    return Outcome("成功", f"完成:{summary}" if summary else "完成", False)
