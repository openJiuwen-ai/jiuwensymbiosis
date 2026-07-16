# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run-end status + narration: map a ``run_finished`` result to a status badge and one-line narration.

Key rule: when the agent loop ends without truly completing the task (``result_type=="error"``, most
often hitting the max step count) it must show as a **non-green "incomplete"** state, not success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jiuwensymbiosis.gui import humanize

__all__ = ["STATUS_COLORS", "Outcome", "incomplete_message", "outcome_from_result"]

# Status badge colors (match the run page badges). "Incomplete" uses orange-red to clearly distinguish
# it from green "success".
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
    """How a finished run is presented: status label, narration text, and whether to open diagnostics.

    ``detail`` holds the detailed cause (often an English technical string): the area under the camera
    shows only the short ``narration``; ``detail`` is routed to the 「错误诊断」 tab + raw log so a long
    English string doesn't clutter the main view.
    """

    status: str
    narration: str
    is_failure: bool
    detail: str = ""


def incomplete_message(summary: str) -> str:
    """Turn the agent's "incomplete" fallback result into one user-facing Chinese sentence."""
    if "Max iterations reached without completion" in summary:
        return (
            "未完成:达到最大步数仍未完成任务。可在「执行方式 → 最大步数」调高上限;"
            "若是反复识别不到物体,多为相机/检测异常,请先确认视觉是否正常。"
        )
    return f"未完成:{summary}" if summary else "未完成:agent 结束但未确认任务已完成。"


def outcome_from_result(result: dict[str, Any]) -> Outcome:
    """Derive the status badge and narration from a ``run_finished`` result.

    On failure (``ok`` false) the caller runs ``diagnose`` to render the diagnostics page, so
    ``is_failure=True``.
    """
    if not result.get("ok"):
        return Outcome("失败", "运行失败,已在下方「错误诊断」给出原因与处理建议。", True)
    payload = result.get("result")
    # The fast path's result shape is {"ok", "steps_done", "steps":[...], ...} with no result_type;
    # the outer RunEngine ok only means "no exception raised", so real success/failure is this inner
    # ok — otherwise a failed step would fall through to the final "success" branch.
    if isinstance(payload, dict) and "steps_done" in payload:
        if payload.get("ok"):
            return Outcome("成功", "完成", False)
        failed = next((s for s in payload.get("steps", []) if isinstance(s, dict) and not s.get("ok")), None)
        if failed is not None:
            reason = str(failed.get("reason", "")).strip()
            # Keep only a short 「未完成」 under the camera; the failed op + reason (often English) go
            # into detail, routed to 「错误诊断」. Use the same friendly name as the timeline, not fast's
            # internal step index (0-based, includes off-timeline track_detect, so it wouldn't match).
            label = humanize.friendly_label(str(failed.get("op", "")), {})
            detail = f"{label} 这一步失败:{reason}" if reason else f"{label} 这一步失败。"
            return Outcome("未完成", "未完成", False, detail=detail)
        return Outcome("未完成", "未完成", False, detail="动作序列未跑完。")
    rtype = payload.get("result_type") if isinstance(payload, dict) else ""
    summary = payload.get("output") if isinstance(payload, dict) else payload
    if rtype == "stopped":
        return Outcome("已停止", f"已停止:{summary}", False)
    if rtype == "error":
        return Outcome("未完成", incomplete_message(str(summary)), False)
    return Outcome("成功", f"完成:{summary}" if summary else "完成", False)
