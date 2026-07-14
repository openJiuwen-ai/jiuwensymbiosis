# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UIBridgeRail —— 把 agent 运行时的每一步翻译成界面事件。

作为 ``RobotAgentConfig.extra_rails`` 注入(优先级设低,使其在 SafetyRail 判定
之后观察),在 ``before/after_tool_call`` / ``on_tool_exception`` 钩子里把工具
调用变成"步骤开始/结束/失败"事件、抓取相机帧、生成自然语言叙述,并识别安全护栏
拦截。事件通过一个 ``emitter`` 对象向外发(运行时是 NiceGUI 侧的事件队列引擎
``RunEngine``,测试时是记录型假对象),因此本模块**不依赖界面框架**,可独立单测。

回调运行在后台 worker 线程的事件循环内;emitter 只负责"发出去",绝不直接碰控件。
"""

from __future__ import annotations

import time
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.gui import humanize
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["UIBridgeRail", "summarize_output"]

# 技术抽屉「返回」栏展示的工具返回值上限。运行页的详情框是可滚动 QTextEdit,
# 长内容(如 skill_tool 返回的整篇 SKILL.md ~8KB)靠滚动条查看即可,故这里放到
# 足以容下一整篇技能文档;仅对病态超长输出(如误塞进来的大数组 repr)才截断兜底。
_MAX_SUMMARY = 16000


def summarize_output(result: Any) -> str:
    """把工具返回值转成文本(供技术抽屉「返回」栏展示,由可滚动详情框呈现)。

    正常输出(含整篇 skill 文档)原样返回;仅当超过 ``_MAX_SUMMARY`` 才截断,防止
    病态超长输出撑爆界面。
    """
    text = repr(result)
    return text if len(text) <= _MAX_SUMMARY else text[:_MAX_SUMMARY] + "…"


def _extract(ctx: Any) -> tuple[str, dict]:
    """从回调 ctx 里取真实工具名与参数(解包 robot_control)。"""
    inputs = getattr(ctx, "inputs", None)
    name = getattr(inputs, "tool_name", "") or ""
    args = getattr(inputs, "tool_args", None)
    return humanize.unwrap_robot_control(name, args)


def _result_ok_error(result: Any) -> tuple[bool, str]:
    """从工具返回值判定这一步成功与否 + 错误串。

    工具未抛异常≠成功:失败常以「携带成功标记的返回值」汇报(如越界、检测未命中,
    fast 路径尤甚)。兼容两种形态——openjiuwen 的 ``ToolOutput``(``.success`` / ``.error``
    属性)与直接返回的 dict(``ok`` / ``success`` 键,原因在 ``error`` / ``reason``)。
    取不到成功标记时按成功处理(默认 True),不给正常步骤误判失败。
    """
    if isinstance(result, dict):
        ok = result.get("ok", result.get("success", True))
        err = result.get("error") or result.get("reason") or ""
    else:
        ok = getattr(result, "success", getattr(result, "ok", True))
        err = getattr(result, "error", "") or ""
    return bool(ok), str(err)


class UIBridgeRail(AgentRail):
    """把工具调用事件桥接到界面 emitter。

    ``emitter`` 需提供这些方法(运行时由 ``RunEngine`` 实现,测试可用假对象):
    ``step_started(dict)`` / ``step_finished(dict)`` / ``frame(rgb)`` /
    ``narration(str)`` / ``safety_event(dict)``。
    """

    # 低优先级:在 SafetyRail(默认 50)判定之后再观察结果。
    priority = 1

    def __init__(self, emitter: Any, session: Any, should_stop: Any = None) -> None:
        """绑定界面 emitter 与会话(用于抓取相机帧)。

        ``should_stop`` 可选,是一个无参可调用对象;在每步开始前若返回真,则请求
        agent 立即结束(用户点了"停止")。
        """
        self.emitter = emitter
        self.session = session
        self.should_stop = should_stop
        self._counter = 0
        # 本轮 LLM 的文本(思考/说明);由 after_model_call 更新,挂到随后各步详情里。
        self._turn_text = ""

    # ---------------------------------------------------------------- 钩子
    async def after_model_call(self, ctx: Any) -> None:
        """捕获本轮 LLM 回复的文本内容,随后挂到该轮触发的各步详情里。

        让用户在「原始细节」里看到"AI 这一步在想什么 / 为什么这么做",而不仅是工具名
        和参数。``ctx.inputs.response`` 是本轮 LLM 回复(``ModelCallInputs``),``.content``
        即助手文本;非字符串(如多模态分块)时留空,不影响运行。
        """
        resp = getattr(getattr(ctx, "inputs", None), "response", None)
        content = getattr(resp, "content", None)
        self._turn_text = content.strip() if isinstance(content, str) else ""

    async def before_tool_call(self, ctx: Any) -> None:
        """一步开始:计数、记时,发出"步骤开始"与当前动作叙述。"""
        if self.should_stop is not None and self.should_stop():
            ctx.request_force_finish({"output": "用户已停止运行", "result_type": "stopped"})
            return
        name, params = _extract(ctx)
        self._counter += 1
        ctx.extra["_ui_idx"] = self._counter
        ctx.extra["_ui_t0"] = time.monotonic()
        ctx.extra["_ui_open"] = True
        self._emit_started(self._counter, name, params)

    async def after_tool_call(self, ctx: Any) -> None:
        """一步结束(未抛异常):按返回值判成败发出"步骤完成",运动/抓取后刷新相机画面。"""
        name, params = _extract(ctx)
        idx, dur = self._close(ctx)
        result = getattr(getattr(ctx, "inputs", None), "tool_result", None)
        ok, err = _result_ok_error(result)
        info = {
            "index": idx,
            "tool": name,
            "label": humanize.friendly_label(name, params),
            "ok": ok,
            "duration_s": dur,
            "output": summarize_output(result),
            "params": params,
            "assistant_text": self._turn_text,
        }
        if not ok and err:
            info["error"] = err  # 失败步详情按 error 展示(与 on_tool_exception 一致)
        self._safe(self.emitter.step_finished, info)
        if name in humanize.FRAME_AFTER_TOOLS:
            self._grab_frame()

    async def on_tool_exception(self, ctx: Any) -> None:
        """一步失败:发出"步骤失败";若是安全护栏拦截,另发安全事件。"""
        name, params = _extract(ctx)
        if not ctx.extra.get("_ui_open"):
            # before_tool_call 未运行(如 SafetyRail 在更高优先级直接拦截)——补一个开始事件。
            self._counter += 1
            ctx.extra["_ui_idx"] = self._counter
            ctx.extra["_ui_t0"] = time.monotonic()
            ctx.extra["_ui_open"] = True
            self._emit_started(self._counter, name, params)
        idx, dur = self._close(ctx)
        err = str(getattr(ctx, "exception", "") or "")
        self._safe(
            self.emitter.step_finished,
            {
                "index": idx,
                "tool": name,
                "label": humanize.friendly_label(name, params),
                "ok": False,
                "duration_s": dur,
                "error": err,
                "params": params,
                "assistant_text": self._turn_text,
            },
        )
        if "SafetyRail" in err:
            self._safe(
                self.emitter.safety_event,
                {"rail": "SafetyRail", "kind": "reject", "detail": err},
            )

    # ---------------------------------------------------------------- 内部
    def _emit_started(self, idx: int, name: str, params: dict) -> None:
        self._safe(
            self.emitter.step_started,
            {
                "index": idx,
                "tool": name,
                "label": humanize.friendly_label(name, params),
                "params": params,
                "assistant_text": self._turn_text,
            },
        )
        self._safe(self.emitter.narration, humanize.narration(name, params))

    def _close(self, ctx: Any) -> tuple[int, float]:
        """结束当前步:返回 (index, 用时秒),并清除 open 标记。"""
        idx = int(ctx.extra.get("_ui_idx", self._counter))
        t0 = ctx.extra.get("_ui_t0")
        dur = round(time.monotonic() - t0, 3) if isinstance(t0, int | float) else 0.0
        ctx.extra["_ui_open"] = False
        return idx, dur

    def _grab_frame(self) -> None:
        """抓取最新相机帧发给界面(失败只记日志,绝不影响运行)。"""
        try:
            rgb = self.session.env.get_observation().rgb
        except Exception as exc:  # 取帧失败不能中断任务
            logger.debug("grab frame failed: %s", exc)
            return
        if rgb is not None:
            self._safe(self.emitter.frame, rgb)

    @staticmethod
    def _safe(fn: Any, *args: Any) -> None:
        """调用 emitter 方法;失败只记日志,保证界面桥接绝不打断机器人任务。"""
        try:
            fn(*args)
        except Exception as exc:  # UI 桥接绝不影响机器人执行
            logger.debug("emitter %s failed: %s", getattr(fn, "__name__", fn), exc)
