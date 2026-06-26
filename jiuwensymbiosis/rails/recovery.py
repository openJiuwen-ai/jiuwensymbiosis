# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RecoveryRail — auto-home + drop on tool exceptions.

If a motion tool raises (e.g. driver lost connection, IK failed), leaving
the arm mid-motion at an unknown pose is unsafe for the next call. The
rail attempts a `home()` and `deactivate_suction()` (best-effort, both
optional) to bring the robot to a known state before the LLM retries.

Notes:
- Only triggers for tools whose tags include "motion" or "grasp", to avoid
  reacting to vision/diagnostic exceptions.
- Recovery actions themselves are NOT railed (they wrap the env directly,
  not the api), so a recovery failure is logged but does not crash the
  rail.
"""

from __future__ import annotations

import logging
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import TraceEventSink

logger = logging.getLogger(__name__)


class RecoveryRail(AgentRail):
    """Attempt to return the robot to a safe state when a motion/grasp tool errors."""

    def __init__(
        self,
        session: Any,
        *,
        also_release_grip: bool = True,
        watch_tags: set[str] | None = None,
        trace_sink: TraceEventSink | None = None,
    ) -> None:
        """Initialize the recovery rail."""
        super().__init__()
        self.session = session
        self.also_release_grip = also_release_grip
        self.watch_tags = frozenset(watch_tags) if watch_tags else frozenset({"motion", "grasp"})
        self.trace_sink = trace_sink

    async def on_tool_exception(self, ctx: Any) -> None:
        """Recover to a safe state after a motion/grasp tool exception."""
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        tool_args = getattr(inputs, "tool_args", None)
        if not self._is_watched(tool_name, tool_args=tool_args):
            return
        api = self.session.api
        released_ok = False
        home_ok = False
        # Best-effort: each step is independent.
        if self.also_release_grip:
            for stop_method in ("deactivate_suction", "open_gripper"):
                fn = getattr(api, stop_method, None)
                if callable(fn):
                    try:
                        fn()
                        released_ok = True
                        logger.info("RecoveryRail: %s succeeded", stop_method)
                        break
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("RecoveryRail: %s failed: %s", stop_method, exc)
        home = getattr(api, "home", None)
        if callable(home):
            try:
                home()
                home_ok = True
                logger.info("RecoveryRail: home() succeeded")
            except Exception as exc:  # noqa: BLE001
                logger.warning("RecoveryRail: home() failed: %s", exc)
        self._notify_recover(tool_name, home_ok=home_ok, released_ok=released_ok)

    def _notify_recover(self, tool_name: str, *, home_ok: bool, released_ok: bool) -> None:
        sink = self.trace_sink
        if sink is None:
            return
        try:
            sink.record_rail_event(
                rail_name="RecoveryRail",
                kind="recover",
                detail={
                    "tool_name": tool_name,
                    "home_ok": home_ok,
                    "released_ok": released_ok,
                },
                success=home_ok,
            )
        except (AttributeError, TypeError, ValueError):
            # tracing must never break recovery
            pass

    def _is_watched(self, tool_name: str, *, tool_args: Any = None) -> bool:
        """Check whether the given tool triggers recovery.

        When ``RobotControlTool`` is used, ``tool_name`` is ``"robot_control"``
        and the actual action lives in ``tool_args["action"]``.  We unwrap this
        so that motion / grasp actions dispatched through the single entry point
        still trigger recovery.
        """
        effective_name = tool_name
        if tool_name == "robot_control":
            args = tool_args if isinstance(tool_args, dict) else {}
            action = args.get("action", "")
            if action:
                effective_name = str(action)
        api = getattr(self.session, "api", None)
        if api is None:
            return False
        method = getattr(api, effective_name, None)
        meta = getattr(method, "__robot_tool__", None)
        if meta is None:
            return False
        return bool(set(meta.tags) & self.watch_tags)
