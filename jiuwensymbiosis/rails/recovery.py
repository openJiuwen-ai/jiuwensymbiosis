# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RecoveryRail — payload-aware auto-home/release on tool exceptions.

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
- Adapter-specific exceptions may set ``skip_recovery = True`` when the command
  was rejected before leaving the arm in an unsafe state; those are surfaced to
  the caller without an automatic home move.
- A motion failure while the adapter reports ``holding_payload=True`` keeps the
  end effector closed and attempts a payload-aware recovery home. It never
  drops the payload merely because endpoint settle was outside tolerance.
- If ``home`` itself fails, recovery does not retry the same failed home motion.

Two optional api hooks let a body join this without a bespoke rail, both looked
up by name so no adapter import is needed:

- ``release_effector()`` — release for bodies whose end effector is neither a
  suction cup nor a parallel gripper (dual-arm paddles, magnets, …). Tried after
  ``deactivate_suction`` / ``open_gripper``, before the env-level fallback.
- ``recovery_home()`` — the payload-aware retreat, preferred over ``home()``.
  A body whose plain ``home()`` would drop a held payload implements it.
"""

from __future__ import annotations

import logging
from typing import Any

from jiuwensymbiosis.agent.abstractions import AgentRail
from jiuwensymbiosis.agent.trace import TraceEventSink

logger = logging.getLogger(__name__)


def read_holding_payload(session: Any) -> bool | None:
    """Read payload state as true / false / unavailable."""
    api = getattr(session, "api", None)
    env = getattr(session, "env", None)
    if env is None and api is not None:
        env = getattr(api, "env", None)
    if env is None:
        return None
    missing = object()
    try:
        value = getattr(env, "holding_payload", missing)
    except Exception as exc:  # noqa: BLE001 - recovery remains best-effort
        logger.warning("Recovery: could not read holding_payload: %s", exc)
        return None
    if value is missing or value is None:
        return None
    if isinstance(value, bool):
        return value
    logger.warning(
        "Recovery: ignoring non-boolean holding_payload=%r from %s",
        value,
        type(env).__name__,
    )
    return None


def recover_session(
    session: Any,
    *,
    release: bool,
    home: bool,
    log_prefix: str,
) -> tuple[bool, bool]:
    """Run the shared best-effort release/home actions."""
    api = session.api
    released_ok = False
    if release:
        for stop_method in ("deactivate_suction", "open_gripper", "release_effector"):
            fn = getattr(api, stop_method, None)
            if not callable(fn):
                continue
            try:
                result = fn()
                if isinstance(result, dict) and result.get("ok") is False:
                    raise RuntimeError(str(result.get("reason") or "returned ok=False"))
                released_ok = True
                logger.info("%s: %s succeeded", log_prefix, stop_method)
                break
            except Exception as exc:  # noqa: BLE001 - every recovery action is independent
                logger.warning("%s: %s failed: %s", log_prefix, stop_method, exc)

        if not released_ok:
            env = getattr(session, "env", None)
            set_end_effector = getattr(env, "set_end_effector", None)
            if callable(set_end_effector):
                try:
                    result = set_end_effector(False)
                    if isinstance(result, dict) and result.get("ok") is False:
                        raise RuntimeError(str(result.get("reason") or "returned ok=False"))
                    released_ok = True
                    logger.info("%s: env.set_end_effector(False) succeeded", log_prefix)
                except Exception as exc:  # noqa: BLE001 - home must still be attempted
                    logger.warning("%s: env release fallback failed: %s", log_prefix, exc)

    home_ok = False
    if home:
        home_fn = getattr(api, "recovery_home", None)
        if not callable(home_fn):
            home_fn = getattr(api, "home", None)
        if callable(home_fn):
            try:
                result = home_fn()
                if isinstance(result, dict) and result.get("ok") is False:
                    raise RuntimeError(str(result.get("reason") or "returned ok=False"))
                home_ok = True
                logger.info("%s: home() succeeded", log_prefix)
            except Exception as exc:  # noqa: BLE001 - recovery must remain best-effort
                logger.warning("%s: home() failed: %s", log_prefix, exc)
    return released_ok, home_ok


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
        # Some adapters raise a typed "not reached" error after rejecting a
        # compensation command.  The arm is already in its last validated safe
        # pose; blindly homing can be less safe (and destroys the useful current
        # pose).  Adapters opt out without making this generic rail import them.
        exception = getattr(ctx, "exception", None)
        skip_exception = self._find_skip_recovery_exception(exception)
        if skip_exception is not None:
            logger.warning(
                "RecoveryRail: skipping recovery for wrapped %s: %s",
                type(skip_exception).__name__,
                skip_exception,
            )
            return
        inputs = getattr(ctx, "inputs", None)
        tool_name = getattr(inputs, "tool_name", "") or ""
        tool_args = getattr(inputs, "tool_args", None)
        if not self._is_watched(tool_name, tool_args=tool_args):
            return
        effective_name = self._effective_tool_name(tool_name, tool_args=tool_args)
        effective_tags = self._effective_tags(effective_name)
        holding_payload = read_holding_payload(self.session)
        preserve_payload = holding_payload is True and "motion" in effective_tags
        if preserve_payload:
            logger.warning(
                "RecoveryRail: preserving held payload after motion failure; "
                "attempting recovery home without opening the end effector"
            )
        skip_home = effective_name in {"home", "recovery_home"}
        if skip_home:
            logger.warning("RecoveryRail: not retrying home after %s failed", effective_name)
        released_ok, home_ok = recover_session(
            self.session,
            # A false-negative contact inference can report False while a thin
            # or compliant object is still between the fingers. Release unless
            # a payload is positively confirmed and intentionally preserved.
            release=self.also_release_grip and not preserve_payload,
            home=not skip_home,
            log_prefix="RecoveryRail",
        )
        self._notify_recover(
            tool_name,
            home_ok=home_ok,
            released_ok=released_ok,
            holding_payload=holding_payload,
            payload_preserved=preserve_payload,
        )

    def _notify_recover(
        self,
        tool_name: str,
        *,
        home_ok: bool,
        released_ok: bool,
        holding_payload: bool | None,
        payload_preserved: bool,
    ) -> None:
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
                    "holding_payload": holding_payload,
                    "payload_preserved": payload_preserved,
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
        effective_name = self._effective_tool_name(tool_name, tool_args=tool_args)
        api = getattr(self.session, "api", None)
        if api is None:
            return False
        method = getattr(api, effective_name, None)
        meta = getattr(method, "__robot_tool__", None)
        if meta is None:
            return False
        return bool(set(meta.tags) & self.watch_tags)

    def _effective_tags(self, effective_name: str) -> set[str]:
        """Return the effective API method's robot-tool tags."""
        api = getattr(self.session, "api", None)
        method = getattr(api, effective_name, None) if api is not None else None
        meta = getattr(method, "__robot_tool__", None)
        return set(getattr(meta, "tags", ()))

    @staticmethod
    def _find_skip_recovery_exception(exception: Any) -> BaseException | None:
        """Find an adapter opt-out marker through framework exception wrappers.

        OpenJiuwen wraps tool failures in ``AbilityExecutionError``. Its
        ``cause``/``__cause__`` retains the original adapter exception, so only
        checking the outer object loses ``skip_recovery`` and triggers an
        unnecessary release/home after a pre-dispatch rejection.
        """
        if not isinstance(exception, BaseException):
            return None
        pending: list[BaseException] = [exception]
        visited: set[int] = set()
        wrapper_attrs = (
            "__cause__",
            "cause",
            "exception",
            "original_exception",
            "original_error",
        )
        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            if bool(getattr(current, "skip_recovery", False)):
                return current
            for attr in wrapper_attrs:
                try:
                    nested = getattr(current, attr, None)
                except Exception:  # noqa: BLE001 - wrapper inspection must be fail-safe
                    continue
                if isinstance(nested, BaseException) and id(nested) not in visited:
                    pending.append(nested)
        return None

    @staticmethod
    def _effective_tool_name(tool_name: str, *, tool_args: Any = None) -> str:
        """Return the adapter action name, unwrapping ``robot_control``."""
        if tool_name != "robot_control":
            return tool_name
        args = tool_args if isinstance(tool_args, dict) else {}
        action = args.get("action", "")
        return str(action) if action else tool_name
