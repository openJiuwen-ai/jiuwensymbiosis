# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bind a ``RobotSession`` to the generic servo IO the real-time loop needs.

The ``realtime`` core (``ServoController`` / ``StreamingFrameSource`` /
``BackgroundTracker``) is deliberately hardware-agnostic: it speaks in plain
callables and pose dicts. ``ServoBinding`` is the one place that knows how to
pull those callables out of a concrete session (Piper, mock, or any adapter
whose api exposes the conventional tool surface):

  * ``read_pose``  → ``api.get_pose()`` (TIP frame, dict).
  * ``servo_to``   → ``api.servo_to_tip(pose)`` when present (Piper does the
    tip→flange conversion there); otherwise ``env.servo_to_flange(pose)``
    (correct when tip == flange, e.g. the mock arm).
  * ``grip``       → ``api.close_gripper`` / ``open_gripper`` when present;
    otherwise ``env.set_end_effector``.
  * ``frames``     → a ``StreamingFrameSource`` over ``env.get_observation``.

Requires the env to declare ``motion.servo``; raises a clear error otherwise so
"this body can't servo" is a config error, not a mystery hang.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from jiuwensymbiosis.agent.fast.realtime.streaming import StreamingFrameSource
from jiuwensymbiosis.rails.safety import SafetyRail

logger = logging.getLogger(__name__)

Pose = dict[str, float]


class ServoBinding:
    """Adapter between a ``RobotSession`` and the generic servo IO callables."""

    def __init__(self, session: Any, *, frame_max_hz: float = 30.0) -> None:
        self.session = session
        self.api = session.api
        self.env = session.env
        if "motion.servo" not in set(getattr(self.env, "capabilities", frozenset())):
            raise RuntimeError(
                f"{getattr(self.env, 'name', 'env')}: real-time servo needs the "
                "'motion.servo' capability (non-blocking streaming motion). "
                "This body does not declare it."
            )
        self._servo_to_tip = getattr(self.api, "servo_to_tip", None)
        # SO-101 opts into a command-state planning chain. Other adapters keep
        # the historical live-pose slew origin unless they explicitly declare
        # the same contract on their API.
        self.slew_from_last_command = bool(getattr(self.api, "servo_slew_from_last_command", False))
        # Keep command orientation separate from completion orientation. Most
        # robots use every angular key in the target (None). Underactuated
        # adapters may opt into a subset, including (), while still receiving
        # the full best-effort pose in servo_to().
        reached_angular_keys = getattr(self.api, "servo_reached_angular_keys", None)
        self.reached_angular_keys = None if reached_angular_keys is None else tuple(reached_angular_keys)
        self._log_servo_ticks = bool(getattr(self.api, "servo_log_ticks", False))
        self._safety = SafetyRail(session)
        self.frames = StreamingFrameSource(self._grab_frame, max_hz=frame_max_hz, name="servo")

    def make_tick_logger(self, phase: str) -> Callable[[dict], None] | None:
        """Return an INFO tick callback for adapters that explicitly opt in."""
        if not self._log_servo_ticks:
            return None

        def log_tick(info: dict) -> None:
            pose = info.get("pose")
            target = info.get("target")
            logger.info(
                "[servo-tick] phase=%s tick=%d live=%s target=%s pos_err_mm=%s ang_err_deg=%s in_tol=%d",
                phase,
                int(info["tick"]),
                _rounded_pose(pose),
                _rounded_pose(target),
                _rounded_number(info.get("position_error_mm")),
                _rounded_number(info.get("angular_error_deg")),
                int(info.get("in_tol", 0)),
            )

        return log_tick

    # ------------------------------------------------------------------ motion
    def read_pose(self) -> Pose:
        """Current TIP-frame pose as a dict (the controller's pose reader)."""
        return cast(Pose, self.api.get_pose())

    def servo_to(self, pose: Pose) -> bool | None:
        """Command toward a TIP-frame pose and preserve dispatch acknowledgement."""
        self._safety.validate_pose(pose)
        if self._servo_to_tip is not None:
            return cast(bool | None, self._servo_to_tip(pose))
        # tip == flange (e.g. mock): the env's non-blocking verb is correct.
        return cast(bool | None, self.env.servo_to_flange(pose))

    # ----------------------------------------------------------------- gripper
    def grip(self, closed: bool) -> None:
        """Close (True) / open (False) the end effector via the api or env."""
        if closed:
            fn = getattr(self.api, "close_gripper", None)
            if fn is not None:
                fn()
                return
        else:
            fn = getattr(self.api, "open_gripper", None)
            if fn is not None:
                fn()
                return
        self.env.set_end_effector(closed)

    # ------------------------------------------------------------------ frames
    def _grab_frame(self) -> tuple | None:
        """Pull one ``(rgb, depth)`` from the env observation (frame source)."""
        try:
            obs = self.env.get_observation()
        except Exception:  # noqa: BLE001 - frame grab best-effort; None on failure
            return None
        if obs is None:
            return None
        return (obs.rgb, obs.depth)

    def start_frames(self) -> StreamingFrameSource:
        """Start (and return) the background frame source."""
        return self.frames.start()

    def stop_frames(self) -> None:
        """Stop the background frame source."""
        self.frames.stop()


def _rounded_number(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def _rounded_pose(pose: Any) -> dict[str, float] | None:
    if not isinstance(pose, dict):
        return None
    return {str(key): round(float(value), 3) for key, value in pose.items()}
