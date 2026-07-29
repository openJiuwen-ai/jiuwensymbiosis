# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""High-frequency servo control loop (the "实时执行" half).

``ServoController`` runs a ``control_hz`` loop that, each tick:

  1. reads the current tip pose (``read_pose``),
  2. asks a ``target_provider`` for the latest goal pose (fed by a background
     perception tracker; ``None`` means "no target right now → hold"),
  3. slew-limits a step toward the goal (so a far or jumpy detection can't
     cause a violent jump), optionally chaining from the previous commanded
     step instead of re-anchoring the plan to the live pose, and
  4. issues a **non-blocking** pose command (``servo_to``).

It returns when the tip has settled within tolerance of the goal for
``settle_ticks`` consecutive ticks, when the live pose stops making progress,
when an optional absolute safety deadline expires, or when the target is lost
for longer than ``lost_target_grace_s``.

Pose representation is a plain ``dict`` so the same controller drives a 4-DoF
SCARA (``x,y,z,r``) and a 6-DoF arm (``x,y,z,rx,ry,rz``): linear keys
(``x/y/z``) are slewed by ``max_lin_step_mm`` and angular keys
(``r/rx/ry/rz``) by ``max_ang_step_deg``; keys absent from the current pose are
set directly.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

Pose = dict[str, float]

_LINEAR_KEYS = ("x", "y", "z")
_ANGULAR_KEYS = ("r", "rx", "ry", "rz")


def _ang_diff_deg(a: float, b: float) -> float:
    """Shortest signed angular difference ``a-b`` in degrees, wrapped to [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


@dataclass
class ServoConfig:
    """Servo-loop tuning. Conservative defaults are safe for first bring-up."""

    control_hz: float = 30.0  # control-loop rate
    max_lin_step_mm: float = 6.0  # max linear move per tick (slew limit)
    max_ang_step_deg: float = 5.0  # max angular move per tick (slew limit)
    pos_tol_mm: float = 4.0  # "reached" position tolerance
    ang_tol_deg: float = 3.0  # "reached" angular tolerance
    settle_ticks: int = 3  # consecutive in-tolerance ticks to finish
    # Continuous no-progress timeout. This is deliberately not a total move
    # deadline: a long approach may run past it while the live pose is still
    # measurably approaching the latest target.
    timeout_s: float = 20.0
    # Final safety ceiling for a moving target that can keep a servo phase alive
    # indefinitely. None disables the ceiling.
    absolute_timeout_s: float | None = 60.0
    # Minimum accumulated live-pose improvement that refreshes timeout_s.
    progress_pos_epsilon_mm: float = 0.5
    progress_ang_epsilon_deg: float = 0.5
    lost_target_grace_s: float = 3.0  # abort if no target this long

    def __post_init__(self) -> None:
        # Reject non-finite / non-positive tuning; clamp control_hz to the
        # hardware-validated band (<=0 would busy-loop with period=0).
        for name, val in (
            ("control_hz", self.control_hz),
            ("max_lin_step_mm", self.max_lin_step_mm),
            ("max_ang_step_deg", self.max_ang_step_deg),
            ("pos_tol_mm", self.pos_tol_mm),
            ("ang_tol_deg", self.ang_tol_deg),
            ("timeout_s", self.timeout_s),
            ("progress_pos_epsilon_mm", self.progress_pos_epsilon_mm),
            ("progress_ang_epsilon_deg", self.progress_ang_epsilon_deg),
            ("lost_target_grace_s", self.lost_target_grace_s),
        ):
            if isinstance(val, bool) or not (
                isinstance(val, (int, float)) and math.isfinite(float(val)) and float(val) > 0
            ):
                raise ValueError(f"ServoConfig.{name} must be finite and > 0, got {val!r}.")
        if self.absolute_timeout_s is not None and (
            isinstance(self.absolute_timeout_s, bool)
            or not (
                isinstance(self.absolute_timeout_s, (int, float))
                and math.isfinite(float(self.absolute_timeout_s))
                and float(self.absolute_timeout_s) > 0
            )
        ):
            raise ValueError(
                f"ServoConfig.absolute_timeout_s must be None or finite and > 0, got {self.absolute_timeout_s!r}."
            )
        if isinstance(self.settle_ticks, bool) or not isinstance(self.settle_ticks, int):
            raise ValueError(f"ServoConfig.settle_ticks must be int, got {self.settle_ticks!r}.")
        if self.settle_ticks < 1:
            raise ValueError(f"ServoConfig.settle_ticks must be >= 1, got {self.settle_ticks}.")
        if not (1.0 <= float(self.control_hz) <= 200.0):
            raise ValueError(f"ServoConfig.control_hz must be in [1, 200] (hardware-validated), got {self.control_hz}.")


@dataclass
class ServoResult:
    """Outcome of one ``ServoController.run()``."""

    ok: bool
    reason: str  # "reached" | "timeout" | "target_lost" | "stopped"
    ticks: int
    elapsed_s: float
    final_pose: Pose | None
    target_pose: Pose | None
    error: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict:
        """Flatten to a plain dict for tool/agent return values."""
        result = {
            "ok": self.ok,
            "reason": self.reason,
            "ticks": self.ticks,
            "elapsed_s": round(self.elapsed_s, 3),
            "final_pose": self.final_pose,
            "target_pose": self.target_pose,
        }
        if self.error:
            result["error"] = self.error
        if self.error_code:
            result["error_code"] = self.error_code
        return result


def _pose_error(
    cur: Pose,
    tgt: Pose,
    *,
    angular_keys: tuple[str, ...] | None = None,
) -> tuple[float, float]:
    """Return ``(position_err_mm, angular_err_deg)`` between two poses.

    ``angular_keys=None`` preserves the generic behavior: every angular key
    present in both poses participates in the reached check. An adapter may
    pass an explicit subset (including ``()``) when an underactuated arm should
    still receive a best-effort orientation command without making that
    orientation a hard completion gate.
    """
    pos_sq = 0.0
    for k in _LINEAR_KEYS:
        if k in tgt and k in cur:
            pos_sq += (float(tgt[k]) - float(cur[k])) ** 2
    ang_err = 0.0
    checked_angular_keys = _ANGULAR_KEYS if angular_keys is None else angular_keys
    for k in checked_angular_keys:
        if k in tgt and k in cur:
            ang_err = max(ang_err, abs(_ang_diff_deg(float(tgt[k]), float(cur[k]))))
    return math.sqrt(pos_sq), ang_err


def _slew(cur: Pose, tgt: Pose, max_lin: float, max_ang: float) -> Pose:
    """Return a pose one slew-limited step from ``cur`` toward ``tgt``."""
    nxt: Pose = dict(cur)

    # Limit the translational *vector*, rather than clipping x/y/z
    # independently.  Independent clipping lets a diagonal move travel up to
    # ``sqrt(3) * max_lin`` in one tick, which is both harder on the arm and
    # makes the effective speed depend on the direction of the target.  Keep
    # the angular channels independent because their units/axes are different.
    linear_keys = [k for k in _LINEAR_KEYS if k in tgt and k in cur]
    if linear_keys:
        linear_delta = {k: float(tgt[k]) - float(cur[k]) for k in linear_keys}
        linear_norm = math.sqrt(sum(delta * delta for delta in linear_delta.values()))
        scale = min(1.0, float(max_lin) / linear_norm) if linear_norm > 0.0 else 1.0
        for k in linear_keys:
            nxt[k] = float(cur[k]) + linear_delta[k] * scale

    angular_keys = [key for key in ("rx", "ry", "rz") if key in tgt and key in cur]
    if len(angular_keys) == 3:
        current_rotation = Rotation.from_euler("xyz", [float(cur[key]) for key in angular_keys], degrees=True)
        target_rotation = Rotation.from_euler("xyz", [float(tgt[key]) for key in angular_keys], degrees=True)
        relative_rotvec = (current_rotation.inv() * target_rotation).as_rotvec()
        angle_deg = math.degrees(math.sqrt(sum(float(value) ** 2 for value in relative_rotvec)))
        rotation_scale = min(1.0, float(max_ang) / angle_deg) if angle_deg > 0.0 else 1.0
        next_euler = (current_rotation * Rotation.from_rotvec(relative_rotvec * rotation_scale)).as_euler(
            "xyz", degrees=True
        )
        for key, value in zip(angular_keys, next_euler, strict=True):
            nxt[key] = float(value)

    for k, v in tgt.items():
        v = float(v)
        if k not in cur:
            nxt[k] = v
            continue
        c = float(cur[k])
        if k in _LINEAR_KEYS:
            # Already handled as one XYZ vector above.  A key absent from the
            # current pose cannot participate in the norm and retains the
            # historical direct-assignment behavior.
            if k not in linear_keys:
                nxt[k] = v
        elif k in _ANGULAR_KEYS:
            if len(angular_keys) == 3:
                continue
            delta = _ang_diff_deg(v, c)
            if abs(delta) > max_ang:
                delta = math.copysign(max_ang, delta)
            nxt[k] = c + delta
        else:
            nxt[k] = v
    return nxt


class ServoController:
    """Drive the tip toward a (possibly moving) target at ``control_hz``.

    Args:
        read_pose: ``() -> Pose`` current tip pose (dict).
        servo_to: non-blocking pose command. An explicit ``False`` return means
            the low-level controller did not advance this tick (for example, a
            rate-gate skip or tracking catch-up hold); ``True`` and legacy
            ``None`` mean the command was accepted.
        target_provider: ``() -> Optional[Pose]`` latest goal pose; ``None`` =
            hold position for this tick, then abort with ``target_lost`` if no
            target recovers within ``lost_target_grace_s``.
        config: ``ServoConfig``.
        on_tick: optional ``(tick_info: dict) -> None`` progress callback.
        should_continue: optional ``() -> bool``; return False to stop early.
        target_is_live: optional detector-progress watchdog. Returning False
            aborts immediately as ``target_lost`` without adding another grace
            period; useful when a cached target can remain non-None while its
            detector has stalled.
        slew_from_last_command: if true, the first slew starts at the live pose
            and later slews start at the previous successfully issued command.
            The live pose is still used for reached/settle feedback. This is an
            opt-in for adapters whose low-level IK also maintains a planned
            seed chain; the default preserves the generic live-pose behavior.
        reached_angular_keys: angular pose keys that participate in the hard
            reached check. ``None`` checks every angular key shared by current
            and target poses; ``()`` makes completion position-only while the
            full target pose is still sent to ``servo_to``.
    """

    def __init__(
        self,
        read_pose: Callable[[], Pose],
        servo_to: Callable[[Pose], bool | None],
        target_provider: Callable[[], Pose | None],
        *,
        config: ServoConfig | None = None,
        on_tick: Callable[[dict], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
        target_is_live: Callable[[], bool] | None = None,
        slew_from_last_command: bool = False,
        reached_angular_keys: tuple[str, ...] | None = None,
    ) -> None:
        self._read_pose = read_pose
        self._servo_to = servo_to
        self._target_provider = target_provider
        self._cfg = config or ServoConfig()
        self._on_tick = on_tick
        self._should_continue = should_continue
        self._target_is_live = target_is_live
        self._slew_from_last_command = bool(slew_from_last_command)
        if reached_angular_keys is not None:
            invalid = [key for key in reached_angular_keys if key not in _ANGULAR_KEYS]
            if invalid:
                raise ValueError(f"reached_angular_keys contains unsupported keys: {invalid}")
            reached_angular_keys = tuple(dict.fromkeys(reached_angular_keys))
        self._reached_angular_keys = reached_angular_keys

    def run(self) -> ServoResult:
        """Run until reached, stalled, absolutely timed out, target lost, or stopped."""
        cfg = self._cfg
        period = 1.0 / cfg.control_hz  # __post_init__ guarantees control_hz in [1, 200]
        t0 = time.monotonic()
        last_target_t = t0
        last_progress_t = t0
        in_tol = 0
        ticks = 0
        last_pose: Pose | None = None
        last_target: Pose | None = None
        last_command: Pose | None = None
        progress_pose: Pose | None = None

        while True:
            tick_start = time.monotonic()
            ticks += 1

            if self._should_continue is not None and not self._should_continue():
                return ServoResult(False, "stopped", ticks, tick_start - t0, last_pose, last_target)

            if self._target_is_live is not None and not self._target_is_live():
                return ServoResult(False, "target_lost", ticks, tick_start - t0, last_pose, last_target)

            cur: Pose | None
            try:
                cur = self._read_pose()
            except Exception as exc:  # noqa: BLE001 - pose read may glitch; reuse last
                logger.debug("[servo] read_pose failed: %s", exc)
                cur = last_pose
            last_pose = cur

            tgt = self._target_provider()
            if tgt is not None:
                last_target = tgt
                last_target_t = tick_start

            pos_err: float | None = None
            ang_err: float | None = None
            reached = False
            if cur is not None and tgt is not None:
                pos_err, ang_err = _pose_error(cur, tgt, angular_keys=self._reached_angular_keys)
                if progress_pose is None:
                    # Target acquisition/warm-up is governed by the detector
                    # watchdog, not charged against the motion-progress budget.
                    progress_pose = dict(cur)
                    last_progress_t = tick_start
                else:
                    # Compare both poses against this tick's latest target. This
                    # proves that the robot moved toward the target rather than
                    # mistaking a target update for robot progress.
                    ref_pos_err, ref_ang_err = _pose_error(
                        progress_pose,
                        tgt,
                        angular_keys=self._reached_angular_keys,
                    )
                    position_progress = (
                        pos_err > cfg.pos_tol_mm and ref_pos_err - pos_err >= cfg.progress_pos_epsilon_mm
                    )
                    angular_progress = (
                        ang_err > cfg.ang_tol_deg and ref_ang_err - ang_err >= cfg.progress_ang_epsilon_deg
                    )
                    if position_progress or angular_progress:
                        progress_pose = dict(cur)
                        last_progress_t = tick_start
                if pos_err <= cfg.pos_tol_mm and ang_err <= cfg.ang_tol_deg:
                    in_tol += 1
                    if in_tol >= cfg.settle_ticks:
                        reached = True
                else:
                    in_tol = 0
                    slew_origin = last_command if self._slew_from_last_command and last_command is not None else cur
                    step = _slew(slew_origin, tgt, cfg.max_lin_step_mm, cfg.max_ang_step_deg)
                    try:
                        dispatched = self._servo_to(step)
                    except Exception as exc:  # noqa: BLE001 - abort servo on command failure
                        logger.warning("[servo] servo_to failed: %s", exc)
                        return ServoResult(
                            False,
                            "stopped",
                            ticks,
                            tick_start - t0,
                            cur,
                            tgt,
                            str(exc),
                            getattr(exc, "code", None),
                        )
                    # ``False`` explicitly means the low-level plan did not
                    # advance. Legacy sinks return None, which remains accepted
                    # for compatibility.
                    if dispatched is not False:
                        last_command = dict(step)

            if self._on_tick is not None:
                self._on_tick(
                    {
                        "tick": ticks,
                        "pose": cur,
                        "target": tgt,
                        "position_error_mm": pos_err,
                        "angular_error_deg": ang_err,
                        "in_tol": in_tol,
                    }
                )
            if reached:
                return ServoResult(True, "reached", ticks, tick_start - t0, cur, tgt)

            now = time.monotonic()
            if tgt is None and (now - last_target_t) > cfg.lost_target_grace_s:
                return ServoResult(False, "target_lost", ticks, now - t0, cur, last_target)
            if cfg.absolute_timeout_s is not None and (now - t0) > cfg.absolute_timeout_s:
                return ServoResult(
                    False,
                    "timeout",
                    ticks,
                    now - t0,
                    cur,
                    last_target,
                    f"absolute servo safety limit exceeded ({float(cfg.absolute_timeout_s):g}s)",
                    "servo_absolute_timeout",
                )
            no_progress_s = now - last_progress_t
            if no_progress_s > cfg.timeout_s:
                return ServoResult(
                    False,
                    "timeout",
                    ticks,
                    now - t0,
                    cur,
                    last_target,
                    f"live pose made no effective progress for {no_progress_s:.2f}s (limit {float(cfg.timeout_s):g}s)",
                    "servo_no_progress_timeout",
                )

            # Maintain the loop rate.
            sleep_s = period - (now - tick_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
