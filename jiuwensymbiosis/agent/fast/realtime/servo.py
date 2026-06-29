# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""High-frequency servo control loop (the "实时执行" half).

``ServoController`` runs a ``control_hz`` loop that, each tick:

  1. reads the current tip pose (``read_pose``),
  2. asks a ``target_provider`` for the latest goal pose (fed by a background
     perception tracker; ``None`` means "no target right now → hold"),
  3. slew-limits a step toward the goal (so a far or jumpy detection can't
     cause a violent jump), and
  4. issues a **non-blocking** pose command (``servo_to``).

It returns when the tip has settled within tolerance of the goal for
``settle_ticks`` consecutive ticks, on timeout, or when the target is lost
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
    timeout_s: float = 20.0  # hard timeout for one servo move
    lost_target_grace_s: float = 3.0  # abort if no target this long


@dataclass
class ServoResult:
    """Outcome of one ``ServoController.run()``."""

    ok: bool
    reason: str  # "reached" | "timeout" | "target_lost" | "stopped"
    ticks: int
    elapsed_s: float
    final_pose: Pose | None
    target_pose: Pose | None

    def as_dict(self) -> dict:
        """Flatten to a plain dict for tool/agent return values."""
        return {
            "ok": self.ok,
            "reason": self.reason,
            "ticks": self.ticks,
            "elapsed_s": round(self.elapsed_s, 3),
            "final_pose": self.final_pose,
            "target_pose": self.target_pose,
        }


def _pose_error(cur: Pose, tgt: Pose) -> tuple[float, float]:
    """Return ``(position_err_mm, angular_err_deg)`` between two poses."""
    pos_sq = 0.0
    for k in _LINEAR_KEYS:
        if k in tgt and k in cur:
            pos_sq += (float(tgt[k]) - float(cur[k])) ** 2
    ang_err = 0.0
    for k in _ANGULAR_KEYS:
        if k in tgt and k in cur:
            ang_err = max(ang_err, abs(_ang_diff_deg(float(tgt[k]), float(cur[k]))))
    return math.sqrt(pos_sq), ang_err


def _slew(cur: Pose, tgt: Pose, max_lin: float, max_ang: float) -> Pose:
    """Return a pose one slew-limited step from ``cur`` toward ``tgt``."""
    nxt: Pose = dict(cur)
    for k, v in tgt.items():
        v = float(v)
        if k not in cur:
            nxt[k] = v
            continue
        c = float(cur[k])
        if k in _LINEAR_KEYS:
            delta = v - c
            if abs(delta) > max_lin:
                delta = math.copysign(max_lin, delta)
            nxt[k] = c + delta
        elif k in _ANGULAR_KEYS:
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
        servo_to: ``(Pose) -> None`` non-blocking pose command.
        target_provider: ``() -> Optional[Pose]`` latest goal pose; ``None`` =
            no target this tick (hold position).
        config: ``ServoConfig``.
        on_tick: optional ``(tick_info: dict) -> None`` progress callback.
        should_continue: optional ``() -> bool``; return False to stop early.
    """

    def __init__(
        self,
        read_pose: Callable[[], Pose],
        servo_to: Callable[[Pose], None],
        target_provider: Callable[[], Pose | None],
        *,
        config: ServoConfig | None = None,
        on_tick: Callable[[dict], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> None:
        self._read_pose = read_pose
        self._servo_to = servo_to
        self._target_provider = target_provider
        self._cfg = config or ServoConfig()
        self._on_tick = on_tick
        self._should_continue = should_continue

    def run(self) -> ServoResult:
        """Run the control loop until reached / timeout / target lost / stopped."""
        cfg = self._cfg
        period = 1.0 / cfg.control_hz if cfg.control_hz > 0 else 0.0
        t0 = time.monotonic()
        last_target_t = t0
        in_tol = 0
        ticks = 0
        last_pose: Pose | None = None
        last_target: Pose | None = None

        while True:
            tick_start = time.monotonic()
            ticks += 1

            if self._should_continue is not None and not self._should_continue():
                return ServoResult(False, "stopped", ticks, tick_start - t0, last_pose, last_target)

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

            if cur is not None and tgt is not None:
                pos_err, ang_err = _pose_error(cur, tgt)
                if pos_err <= cfg.pos_tol_mm and ang_err <= cfg.ang_tol_deg:
                    in_tol += 1
                    if in_tol >= cfg.settle_ticks:
                        return ServoResult(True, "reached", ticks, tick_start - t0, cur, tgt)
                else:
                    in_tol = 0
                    step = _slew(cur, tgt, cfg.max_lin_step_mm, cfg.max_ang_step_deg)
                    try:
                        self._servo_to(step)
                    except Exception as exc:  # noqa: BLE001 - abort servo on command failure
                        logger.warning("[servo] servo_to failed: %s", exc)
                        return ServoResult(False, "stopped", ticks, tick_start - t0, cur, tgt)

            if self._on_tick is not None:
                self._on_tick({"tick": ticks, "pose": cur, "target": tgt, "in_tol": in_tol})

            now = time.monotonic()
            if tgt is None and (now - last_target_t) > cfg.lost_target_grace_s:
                return ServoResult(False, "target_lost", ticks, now - t0, cur, last_target)
            if (now - t0) > cfg.timeout_s:
                return ServoResult(False, "timeout", ticks, now - t0, cur, last_target)

            # Maintain the loop rate.
            sleep_s = period - (now - tick_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
