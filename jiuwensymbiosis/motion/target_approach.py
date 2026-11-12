"""Shared, bounded RGB-D target approach for real and simulated mobile bases."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

from jiuwensymbiosis.motion.base_goal import plan_target_step
from jiuwensymbiosis.perception.scene3d import locate_target


@dataclass(frozen=True)
class TargetApproachTuning:
    max_iterations: int = 16
    max_observation_retries: int = 2
    timeout_s: float = 180.0
    distance_tolerance_m: float = 0.05
    bearing_tolerance_rad: float = math.radians(3)
    max_forward_m: float = 0.25
    max_turn_rad: float = 0.35
    min_target_distance_m: float = 0.8
    max_target_distance_m: float = 3.0
    body_radius_m: float = 0.55
    min_clearance_m: float = 0.15

    def __post_init__(self):
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 64:
            raise ValueError("Invalid approach iteration budget")
        if type(self.max_observation_retries) is not int or not 0 <= self.max_observation_retries <= 8:
            raise ValueError("Invalid observation retry budget")
        for key, value in vars(self).items():
            if key == "max_observation_retries":
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Approach tuning must be finite and positive")
        if self.min_target_distance_m > self.max_target_distance_m:
            raise ValueError("Invalid target-distance interval")


class TargetStepGuard(Protocol):
    """Env extension required before target approach can issue a nonzero step."""

    def check_target_step(self, forward: float, turn: float, frame_id: str, tuning: TargetApproachTuning) -> str | None:
        """Certify the proposed swept body volume from fresh sensor evidence.

        ``forward`` is metres and ``turn`` radians (turn first, then advance).
        Resolve ``frame_id`` to the current validated RGB-D frame, check its
        epoch/freshness and the full body envelope, including a pure turn.
        Return None only when safe; otherwise return a nonempty refusal reason.
        Missing or insufficient evidence must refuse, never infer free space.
        """


def approach_target(
    api: Any,
    object_name: str,
    target_distance_m: float = 1.2,
    reference: str | None = None,
    relation: str = "on",
) -> dict:
    """One explicit action owns this loop; there is no background retry or routing.

    The binding supplies calibrated RGB-D and measured base pose (CameraFrame),
    ``env.check_target_step`` (TargetStepGuard), and ``env.navigate_relative``
    or ``api.nav_relative`` with MeasuredBaseDriver stopping semantics. Missing
    clearance evidence or unconfirmed stopping prevents further motion.
    See docs/en/reference/robot-api.md for optional frame-validation hooks.
    """
    tuning = getattr(api, "target_approach_tuning", TargetApproachTuning())
    if (
        isinstance(target_distance_m, bool)
        or not isinstance(target_distance_m, (int, float))
        or not (
            math.isfinite(target_distance_m)
            and tuning.min_target_distance_m <= target_distance_m <= tuning.max_target_distance_m
        )
    ):
        raise ValueError("target_distance_m is outside this body's supported approach interval")
    nav = getattr(api, "nav_relative", None) or api.env.navigate_relative
    deadline = time.monotonic() + tuning.timeout_s
    result: dict[str, Any] = {
        "ok": False,
        "goal_reached": False,
        "settled": False,
        "stop_confirmation": "unknown",
        "reason": "max_iterations",
        "iterations": 0,
        "target_distance_m": target_distance_m,
        "steps": [],
        "observations": 0,
        "observation_retries": [],
    }
    previous = getattr(api, "target_observation", None)
    if previous is not None and (
        previous.query != (object_name, reference, relation) or time.monotonic() - previous.captured_at > 5.0
    ):
        previous = None
    # An initial zero-distance action confirms stopping even when the target is already in band.
    motion = nav(0.0, 0.0, 0.0)
    retries_left = tuning.max_observation_retries
    last_step_observation = None

    def plan(measurement):
        return plan_target_step(
            measurement["point_base_mm"],
            target_distance_m,
            distance_tolerance_m=tuning.distance_tolerance_m,
            bearing_tolerance_rad=tuning.bearing_tolerance_rad,
            max_forward_m=tuning.max_forward_m,
            max_turn_rad=tuning.max_turn_rad,
        )

    for iteration in range(tuning.max_iterations + 1):
        result["iterations"] = iteration
        result["settled"] = motion.get("settled") is True
        result["stop_confirmation"] = motion.get("stop_confirmation", "unknown")
        if not motion.get("ok") or not result["settled"] or result["stop_confirmation"] != "measured":
            result["reason"] = motion.get("reason", "motion_failed") if not motion.get("ok") else "stop_unconfirmed"
            return result
        pair: list[dict] = []
        while len(pair) < 2:
            if time.monotonic() >= deadline:
                result["reason"] = "deadline_exceeded"
                return result
            measured = locate_target(api, object_name, reference, relation, previous=previous)
            result["observations"] += 1
            result["measurement"] = measured
            if not measured.get("ok"):
                reason = measured.get("reason", "target_not_found")
                # Only retry observations of an already acquired target. Never
                # select one of ambiguous candidates or move to acquire a view.
                if reason in {"no_detection", "ambiguous_target"} and previous is not None and retries_left:
                    retries_left -= 1
                    result["observation_retries"].append({"reason": reason, "observation": result["observations"]})
                    pair.clear()
                    continue
                if pair:
                    result["confirmation_failure"] = measured
                result["reason"] = reason
                return result
            previous = api.target_observation
            if previous is None:
                result["reason"] = "association_odometry_unavailable"
                return result
            if time.monotonic() >= deadline:
                result["reason"] = "deadline_exceeded"
                return result
            if pair:
                first = pair[0]
                if (
                    not measured.get("frame_id")
                    or measured["frame_id"] == first.get("frame_id")
                    or measured.get("captured_monotonic_s", 0) <= first.get("captured_monotonic_s", 0)
                ):
                    result["reason"] = "stale_confirmation_frame"
                    return result
                first_state, measured_state = plan(first)[2], plan(measured)[2]
                # Two far observations can disagree on the visible median while
                # both justify a forward step. Treat them as an interval and
                # control from its near end; do not average them into arrival.
                consistent = (
                    (
                        first_state == measured_state == "advance"
                        or abs(first["distance_m"] - measured["distance_m"]) <= tuning.distance_tolerance_m
                    )
                    and abs(math.remainder(first["bearing_rad"] - measured["bearing_rad"], 2 * math.pi))
                    <= tuning.bearing_tolerance_rad
                    and first_state == measured_state
                )
                if not consistent:
                    reason = "target_measurement_unstable"
                    if "reached" in {first_state, measured_state}:
                        reason = "arrival_not_stable"
                    if not retries_left:
                        result["reason"] = reason
                        return result
                    retries_left -= 1
                    result["observation_retries"].append({"reason": reason, "observation": result["observations"]})
                    pair = [measured]
                    continue
            pair.append(measured)
        first, measured = pair
        turn, forward, state = plan(measured)
        result.update(
            distance_error_m=measured["distance_m"] - target_distance_m,
            bearing_error_rad=measured["bearing_rad"],
            iterations=iteration,
        )
        if state == "reached":
            result.update(
                ok=True,
                goal_reached=True,
                reason="reached",
                confirmation_frame_ids=[first["frame_id"], measured["frame_id"]],
            )
            return result
        if state == "too_close" or iteration == tuning.max_iterations:
            result["reason"] = state if state == "too_close" else "max_iterations"
            return result
        visible_variation = abs(first["distance_m"] - measured["distance_m"])
        if last_step_observation is not None:
            visible_variation = max(visible_variation, last_step_observation.visible_center_shift_m(previous))
        if forward > 0:
            remaining = min(first["distance_m"], measured["distance_m"]) - target_distance_m
            # Retain bounded progress for stable observations. Reduce the gain
            # when the measured visible surface changes, or when the very first
            # step is already near the goal and no cross-view evidence exists.
            uncertain = visible_variation > tuning.distance_tolerance_m or (
                last_step_observation is None and remaining <= tuning.max_forward_m
            )
            if uncertain:
                forward = min(forward, remaining / 2)
            else:
                forward = min(forward, remaining - tuning.distance_tolerance_m / 2)
            available = (
                min(first["near_surface_x_m"], measured["near_surface_x_m"])
                - tuning.body_radius_m
                - tuning.min_clearance_m
            )
            if available < min(forward, tuning.distance_tolerance_m):
                result["reason"] = "insufficient_target_clearance"
                return result
            forward = min(forward, available)
        guard = getattr(api.env, "check_target_step", None)
        if guard is None:
            result["reason"] = "clearance_unavailable"
            return result
        refusal = guard(forward, turn, measured["frame_id"], tuning)
        if refusal:
            result["reason"] = refusal
            return result
        api.invalidate_sensing_cache()
        last_step_observation = previous
        motion = nav(forward, 0.0, turn)
        result["steps"].append(
            {
                "frame_id": measured["frame_id"],
                "support_frame_ids": [first["frame_id"], measured["frame_id"]],
                "forward_m": forward,
                "turn_rad": turn,
                "visible_variation_m": visible_variation,
                "motion": motion,
            }
        )
    return result
