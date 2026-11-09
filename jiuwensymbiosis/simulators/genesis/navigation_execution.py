"""Execute one planned route through the existing leased, measured wheel actions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from jiuwensymbiosis.contracts import NavigationResult
from jiuwensymbiosis.simulators.genesis.base_motion import wrap

if TYPE_CHECKING:
    from jiuwensymbiosis.simulators.genesis.agent import GenesisDriver


def _stopped(robot: dict) -> bool:
    return (
        math.sqrt(sum(v * v for v in robot["linear_velocity_m_s"])) < 0.008
        and math.sqrt(sum(v * v for v in robot["angular_velocity_rad_s"])) < 0.02
    )


def navigate(driver: GenesisDriver, params: dict) -> NavigationResult:
    """Follow one route, checking fresh physical feedback before each next command."""
    driver.require_motion()
    planned = driver.query_truth("plan_navigation", params)
    plan = planned["plan"]
    binding = planned["scene_instance_id"], planned["reset_generation"]

    def observe():
        current = driver.query_truth("robot_state", {})
        if (current["scene_instance_id"], current["reset_generation"]) != binding:
            raise RuntimeError("Navigation scene changed")
        return current

    def failure(result: dict, state: dict | None = None) -> NavigationResult:
        if result.get("settled") is not True:
            driver.fail_stop_confirmation()
        outcome: NavigationResult = {
            "ok": False,
            "goal_reached": False,
            "settled": result.get("settled") is True,
            "stop_confirmation": "measured" if result.get("settled") else "unknown",
            "reason": result.get("reason", "navigation_interrupted"),
            "plan": plan,
            "source": "simulation_ground_truth",
            "scene_instance_id": binding[0],
            "reset_generation": binding[1],
            "recoverable": result.get("recoverable", False),
        }
        if state is not None:
            outcome["final_state"] = state
        driver.results.append(dict(outcome))
        return outcome

    current = observe()
    if math.dist(current["robot"]["position_m"][:2], planned["robot"]["position_m"][:2]) > 0.02:
        raise ValueError("Navigation start changed after planning; query and plan again")
    for waypoint in plan["waypoints"]:
        robot = current["robot"]
        dx, dy = waypoint[0] - robot["position_m"][0], waypoint[1] - robot["position_m"][1]
        c, s = math.cos(robot["body_yaw_rad"]), math.sin(robot["body_yaw_rad"])
        result = driver.navigate_relative(c * dx + s * dy, -s * dx + c * dy)
        if not (result.get("ok") and result.get("settled")):
            return failure(result)
        current = observe()
        if not _stopped(current["robot"]):
            driver.fail_stop_confirmation()
            return failure({"settled": False, "reason": "navigation_readback_unsettled"}, current)
        if math.dist(current["robot"]["position_m"][:2], waypoint) > 0.04:
            return failure({"settled": True, "reason": "navigation_waypoint_error"}, current)
    result = driver.navigate_relative(0, 0, wrap(plan["goal"][2] - current["robot"]["body_yaw_rad"]))
    if not (result.get("ok") and result.get("settled")):
        return failure(result)
    current = observe()
    robot = current["robot"]
    distance = math.dist(robot["position_m"][:2], plan["goal"][:2])
    angle = wrap(plan["goal"][2] - robot["body_yaw_rad"])
    reached = distance <= plan["position_tolerance_m"] and abs(angle) <= plan["yaw_tolerance_rad"]
    settled = _stopped(robot)
    outcome: NavigationResult = {
        "ok": reached and settled,
        "goal_reached": reached,
        "settled": settled,
        "stop_confirmation": "measured" if settled else "unknown",
        "reason": "reached" if reached and settled else "navigation_final_state_error",
        "position_error_m": distance,
        "yaw_error_rad": angle,
        "final_state": current,
        "plan": plan,
        "source": "simulation_ground_truth",
        "scene_instance_id": binding[0],
        "reset_generation": binding[1],
    }
    driver.results.append(dict(outcome))
    if not settled:
        driver.fail_stop_confirmation()
    return outcome
