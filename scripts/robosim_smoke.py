"""Run a deterministic JiuwenSymbiosis action sequence against real Isaac physics.

This validates the tool/rail/runner/adapter path without an LLM or a mock robot.
The service is launched separately using run_robosim_service.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence
from jiuwensymbiosis.api.actions import ACTIONS
from jiuwensymbiosis.introspect import build_session
from jiuwensymbiosis.rails.safety import SafetyRail
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool
from jiuwensymbiosis.utils.logging import configure_logging, get_logger

logger = get_logger("jiuwensymbiosis.robosim_smoke")


def main() -> int:
    """Persist measured before/after state and return nonzero on any acceptance failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robosim/moz1.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/robosim_smoke/headless.json"))
    parser.add_argument("--cartesian", action="store_true", help="Also exercise cuMotion TCP planning")
    args = parser.parse_args()
    configure_logging()
    report = {"ok": False, "robot_backend": "real_isaac_physx", "planner": "fixed_action_sequence", "actions": []}
    started = time.monotonic()
    try:
        with build_session(args.config) as session:
            rail = SafetyRail(session)
            tool = RobotControlTool(session.api, env=session.env)

            def execute(op: str, params: dict) -> dict:
                """Run the normal tool dispatcher after the synchronous rail check."""
                rail.validate_motion(op, params)
                output = asyncio.run(tool.invoke({"action": op, "params": params}))
                entry = {
                    "op": op,
                    "params": params,
                    "success": output.success,
                    "data": output.data,
                    "error": output.error,
                }
                report["actions"].append(entry)
                logger.info("%s", json.dumps(entry, ensure_ascii=False))
                data = output.data or {}
                return {
                    "ok": output.success,
                    "result": data.get("result"),
                    "reason": output.error,
                    "error_code": data.get("error_code"),
                    "outcome_unknown": data.get("outcome_unknown") is True,
                    "skip_recovery": data.get("skip_recovery") is True,
                }

            initial = session.env.low_level.observe()
            report["initial"] = initial
            joints = initial["joint_positions"]
            joint = list(joints)[-1]
            start = joints[joint]
            target = start + 0.1
            sequence = [
                {"op": "open_gripper"},
                {"op": "move_joint", "params": {"targets": {joint: target}}},
                {"op": "get_joint_positions"},
            ]
            allowed = {name: ACTIONS[name] for name in tool.available_actions}
            steps = parse_sequence(sequence, allowed_ops=allowed)
            result = run_sequence(session, steps, executor=execute)
            report["sequence"] = result
            if not result.get("ok"):
                raise RuntimeError(f"Action sequence failed: {result}")
            moved = session.api.get_joint_positions()
            report["joint_target"] = {joint: target}
            report["joint_measured"] = moved
            if abs(moved[joint] - target) > session.env.cfg.joint_tolerance_rad:
                raise RuntimeError("Joint target was not reached")
            if abs(moved[joint] - start) < 0.05:
                raise RuntimeError("No physical joint displacement measured")
            if not execute("move_joint", {"targets": {joint: start}})["ok"]:
                raise RuntimeError("Joint restoration failed")

            if args.cartesian:
                original_pose = session.api.get_pose()
                pose = {**original_pose, "z": original_pose["z"] + 30.0}
                if not execute("goto_pose", {"pose": pose})["ok"]:
                    raise RuntimeError("TCP movement failed")
                report["tcp_target"] = pose
                report["tcp_measured"] = session.api.get_pose()
                if not execute("goto_pose", {"pose": original_pose})["ok"]:
                    raise RuntimeError("TCP restoration failed")

            # There is no object in this smoke scene. Actuation is real, but the
            # payload.held effect must not be admitted into ExecutionMemory.
            empty_close = execute("close_gripper", {})
            close_data = report["actions"][-1]["data"] or {}
            report["empty_close_rejected"] = (
                not empty_close["ok"] and close_data.get("error_code") == "grasp_not_confirmed"
            )
            if not execute("open_gripper", {})["ok"]:
                raise RuntimeError("Gripper reopen failed")
            if not report["empty_close_rejected"]:
                raise RuntimeError("Expected grasp_not_confirmed for an empty closure")
            if not execute("home", {})["ok"]:
                raise RuntimeError("Planned home failed")
            report["final"] = session.env.low_level.observe()
            report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["duration_s"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "%s",
        json.dumps({"ok": report["ok"], "output": str(args.output), "error": report.get("error")}, ensure_ascii=False),
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
