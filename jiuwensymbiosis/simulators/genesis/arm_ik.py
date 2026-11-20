"""Scene-owned cuRobo IK only: cached initialization, no collision or path search."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def solve_targets(planner, state, group, tool, targets, directory, *, warmup=False):
    import torch
    from curobo.types import GoalToolPose, JointState

    names = planner.profile["groups"][group]["joints"]
    started = time.monotonic()
    record = {
        "backend": "curobo_ik",
        "pid": os.getpid(),
        "collision_checked": False,
        "operation": "warmup" if warmup else "ik",
        "solves": [],
    }
    answers = []
    try:
        planner.prepare_ik(group, state)
        backend = planner.backend
        seed = state.copy()
        for target in targets:
            begin = time.monotonic()
            values = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat(scalar_first=True)]
            goal = GoalToolPose(
                tool_frames=backend.tool_frames,
                position=torch.tensor(values[:3], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1, 3),
                quaternion=torch.tensor(values[3:], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1, 4),
            )
            solve_record = {"success": False}
            record["solves"].append(solve_record)
            try:
                result = backend.solve_pose(
                    goal,
                    current_state=JointState.from_position(
                        torch.tensor([[seed[n] for n in names]], device="cuda", dtype=torch.float32), joint_names=names
                    ),
                )
                success = bool(result.success.all().item())
                solve_record["success"] = success
            finally:
                solve_record["wall_s"] = time.monotonic() - begin
            if not success:
                return []
            q = result.solution.detach().cpu().numpy().reshape(-1, len(names))[0]
            if not np.isfinite(q).all():
                raise ValueError("cuRobo IK returned nonfinite joints")
            candidate = dict(zip(names, q.tolist(), strict=True))
            for name, value in candidate.items():
                limit = planner.joints[name].find("limit")
                if not float(limit.get("lower", "-inf")) <= value <= float(limit.get("upper", "inf")):
                    raise ValueError("cuRobo IK exceeded joint limits")
            # Independent model FK validates the returned IK, without any collision query.
            planning = planner_module(planner)
            fk = planning.forward_kinematics(planner.root, state | candidate)
            actual = np.linalg.inv(fk[planner.profile["base_link"]]) @ fk[tool]
            error = planning.pose_error(target, actual)
            record["solves"][-1]["fk_error"] = error
            if error["position_m"] > 0.002 or error["rotation_rad"] > 0.02:
                return []
            answers.append(candidate)
            seed.update(candidate)
        return answers
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.update(planner.prepare_timing)
        record.update(total_s=time.monotonic() - started, targets=answers)
        (directory / f"ik-{uuid.uuid4().hex}.json").write_text(json.dumps(record, indent=2, allow_nan=False))


def planner_module(planner):
    from importlib import import_module

    return import_module(type(planner).__module__)


def stop_worker(arm):
    process = getattr(arm, "ik_process", None)
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        process.stdin.close()
        arm.ik_process = None


def worker_environment():
    """Use the installed matching NVRTC builtins in each isolated cuRobo worker."""
    environment = os.environ.copy()
    runtime = find_spec("nvidia.cu13")
    if runtime is not None:
        libraries = [str(Path(root) / "lib") for root in runtime.submodule_search_locations or ()]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([*libraries, environment.get("LD_LIBRARY_PATH", "")])
    return environment


def solve(arm, group, tool, targets, *, warmup=False):
    """Serial scene-owned subprocess isolates cuRobo's CUDA compiler from Genesis."""
    started = time.monotonic()
    state, _ = arm.read()
    token = uuid.uuid4().hex
    request = arm.directory / f"ik-request-{token}.json"
    result_path = request.with_suffix(".result.json")
    request.write_text(
        json.dumps(
            {
                "manifest": arm.planner.model,
                "profile": arm.profile,
                "state": state,
                "group": group,
                "tool": tool,
                "poses": targets.tolist(),
                "warmup": warmup,
            },
            allow_nan=False,
        )
    )
    process = getattr(arm, "ik_process", None)
    if process is None or process.poll() is not None:
        stop_worker(arm)
        with (arm.directory / "ik-worker.log").open("a") as log:
            arm.ik_process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())],
                stdin=subprocess.PIPE,
                env=worker_environment(),
                stdout=log,
                stderr=log,
                text=True,
            )
        process = arm.ik_process
    try:
        process.stdin.write(str(request.resolve()) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + 60
        while not result_path.exists():
            if process.poll() is not None:
                raise RuntimeError("cuRobo IK worker exited")
            if time.monotonic() > deadline:
                raise TimeoutError("cuRobo IK worker timeout")
            time.sleep(0.01)
        result = json.loads(result_path.read_text())
        if "error" in result:
            raise RuntimeError(result["error"])
        return result if warmup else result["targets"]
    except Exception:
        stop_worker(arm)
        raise
    finally:
        request.with_suffix(".timing.json").write_text(
            json.dumps(
                {
                    "operation": "warmup" if warmup else "ik",
                    "group": group,
                    "request_wall_s": time.monotonic() - started,
                    "worker_pid": process.pid,
                },
                indent=2,
            )
        )


def serve():
    from arm_planning import CuroboPlanner, identity

    planners = {}
    try:
        for line in sys.stdin:
            path = Path(line.strip())
            data = json.loads(path.read_text())
            started = time.monotonic()
            try:
                key = identity({"manifest": data["manifest"], "profile": data["profile"], "group": data["group"]})
                if key not in planners:
                    planner = CuroboPlanner(data["manifest"], data["profile"], path.parent)
                    planners[key] = planner
                planner = planners[key]
                planner_module(planner).sibling("robot_model").verify_model_files(planner.model)
                targets = solve_targets(
                    planner,
                    data["state"],
                    data["group"],
                    data["tool"],
                    np.asarray(data["poses"]),
                    path.parent,
                    warmup=data.get("warmup", False),
                )
                if data.get("warmup"):
                    if not targets:
                        raise RuntimeError("current-pose IK warmup failed")
                    planner.backend.reset_seed()
                result = {"targets": targets}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            result["timing"] = {"worker_total_s": time.monotonic() - started, "worker_pid": os.getpid()}
            pending = path.with_suffix(".pending.json")
            pending.write_text(json.dumps(result, allow_nan=False))
            pending.replace(path.with_suffix(".result.json"))
    finally:
        for planner in planners.values():
            planner.close()


if __name__ == "__main__":
    serve()
