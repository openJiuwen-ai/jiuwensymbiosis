"""One leased Genesis pick through the same planned-motion and effector commands as manual operation."""

from __future__ import annotations

import time

import numpy as np

from .grasp_pose import rgbd_points


def pick_target(env, backend, target, detector=None):
    if not target.strip() or env.holding_payload is not False:
        raise ValueError("pick requires an identified target and clear gripper")
    driver = env.driver
    driver.require_motion()
    state = driver.preview.status()
    binding = (state["scene"]["instance_id"], state["config_revision"])
    effector = backend.effector

    def validate():
        driver.require_motion()
        state = driver.preview.status()
        if state.get("state") != "ready" or (state["scene"]["instance_id"], state["config_revision"]) != binding:
            raise RuntimeError("grasp observation scene changed")

    def reached(result):
        return (
            result.get("ok")
            and result.get("settled")
            and result.get("goal_reached")
            and result.get("stop_confirmation") == "measured"
        )

    result = env.set_gripper(True, effector)
    if not reached(result):
        return {**result, "ok": False}
    frame = None
    truth = None

    def segmented(frame):
        detections = detector(
            frame.rgb, target, frame_id=frame.frame_id, captured_monotonic_s=frame.captured_monotonic_s
        )
        if len(detections) != 1:
            raise ValueError("pick requires exactly one segmented target")
        env.validate_frame(frame)
        return detections[0]["mask"]

    def visual():
        nonlocal frame
        frame = env.grab_calibrated_frame()
        mask = segmented(frame)
        if np.count_nonzero(mask & np.isfinite(frame.depth_m) & (frame.depth_m > 0)) < 100 and np.any(mask):
            v, u = np.nonzero(mask)
            region = (
                max(0, int(u.min()) - 4),
                max(0, int(v.min()) - 4),
                min(mask.shape[1], int(u.max()) + 5),
                min(mask.shape[0], int(v.max()) + 5),
            )
            frame = env.grab_calibrated_frame(depth_region=region)
            mask = segmented(frame)
        return rgbd_points(frame.depth_m, mask, frame.intrinsics, frame.metadata["T_planning_base_camera_m"])

    def oracle():
        nonlocal truth
        truth = driver.query_truth("grasp_cloud", {"entity_id": target})
        return truth["points_m"]

    def observe():
        return backend.point_cloud(rgbd=visual, ground_truth=oracle)

    points = observe()
    candidates = backend.propose(points)
    validate()

    def refresh():
        # IK screening or cold inference may outlive an observation. Check the
        # geometry again through the same sensor before moving the arm.
        from scipy.spatial import cKDTree

        current = observe()
        if (
            max(
                np.quantile(cKDTree(first).query(second)[0], 0.90)
                for first, second in ((points, current), (current, points))
            )
            > 0.006
        ):
            raise ValueError("target moved or its segmented surface changed during grasp planning")
        if frame is not None:
            env.validate_frame(frame)
        if truth is not None and not 0 <= time.monotonic() - truth["captured_at"] < 5:
            raise ValueError("grasp cloud expired")

    ordered = [i for i in np.argsort(-candidates.scores) if candidates.grippers[i, 2, 2] < -0.85]
    if not ordered:
        raise ValueError("no tabletop approach candidate")
    for idx in ordered[:32]:
        axis = candidates.grippers[idx, :3, 2]
        goal = candidates.tools[idx]
        pre = goal.copy()
        pre[:3, 3] -= 0.08 * axis
        lift = goal.copy()
        lift[2, 3] += 0.08
        validate()
        answer = driver.preview.query_scene(
            "grasp_ik",
            {"group": backend.group, "tool": backend.tool, "poses": [pre.tolist(), goal.tolist(), lift.tolist()]},
            cancelled=driver.cancelled,
        )
        if answer["scene_instance_id"] != binding[0]:
            raise RuntimeError("grasp screening scene changed")
        if not answer["targets"]:
            continue
        refresh()
        pre_q, goal_q, lift_q = answer["targets"]
        for q in (pre_q, goal_q):
            validate()
            result = driver.move_arm(group=backend.group, targets=q)
            if not reached(result):
                break
        if reached(result):
            break
        if (
            result.get("failure_stage") not in {"curobo_planning", "path_collision_check"}
            or result.get("plan_success")
            or not result.get("settled")
            or result.get("stop_confirmation") != "measured"
        ):
            return {**result, "ok": False}
    else:
        return {**result, "ok": False, "goal_reached": False, "reason": "no_feasible_grasp"}
    validate()
    trial = {"effector": effector, "center_m": np.median(points, axis=0).tolist()}
    result = driver.move_arm(group=backend.group, pick_lift={**trial, "joint_goal": lift_q})
    if (
        result.get("failure_stage") in {"curobo_planning", "path_collision_check"}
        and not result.get("plan_success")
        and result.get("settled")
        and result.get("stop_confirmation") == "measured"
    ):
        validate()
        result = driver.move_arm(group=backend.group, pick_lift=trial)
    ok = bool(reached(result) and result.get("grasp_confirmed") and result.get("holding_payload"))
    if ok and truth is not None:
        ok = driver.preview.status()["effectors"][effector]["object_id"] == truth["body_idx"]
    return {
        **result,
        "ok": ok,
        "goal_reached": ok,
        "reason": None if ok else result.get("reason") or "target_hold_unconfirmed",
    }
