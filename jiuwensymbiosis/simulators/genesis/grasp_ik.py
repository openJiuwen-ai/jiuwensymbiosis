"""Screen a candidate's approach, grasp and lift poses against one stationary planning snapshot."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

if __package__:
    from .arm_collision import CollisionGate, snapshot
    from .arm_planning import joint_transform
    from .grasp_pose import rigid
else:
    from arm_collision import CollisionGate, snapshot
    from arm_planning import joint_transform
    from grasp_pose import rigid


def screen(arm, group, tool, poses):
    targets = rigid(poses)
    if targets.shape != (3, 4, 4) or arm.profile["groups"][group]["tools"] != [tool]:
        raise ValueError("expected approach, grasp and lift poses in the configured planning base frame")
    if arm.active or not arm.stopped():
        raise ValueError("grasp screening requires a stationary arm")
    names = arm.profile["groups"][group]["joints"]
    joints = arm.planner.joints
    state, _ = arm.read()
    root = arm.planner.root
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain = []
    link = tool
    while link != arm.profile["base_link"]:
        j = parents[link]
        name = j.get("name")
        idx = names.index(name) if name in names else None
        if idx is not None and j.get("type") != "revolute":
            raise ValueError("grasp IK requires bounded revolute arm joints")
        axis = j.find("axis")
        vector = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1.0, 0, 0])
        chain.append(
            (idx, joint_transform(j, 0 if idx is not None else state.get(name, 0)), vector / np.linalg.norm(vector))
        )
        link = j.find("parent").get("link")
    chain.reverse()
    limits = np.array([[float(joints[n].find("limit").get(k)) for n in names] for k in ("lower", "upper")])
    lo, hi = limits[0] + 0.002, limits[1] - 0.002
    seed = np.array([state[n] for n in names])
    gate = CollisionGate(root, arm.profile, snapshot(arm.entity, arm.profile, arm.effectors.attachments()))
    answers = []
    for goal in targets:

        def error(q, goal=goal):
            t = np.eye(4)
            for i, origin, axis in chain:
                t = t @ origin
                if i is not None:
                    t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis * q[i]).as_matrix()
            return np.r_[5 * (t[:3, 3] - goal[:3, 3]), Rotation.from_matrix(goal[:3, :3].T @ t[:3, :3]).as_rotvec()]

        rng = np.random.default_rng(42)
        for attempt in range(5):
            start = seed if not attempt else seed + rng.normal(0, 0.45, len(seed))
            q = least_squares(error, np.clip(start, lo, hi), bounds=(lo, hi), max_nfev=90).x
            residual = error(q)
            if np.linalg.norm(residual[:3]) > 0.005 or np.linalg.norm(residual[3:]) > 0.01:
                continue
            candidate = dict(zip(names, q.tolist(), strict=True))
            try:
                gate.check_state(state | candidate)
            except ValueError as exc:
                if "physical_geometry_collision" not in str(exc):
                    raise
                continue
            answers.append(candidate)
            seed = q
            break
        else:
            return []
    return answers
