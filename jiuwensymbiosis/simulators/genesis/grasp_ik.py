"""Screen grasp poses for cuRobo IK reachability before collision-aware planning."""

from __future__ import annotations

if __package__:
    from .arm_ik import solve
    from .grasp_pose import rigid
else:
    from arm_ik import solve
    from grasp_pose import rigid


def screen(arm, group, tool, poses):
    targets = rigid(poses)
    if targets.shape != (3, 4, 4) or arm.profile["groups"][group]["tools"] != [tool]:
        raise ValueError("expected approach/grasp/lift poses in the planning base frame")
    if arm.active or not arm.stopped():
        raise ValueError("grasp screening requires a stationary arm")
    return solve(arm, group, tool, targets)
