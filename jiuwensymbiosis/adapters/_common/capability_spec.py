# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability → contract maps shared by the adapter validator and generator.

Single source of truth so ``scripts/validate_adapter.py`` (the checker) and
``scripts/new_adapter/`` (the generator) never drift on *which* actions a
capability gates, *which* low-level driver members it delegates to, or which of
them the framework already implements. Mirrors ``api/components.py`` and
``env/protocol.py``; the capability vocabulary itself lives in
``env/base.py:KNOWN_CAPABILITIES``.

Kept import-light (plain dict literals, no heavy deps) so the validator can run
even when a robot's hardware packages are absent.
"""

from __future__ import annotations

# Capability → the actions it gates (mirrors ``api/actions.py``; kept as literals so
# the validator stays import-light).
#
# These are what a body MAY implement under that capability, not a menu of mutually
# exclusive flavours: any body whose hardware supports an action can implement it.
# ``locate_for_grasp`` needs depth + hand-eye calibration, which a fixed arm has just
# as much as a mobile one — and since its implementation is now a component a body HOLDS,
# an arm can take that one action without also taking its neighbours. The validator
# therefore asks only that a declared capability has SOME action behind it, never all.
CAPABILITY_ACTIONS: dict[str, list[str]] = {
    "motion.cartesian": ["goto_xyzr", "goto_pose", "move_direction", "get_pose", "get_home_pose"],
    "motion.joint": ["move_joint", "move_named_joint", "get_joint_positions"],
    "grasp.parallel": ["open_gripper", "close_gripper"],
    "grasp.suction": ["activate_suction", "deactivate_suction"],
    "vision.camera": ["get_image", "pixel_to_base_xyz"],
    "vision.detection": ["get_grasp_info_simple", "locate_for_grasp", "locate_for_place", "analyze_scene"],
    "motion.base": ["navigate_relative", "rotate_base", "drive_arc"],
    "motion.lift": ["set_lift_pose", "lift_to_clearance"],
    "motion.waist": ["turn_waist"],
    "motion.goal": ["approach_for_grasp", "approach_for_place"],
    "vision.search": ["search_target"],
    "grasp.dual_arm": ["dual_arm_grasp", "dual_arm_place"],
}

# Actions ``api/defaults.py`` implements generically — an adapter gets these by
# forwarding one line. Everything else under a declared capability is real work the
# adapter must write (vendor calibration, IK, paddle geometry, force confirmation),
# which is what makes forgetting one worth a warning.
ACTIONS_WITH_GENERIC_DEFAULT: frozenset[str] = frozenset({
    "home", "get_pose", "get_home_pose", "goto_xyzr", "move_direction", "move_joint",
    "activate_suction", "deactivate_suction", "open_gripper", "close_gripper",
    "get_image", "navigate_relative", "rotate_base", "drive_arc", "set_lift_pose",
    "turn_waist",
})

# Capability → low-level driver members the Env/Api delegate to (structural
# driver contract, mirrors env/protocol.py). Used by validate [D-14].
CAPABILITY_DRIVER_MEMBERS: dict[str, list[str]] = {
    "motion.cartesian": ["home", "get_pose", "move_to_pose_blocking"],
    "motion.joint": ["move_joint_blocking"],
    "grasp.parallel": ["set_gripper"],
    "grasp.suction": ["set_suction"],
    "vision.camera": ["grab_frames"],
    "vision.detection": ["grab_frames"],
    # Mobile-manipulation rows: the Env verb the generic implementation calls, which an Env
    # without its own forwards to the driver under the same name. A body that writes its own
    # action may of course spell its driver call differently.
    "motion.base": ["navigate_relative", "navigate_arc"],
    "motion.base_servo": [
        "start_base_drive",
        "base_drive_running",
        "steer_base_drive",
        "hold_base_drive",
        "stop_base_drive",
    ],
    "motion.lift": ["set_lifter"],
    "motion.waist": ["turn_waist"],
    "motion.goal": ["navigate_relative"],
    "grasp.dual_arm": ["home"],
}

# Capability → JointTransport members a joint-level (motion_backend=joint_ik)
# adapter's transport seam must expose. For these adapters the RobotDriver is the
# shared KinematicArmDriver (adapters/_common/kinematic_driver) — not a per-adapter
# class — so validate [D-14] checks the transport contract instead of the driver
# one when it finds a JointTransport in the adapter's lowlevel module.
CAPABILITY_TRANSPORT_MEMBERS: dict[str, list[str]] = {
    "motion.cartesian": ["read_arm_joints", "send_arm_joints"],
    "motion.joint": ["read_arm_joints", "send_arm_joints"],
    "grasp.parallel": ["read_effector", "send_effector"],
    "grasp.suction": ["read_effector", "send_effector"],
    "vision.camera": ["grab_frames"],
    "vision.detection": ["grab_frames"],
}

# Capability → the component class (in api/components.py) a body HOLDS to get the SHARED
# ALGORITHM behind that capability, when there is one. These two carry state and body
# hooks, so they cannot be plain functions like the rest of ``api/defaults.py`` — but
# they are held, not inherited, so taking one action never drags in its neighbours.
CAPABILITY_COMPONENT: dict[str, list[str]] = {
    # An eye-in-hand single-gripper arm needs neither; it implements the vision
    # actions directly off its own calibration.
    "vision.detection": ["Scene3D"],
    "motion.goal": ["Approach"],
}
