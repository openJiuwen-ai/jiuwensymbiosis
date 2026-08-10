# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability → contract maps shared by the adapter validator and generator.

Single source of truth so ``scripts/validate_adapter.py`` (the checker) and
``scripts/new_adapter/`` (the generator) never drift on *which* mixin owns a
capability, *which* low-level driver members a capability delegates to, or
*which* mixin methods stay abstract. Mirrors ``api/mixins.py`` and
``adapters/_common/protocol.py``; the capability vocabulary itself lives in
``env/base.py:KNOWN_CAPABILITIES``.

Kept import-light (plain dict literals, no heavy deps) so the validator can run
even when a robot's hardware packages are absent.
"""

from __future__ import annotations

# Mixin → its still-abstract methods (the ones that ``raise NotImplementedError``
# in api/mixins.py). Only these have no working default; the rest delegate to the
# Env verbs, so NOT overriding them is normal and must not be flagged.
# ``get_grasp_info_simple`` / ``pixel_to_base_xyz`` are now COMPLETE VisionMixin
# defaults built on the ``_project_pixel_to_base_raw`` projection seam, so an
# adapter must override only that seam plus ``analyze_scene``.
MIXIN_ABSTRACT_METHODS: dict[str, list[str]] = {
    "VisionMixin": ["_project_pixel_to_base_raw", "analyze_scene"],
}

# Capability → low-level driver members the Env/Api delegate to (structural
# driver contract, mirrors adapters/_common/protocol.py). Used by validate [D-14].
CAPABILITY_DRIVER_MEMBERS: dict[str, list[str]] = {
    "motion.cartesian": ["home", "get_pose", "move_to_pose_blocking"],
    "motion.joint": ["move_joint_blocking"],
    "grasp.parallel": ["set_gripper"],
    "grasp.suction": ["set_suction"],
    "vision.camera": ["grab_frames"],
    "vision.detection": ["grab_frames"],
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

# Capability → the mixin class (in api/mixins.py) that owns its @robot_tool
# methods. Capabilities absent here are "marker" capabilities (vision.camera /
# vision.depth / sorting.command / speech.tts) — declared on the Env to advertise
# a sensor/trait, but generating no LLM tool.
CAPABILITY_MIXIN: dict[str, str] = {
    "motion.cartesian": "MotionMixin",
    "motion.joint": "JointMotionMixin",
    "grasp.suction": "SuctionMixin",
    "grasp.parallel": "ParallelGripperMixin",
    "vision.detection": "VisionMixin",
}
