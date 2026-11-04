"""A planned trial lift. Geometry labels stay inside the physics/collision worker."""

from __future__ import annotations

from importlib import import_module

import numpy as np


def sibling(name):
    return import_module(f"{__package__}.{name}" if __package__ else name)


class PickLift:
    def __init__(self, arm, effector, center):
        ee = arm.effectors
        if any(e.holding is not False for e in ee.evidence.values()):
            raise ValueError("trial lift requires clear effectors")
        descriptor = next(c for c in arm.profile["grasp_pose_backends"].values() if c["effector"] == effector)
        self.name = effector
        self.ee = ee
        self.tool = arm.entity.get_link(descriptor["control_link"])
        self.pose = sibling("end_effectors").pose
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("invalid sensed target center")
        center = (arm.base_pose() @ np.r_[center, 1])[:3]
        canonical = self.pose(self.tool) @ np.linalg.inv(descriptor["T_gripper_control"])
        sweep = descriptor["sweep_volume_params"]
        tip = (canonical @ np.r_[sweep["offset_open"], 1])[:3]
        if np.linalg.norm(center - tip) > max(sweep["extents_open"]):
            raise ValueError("gripper is outside sensed target region")
        # Identify the single obstacle at the observed point for collision modelling
        # and independent outcome scoring. This never supplies a grasp pose.
        matches = []
        for body in ee.bodies().values():
            if len(body.links) != 1 or body.links[0].is_fixed:
                continue
            vertices = np.concatenate(
                [sibling("arm_collision").array(g.get_verts()).reshape(-1, 3) for g in body.geoms]
            )
            if np.all(center >= vertices.min(0) - 0.005) and np.all(center <= vertices.max(0) + 0.005):
                matches.append(body)
        if len(matches) != 1 or not ee.supported(matches[0]):
            raise ValueError("trial lift requires one supported body at the sensed position")
        self.body = matches[0]
        body_pose = self.pose(self.body.links[0])
        self.initial_z = body_pose[2, 3]
        self.rise = 0.0
        contacts = self.body.get_contacts()
        valid = sibling("arm_collision").array(contacts["valid_mask"]).astype(bool).reshape(-1)
        touching = set(
            np.concatenate(
                [sibling("arm_collision").array(contacts[k]).reshape(-1)[valid] for k in ("geom_a", "geom_b")]
            )
        )
        support = sorted(
            g.idx
            for b in ee.bodies().values()
            for link in b.links
            if link.is_fixed
            for g in link.geoms
            if g.idx in touching
        )
        self.payload = {
            self.body.idx: {
                "tool": self.tool.name,
                "relative": np.linalg.inv(self.pose(self.tool)) @ body_pose,
                "contact_links": sum(ee.configurations[effector]["contact_groups"], []),
                "support_geoms": support,
            }
        }
        self.started = False
        self.stable_s = 0.0
        self.reference = None

    def close(self):
        self.ee.start(self.name, opened=False)
        # Keep the independent PD ramp. No close-result or attachment gate.
        self.ee.cancel()
        self.started = True

    def measure(self, dt):
        """Accumulate a fresh unsupported bilateral hold window after the planned lift."""
        ee = self.ee
        groups = ee.groups[self.name]
        allowed = set.union(*groups)
        forbidden = {link.idx for link in ee.entity.links} - allowed
        a, b, forces = ee.contact_forces(self.body)
        if forces[np.isin(a, list(forbidden)) | np.isin(b, list(forbidden))].sum() > 0.05:
            raise ValueError("payload contacts a non-grasp link")
        for body in ee.bodies().values():
            if body is self.body or all(link.is_fixed for link in body.links):
                continue
            _, _, other = ee.contact_forces(body)
            if other.sum() > 0.05:
                raise ValueError("trial lift contacts another body")
        obj = self.pose(self.body.links[0])
        relative = np.linalg.inv(self.pose(self.tool)) @ obj
        self.rise = float(obj[2, 3] - self.initial_z)
        bilateral = all(forces[np.isin(a, list(group)) | np.isin(b, list(group))].sum() > 0.05 for group in groups)
        if not bilateral or self.rise < 0.05 or ee.supported(self.body):
            self.reference = None
            self.stable_s = 0.0
            return
        if self.reference is None:
            self.reference = relative
        delta = np.linalg.inv(self.reference) @ relative
        translation = np.linalg.norm(delta[:3, 3])
        rotation = sibling("arm_planning").Rotation.from_matrix(delta[:3, :3]).magnitude()
        evidence = ee.evidence[self.name]
        if translation > evidence.slip_m or rotation > evidence.slip_rad:
            self.stable_s = 0.0
            self.reference = relative
        else:
            self.stable_s += dt

    def finish(self):
        if self.stable_s < 1.0:
            return False
        evidence = self.ee.evidence[self.name]
        evidence.state = "held"
        evidence.object_id = self.body.idx
        evidence.reference = self.reference.copy()
        evidence.height = self.initial_z
        evidence.contact_time = self.stable_s
        evidence.support_time = evidence.loss_time = 0.0
        evidence.reason = None
        self.ee.support_geoms[self.name] = set(self.payload[self.body.idx]["support_geoms"])
        return True

    def uncertain(self):
        if self.started:
            evidence = self.ee.evidence[self.name]
            evidence.state = "unknown"
            evidence.object_id = self.body.idx
            evidence.reason = "trial_lift_unconfirmed"
