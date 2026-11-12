"""Scene-owned profile-driven motors and payload evidence; model owns mimic coupling."""

from __future__ import annotations

from importlib import import_module

import numpy as np


def sibling(name):
    return import_module(f"{__package__}.{name}" if __package__ else name)


def pose(link):
    return sibling("arm_collision").link_pose(link)


class EndEffectors:
    def __init__(self, entity, root, configurations):
        self.entity, self.configurations = entity, configurations
        self.evidence = {}
        self.targets, self.ids, self.groups = {}, {}, {}
        self.opening, self.latched, self.support_geoms = {}, {}, {}
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        links = {link.name: link for link in entity.links}
        used = set()
        for name, config in configurations.items():
            if config["tool"] not in links:
                raise ValueError(f"unknown effector tool: {name}")
            motors = config["joints"]
            if not motors or used.intersection(motors):
                raise ValueError("empty or shared effector motors")
            used.update(motors)
            for joint_name, motor in motors.items():
                joint = joints.get(joint_name)
                if (
                    joint is None
                    or joint.get("type") not in {"revolute", "prismatic"}
                    or joint.find("mimic") is not None
                ):
                    raise ValueError("effector motor must be an independent bounded joint")
                limit = joint.find("limit")
                values = [motor[k] for k in ("open", "close", "velocity", "effort", "kp", "kd", "tolerance")]
                invalid_values = any(isinstance(v, bool) for v in values) or not np.isfinite(values).all()
                if invalid_values or min(values[2:]) <= 0 or values[0] == values[1]:
                    raise ValueError("invalid effector motor settings")
                preload = motor.get("preload", 0.0)
                if not np.isfinite(preload) or not 0 <= preload <= abs(motor["close"] - motor["open"]) / 4:
                    raise ValueError("invalid effector contact preload")
                if (
                    limit is None
                    or not {"lower", "upper"} <= limit.attrib.keys()
                    or any(not float(limit.get("lower")) <= q <= float(limit.get("upper")) for q in values[:2])
                ):
                    raise ValueError("effector target outside model limits")
                if motor["effort"] > float(limit.get("effort", "0")) or motor["velocity"] > float(
                    limit.get("velocity", "0")
                ):
                    raise ValueError("effector actuation exceeds authored limits")
            groups = config["contact_groups"]
            if len(groups) < 2 or any(not group or set(group) - links.keys() for group in groups):
                raise ValueError("at least two known contact groups required")
            flat = [link for group in groups for link in group]
            if len(set(flat)) != len(flat):
                raise ValueError("contact groups must be disjoint")
            self.groups[name] = [{links[link].idx for link in group} for group in groups]
            ids = [entity.get_joint(j).dofs_idx_local[0] for j in motors]
            self.ids[name] = ids
            self.targets[name] = sibling("arm_collision").array(entity.get_dofs_position(ids)).reshape(-1)
            self.opening[name] = False
            self.latched[name] = False
            self.support_geoms[name] = set()
            self.evidence[name] = sibling("grasp_evidence").GraspEvidence()
        # Validate every effector before changing any motor configuration.
        for name, config in configurations.items():
            motors, ids = list(config["joints"].values()), self.ids[name]
            entity.set_dofs_kp([m["kp"] for m in motors], ids)
            entity.set_dofs_kv([m["kd"] for m in motors], ids)
            effort = np.array([m["effort"] for m in motors])
            entity.set_dofs_force_range(-effort, effort, ids)
        self.command = None
        self.result = {}
        self.elapsed = self.stable = 0.0

    def bodies(self):
        return {body.idx: body for body in self.entity.solver.entities if body is not self.entity}

    def contact_forces(self, body):
        array = sibling("arm_collision").array
        contacts = self.entity.get_contacts(with_entity=body)
        a = array(contacts["link_a"]).reshape(-1)
        b = array(contacts["link_b"]).reshape(-1)
        forces = np.linalg.norm(array(contacts["force_a"]).reshape(-1, 3), axis=1)
        if not len(a) == len(b) == len(forces):
            raise ValueError("inconsistent contact readback")
        valid = array(contacts.get("valid_mask", np.ones(len(a), dtype=bool))).reshape(-1).astype(bool)
        penetration = array(contacts["penetration"]).reshape(-1)[valid]
        if penetration.max(initial=0.0) > 0.002:
            raise ValueError("robot-object penetration exceeds grasp tolerance")
        return a[valid], b[valid], forces[valid]

    def supported(self, body):
        array = sibling("arm_collision").array
        contacts = body.get_contacts()
        a = array(contacts["link_a"]).reshape(-1)
        b = array(contacts["link_b"]).reshape(-1)
        force = array(contacts["force_a"]).reshape(-1, 3)
        valid = array(contacts.get("valid_mask", np.ones(len(a), dtype=bool))).reshape(-1).astype(bool)
        a, b, force = a[valid], b[valid], force[valid]
        own = {link.idx for link in body.links}
        robot = {link.idx for link in self.entity.links}
        # Upward external force only: exclude robot links and reverse force_a when the object is link_b.
        vertical = sum(
            (f[2] if x in own else -f[2])
            for x, y, f in zip(a, b, force, strict=True)
            if (x in own or y in own) and x not in robot and y not in robot
        )
        return vertical > 0.05

    def measure(self, dt):
        array = sibling("arm_collision").array
        bodies = self.bodies()
        for name, evidence in self.evidence.items():
            config = self.configurations[name]
            q = array(self.entity.get_dofs_position(self.ids[name])).reshape(-1)
            opened = all(
                abs(value - motor["open"]) <= motor["tolerance"]
                for value, motor in zip(q, config["joints"].values(), strict=True)
            )
            candidates = []
            for idx, body in bodies.items():
                if all(link.is_fixed for link in body.links):
                    continue
                a, b, forces = self.contact_forces(body)
                contact = all(
                    float(forces[np.isin(a, list(group)) | np.isin(b, list(group))].sum()) > 0.05
                    for group in self.groups[name]
                )
                if contact or idx == evidence.object_id:
                    allowed = set().union(*self.groups[name])
                    forbidden = {link.idx for link in self.entity.links} - allowed
                    if forces[np.isin(a, list(forbidden)) | np.isin(b, list(forbidden))].sum() > 0.05:
                        raise ValueError("payload contacts a non-grasp robot link")
                    candidates.append((idx, body, contact))
            if len(candidates) > 1:
                evidence.state = "unknown"
                raise ValueError("ambiguous grasp: multiple contacted objects")
            if not candidates:
                if evidence.object_id is not None:
                    evidence.state = "unknown"
                    raise ValueError("payload disappeared")
                continue
            idx, body, contact = candidates[0]
            if len(body.links) != 1:
                raise ValueError("articulated payloads require a deformation model")
            tool_pose = pose(self.entity.get_link(config["tool"]))
            object_pose = pose(body.links[0])
            supported = self.supported(body)
            if supported and evidence.state in {"clear", "contact"}:
                contacts = body.get_contacts()
                a = array(contacts["geom_a"]).reshape(-1)
                b = array(contacts["geom_b"]).reshape(-1)
                valid = array(contacts.get("valid_mask", np.ones(len(a), dtype=bool))).reshape(-1).astype(bool)
                fixed_links = []
                for other in bodies.values():
                    fixed_links.extend(link for link in other.links if link.is_fixed)
                fixed = {g.idx for link in fixed_links for g in link.geoms}
                self.support_geoms[name].update(int(idx) for idx in np.r_[a[valid], b[valid]] if idx in fixed)
            evidence.observe(
                dt=dt,
                object_id=idx,
                relative=np.linalg.inv(tool_pose) @ object_pose,
                height=object_pose[2, 3],
                contact=contact,
                supported=supported,
                speed=float(np.linalg.norm(array(body.get_vel()))),
                angular_speed=float(np.linalg.norm(array(body.get_ang()))),
                opened=opened,
            )
        owners = [e.object_id for e in self.evidence.values() if e.object_id is not None and e.state != "clear"]
        if len(owners) != len(set(owners)):
            raise ValueError("shared-object multi-effector grasp requires a coupled controller")

    def start(self, name, *, opened):
        if name not in self.evidence or not isinstance(opened, bool):
            raise ValueError("known effector and boolean opened required")
        evidence = self.evidence[name]
        if opened and evidence.holding is not False and evidence.support_time < evidence.release_s:
            raise ValueError("release requires measured stable support")
        if not opened and evidence.state == "unknown":
            raise ValueError("unknown payload requires supported release")
        self.command = name
        if evidence.state == "clear" and not opened:
            self.support_geoms[name].clear()
            evidence.object_id = evidence.reference = evidence.height = None
            evidence.released_time = evidence.support_time = 0.0
        if opened or not self.latched[name]:
            self.targets[name] = (
                sibling("arm_collision").array(self.entity.get_dofs_position(self.ids[name])).reshape(-1)
            )
        self.opening[name] = opened
        if opened:
            self.latched[name] = False
        self.elapsed = self.stable = 0.0
        self.result = {}

    def cancel(self):
        # Keep the last bounded motor target. Never open on cancellation.
        self.command = None

    def tick(self, dt):
        self.measure(dt)
        name = self.command
        if name is None:
            return
        evidence, config = self.evidence[name], self.configurations[name]
        motors = list(config["joints"].values())
        opened = self.opening[name]
        if evidence.state == "unknown" and not opened:
            self.result = {**evidence.report(), "ok": False, "reason": evidence.reason or "payload_unknown"}
            self.command = None
            return
        target = np.array([m["open" if opened else "close"] for m in motors])
        # Stop advancing at sustained contact; retries retain exactly this target.
        if opened or (not self.latched[name] and evidence.contact_time < 0.025):
            rate = np.array([m["velocity"] for m in motors]) * dt
            self.targets[name] += np.clip(target - self.targets[name], -rate, rate)
        elif not self.latched[name]:
            q = sibling("arm_collision").array(self.entity.get_dofs_position(self.ids[name])).reshape(-1)
            preload = np.array([m.get("preload", 0.0) for m in motors])
            self.targets[name] = q + np.sign(target - q) * np.minimum(preload, abs(target - q))
            self.latched[name] = True
        self.entity.control_dofs_position(self.targets[name], self.ids[name])
        array = sibling("arm_collision").array
        q = array(self.entity.get_dofs_position(self.ids[name])).reshape(-1)
        v = array(self.entity.get_dofs_velocity(self.ids[name])).reshape(-1)
        reached = (
            all(abs(value - m["open"]) < m["tolerance"] for value, m in zip(q, motors, strict=True))
            if opened
            else evidence.contact_time >= evidence.contact_s
        )
        settled = max(abs(v)) < 0.02
        self.stable = self.stable + dt if reached and settled else 0.0
        self.elapsed += dt
        if self.stable >= 0.25 or self.elapsed > 5.0:
            success = self.stable >= 0.25 and (not opened or evidence.holding is False)
            self.result = {
                **evidence.report(),
                "ok": success,
                "reason": None if success else "grasp_or_release_unconfirmed",
            }
            self.command = None

    def report(self):
        return {name: evidence.report() for name, evidence in self.evidence.items()}

    def invalidate(self):
        for evidence in self.evidence.values():
            evidence.state = "unknown"
            evidence.support_time = evidence.contact_time = 0.0

    def attachments(self):
        result = {}
        bodies = self.bodies()
        for name, evidence in self.evidence.items():
            if evidence.state == "unknown":
                raise ValueError("payload state unknown")
            if evidence.object_id is None or evidence.state == "clear":
                continue
            if evidence.contact_time < evidence.contact_s:
                raise ValueError("payload contact not stable")
            config = self.configurations[name]
            body = bodies[evidence.object_id]
            result[evidence.object_id] = {
                "tool": config["tool"],
                "relative": np.linalg.inv(pose(self.entity.get_link(config["tool"]))) @ pose(body.links[0]),
                "contact_links": [link for group in config["contact_groups"] for link in group],
                "support_geoms": sorted(self.support_geoms[name]),
            }
        return result

    def unchanged(self, captured):
        """Bind a plan to its own payload measurement, not just the first grasp."""
        try:
            bodies = self.bodies()
            for idx, record in captured.items():
                current = np.linalg.inv(pose(self.entity.get_link(record["tool"]))) @ pose(bodies[idx].links[0])
                error = sibling("arm_planning").pose_error(record["relative"], current)
                if error["position_m"] > 0.002 or error["rotation_rad"] > 0.02:
                    return False
            return True
        except (ValueError, KeyError):
            return False
