"""CPU snapshots and loaded-geometry sampled collision checks, not continuous collision proof."""

from __future__ import annotations

from importlib import import_module

import numpy as np


def _planning():
    return import_module(f"{__package__}.arm_planning" if __package__ else "arm_planning")


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).copy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite collision snapshot")
    return result


def link_pose(link):
    """Finite, single-environment link pose shared by planning and payload readback."""
    return _planning().transform(array(link.get_pos()).reshape(3), array(link.get_quat()).reshape(4))


def snapshot(entity, profile, attachments=None):
    """Call on the scene owner thread after build, before background planning."""
    attachments = attachments or {}
    carried_bodies = [body for body in entity.solver.entities if body.idx in attachments]
    if any(len(body.links) != 1 for body in carried_bodies):
        raise ValueError("single-link payload required")
    planning = _planning()
    base = entity.get_link(profile["base_link"])
    world_from_base = link_pose(base)
    base_from_world = np.linalg.inv(world_from_base)
    own = {g.idx for link in entity.links for g in link.geoms}
    carried = {g.idx for body in carried_bodies for g in body.links[0].geoms}
    # Genesis 1.4.4 exposes no public built-pair table; missing ABI refuses the snapshot.
    pairs = array(getattr(entity.solver.collider, "_valid_collision_pairs"))  # noqa: B009 -- SDK ABI
    pairs = pairs.astype(int).reshape(-1, 2)
    pairs = [tuple(map(int, p)) for p in pairs if p[0] in own | carried or p[1] in own | carried]
    needed = {idx for pair in pairs for idx in pair} | own
    geometries = {}
    world = {"cuboid": {}}
    dynamic = {}
    for other in entity.solver.entities:
        attachment = attachments.get(other.idx)
        for link in other.links:
            pose = base_from_world @ link_pose(link)
            for geom in link.geoms:
                if geom.idx not in needed:
                    continue
                local = planning.transform(array(geom.init_pos), array(geom.init_quat))
                is_robot = geom.idx in own
                kind = getattr(geom.type, "name", str(geom.type)).upper()
                if not is_robot and not link.is_fixed:
                    if attachment is None:
                        if (
                            np.linalg.norm(array(link.get_vel())) > 0.008
                            or np.linalg.norm(array(link.get_ang())) > 0.02
                        ):
                            raise ValueError("moving obstacle unsupported")
                        dynamic[int(link.idx)] = link_pose(link)
                record = {"link": link.name if is_robot else None, "pose": local if is_robot else pose @ local}
                if attachment is not None:
                    record.update(
                        link=attachment["tool"],
                        pose=attachment["relative"] @ local,
                        payload=True,
                        contact_links=attachment["contact_links"],
                        object_id=other.idx,
                        support_geoms=attachment.get("support_geoms", []),
                    )
                if "PLANE" in kind:
                    record["plane"] = True
                    # Workspace bounding box; the independent gate retains the exact halfspace.
                    if is_robot:
                        raise ValueError("robot plane geometry unsupported")
                    center = (pose @ local) @ np.array([0.0, 0.0, -0.05, 1.0])
                    quat = planning.Rotation.from_matrix((pose @ local)[:3, :3]).as_quat(scalar_first=True)
                    world["cuboid"][f"geom_{geom.idx}"] = {"dims": [200.0, 200.0, 0.1], "pose": [*center[:3], *quat]}
                else:
                    vertices = array(geom.init_verts)
                    record.update(
                        vertices=vertices,
                        faces=array(geom.init_faces).astype(int),
                        convex=bool(geom.is_convex),
                    )
                    if not is_robot and attachment is None:
                        lo, hi = vertices.min(0), vertices.max(0)
                        center = record["pose"] @ np.r_[(lo + hi) / 2, 1.0]
                        quat = planning.Rotation.from_matrix(record["pose"][:3, :3]).as_quat(scalar_first=True)
                        world["cuboid"][f"geom_{geom.idx}"] = {
                            "dims": np.maximum(hi - lo, 1e-5).tolist(),
                            "pose": [*center[:3], *quat],
                        }
                geometries[int(geom.idx)] = record
    if needed - geometries.keys():
        raise ValueError("incomplete physical collision snapshot")
    # Remove measured supports only from coarse spheres; full-geometry checks retain every support.
    for attachment in attachments.values():
        for idx in attachment.get("support_geoms", []):
            world["cuboid"].pop(f"geom_{idx}", None)
    return {
        "geometries": geometries,
        "pairs": pairs,
        "world": world,
        "world_from_base": world_from_base,
        "dynamic": dynamic,
    }


def world_unchanged(entity, captured):
    """Recheck resting obstacles throughout planning and physical execution."""
    planning = _planning()
    links = {link.idx: link for body in entity.solver.entities for link in body.links}
    for idx, original in captured.get("dynamic", {}).items():
        link = links.get(int(idx))
        if link is None:
            return False
        if np.linalg.norm(array(link.get_vel())) > 0.008 or np.linalg.norm(array(link.get_ang())) > 0.02:
            return False
        current = link_pose(link)
        error = planning.pose_error(original, current)
        if error["position_m"] > 0.002 or error["rotation_rad"] > 0.02:
            return False
    return True


class CollisionGate:
    def __init__(self, root, profile, captured):
        import fcl

        self.fcl, self.root, self.profile, self.captured = fcl, root, profile, captured
        self.objects = {}
        uncontrolled_links = {
            joint.find("child").get("link")
            for joint in root.findall("joint")
            if joint.get("name") in profile.get("uncontrolled_support_joints", [])
        }
        for idx, record in captured["geometries"].items():
            if record["link"] in uncontrolled_links:
                vertices = record["vertices"] @ record["pose"][:3, :3].T + record["pose"][:3, 3]
                shape = fcl.Sphere(float(np.max(np.linalg.norm(vertices, axis=1))))
                record["pose"] = np.eye(4)
            elif record.get("plane"):
                shape = fcl.Halfspace(np.array([0.0, 0.0, 1.0]), 0.0)
            elif record["convex"]:
                faces = np.column_stack((np.full(len(record["faces"]), 3), record["faces"])).reshape(-1)
                shape = fcl.Convex(record["vertices"], len(record["faces"]), faces)
            else:
                shape = fcl.BVHModel()
                shape.beginModel(num_tris_=len(record["faces"]), num_vertices_=len(record["vertices"]))
                shape.addSubModel(record["vertices"], record["faces"])
                shape.endModel()
            self.objects[idx] = fcl.CollisionObject(shape)
        self.support = set(profile.get("support_links", []))
        if not self.support <= {link.get("name") for link in root.findall("link")}:
            raise ValueError("unknown support link")

    def coverage(self):
        result = {}
        for record in self.captured["geometries"].values():
            link = record["link"]
            if link is None or record.get("plane") or record.get("payload"):
                continue
            vertices = record["vertices"] @ record["pose"][:3, :3].T + record["pose"][:3, 3]
            # Vertices and centroids quantify coverage; they do not prove conservative coverage.
            points = np.concatenate((vertices, vertices[record["faces"]].mean(axis=1)))
            spheres = self.profile["collision_spheres"].get(link, [])
            if spheres:
                centers = np.array([s["center"] for s in spheres])
                radii = np.array([s["radius"] for s in spheres])
                gap = np.min(np.linalg.norm(points[:, None, :] - centers, axis=-1) - radii, axis=1)
                uncovered, maximum = int((gap > 1e-5).sum()), float(max(0.0, gap.max()))
            else:
                uncovered, maximum = len(points), None
            result.setdefault(link, []).append(
                {
                    "samples": len(points),
                    "uncovered": uncovered,
                    "max_uncovered_m": maximum,
                    "sphere_count": len(spheres),
                }
            )
        return result

    def check_state(self, state):
        fk = _planning().forward_kinematics(self.root, state)
        base_inverse = np.linalg.inv(fk[self.profile["base_link"]])
        for idx, record in self.captured["geometries"].items():
            pose = record["pose"] if record["link"] is None else base_inverse @ fk[record["link"]] @ record["pose"]
            self.objects[idx].setTransform(self.fcl.Transform(pose[:3, :3], pose[:3, 3]))
        for first, second in self.captured["pairs"]:
            a, b = self.captured["geometries"][first], self.captured["geometries"][second]
            grasp_a = a.get("payload") and b["link"] in a["contact_links"] and not b.get("payload")
            grasp_b = b.get("payload") and a["link"] in b["contact_links"] and not a.get("payload")
            if grasp_a or grasp_b:
                continue
            # Floor contact only for authored support links; other collision checks still apply.
            floor_a = a["link"] in self.support and b.get("plane")
            floor_b = b["link"] in self.support and a.get("plane")
            if floor_a or floor_b:
                continue
            result = self.fcl.CollisionResult()
            request = self.fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
            if self.fcl.collide(self.objects[first], self.objects[second], request, result):
                support_contact = (a.get("payload") and second in a.get("support_geoms", [])) or (
                    b.get("payload") and first in b.get("support_geoms", [])
                )
                up = np.asarray(self.captured.get("world_from_base", np.eye(4)))[:3, :3].T @ np.array([0.0, 0.0, 1.0])
                bounded = bool(result.contacts) and len(result.contacts) < 1000
                shallow = all(
                    c.penetration_depth <= 0.001 and abs(np.dot(c.normal, up)) >= 0.9 for c in result.contacts
                )
                if support_contact and bounded and shallow:
                    continue
                raise ValueError(f"physical_geometry_collision: {a['link'] or first} / {b['link'] or second}")

    def check(self, trajectory, *, max_joint_step=0.01):
        if not np.isfinite(max_joint_step) or max_joint_step <= 0:
            raise ValueError("invalid collision sampling interval")
        count = 0
        state = dict(trajectory.start)
        samples = trajectory.sample(np.linspace(0.0, trajectory.duration, max(2, len(trajectory.positions) * 4)))
        for first, second in zip(samples[:-1], samples[1:], strict=True):
            n = max(1, int(np.ceil(np.max(np.abs(second - first)) / max_joint_step)))
            for alpha in np.linspace(0.0, 1.0, n + 1):
                state.update(zip(trajectory.names, first + (second - first) * alpha, strict=True))
                self.check_state(state)
                count += 1
        return {
            "checked_samples": count,
            "max_joint_step_rad_or_m": max_joint_step,
            "method": "loaded_geometry_fcl_sampled",
            "continuous_collision_proof": False,
        }
