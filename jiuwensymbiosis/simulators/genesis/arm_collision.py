"""Physical snapshots for cuRobo obstacle planning and live obstacle monitoring."""

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
                    # Approximate the floor halfspace within the planning workspace.
                    if is_robot:
                        raise ValueError("robot plane geometry unsupported")
                    center = (pose @ local) @ np.array([0.0, 0.0, -0.05, 1.0])
                    quat = planning.Rotation.from_matrix((pose @ local)[:3, :3]).as_quat(scalar_first=True)
                    world["cuboid"][f"geom_{geom.idx}"] = {"dims": [200.0, 200.0, 0.1], "pose": [*center[:3], *quat]}
                else:
                    vertices = array(geom.init_verts)
                    record.update(
                        vertices=vertices,
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
    # Exclude the measured contact support from lift planning to allow departure.
    for attachment in attachments.values():
        for idx in attachment.get("support_geoms", []):
            world["cuboid"].pop(f"geom_{idx}", None)
    return {
        "geometries": geometries,
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
