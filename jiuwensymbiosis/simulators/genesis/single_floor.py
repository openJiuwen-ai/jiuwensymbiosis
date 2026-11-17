"""One physical support plane; source meshes remain read-only navigation evidence."""

from __future__ import annotations

import math

import numpy as np


class SingleFloor:
    """No walls/furniture collision bodies, OBJ exports, decomposition or room SDFs."""

    def __init__(self, scene, meshes, *, floor_z=None, radius=0.15, height=1.35):
        import genesis as gs

        if __package__:
            from .indoor import supported_spawn
        else:
            from indoor import supported_spawn

        if floor_z is not None and (
            isinstance(floor_z, bool) or not isinstance(floor_z, (int, float)) or not math.isfinite(floor_z)
        ):
            raise ValueError("floor_z must be finite metres in world Z-up coordinates")
        self.meshes = meshes
        self.spawn = supported_spawn(meshes, radius=radius, height=height, floor_z=floor_z)
        self.height = float(self.spawn[2])
        self.entity = scene.add_entity(
            morph=gs.morphs.Plane(pos=(0, 0, self.height), visualization=False),
            material=gs.materials.Rigid(friction=0.8),
            surface=gs.surfaces.Collision(),
        )

    def summary(self):
        return {
            "collision_method": "single_floor_plane",
            "house_collision_entities": 0,
            "ground_collision_entities": 1,
            "floor_height_m": self.height,
            "floor_selection": "supported_horizontal_source_surface",
            "floor_extent": "infinite_plane; navigation remains restricted to source supported space",
            "room_geometry_role": "navigation_admission_only",
            "wall_furniture_contacts_available": False,
        }

    def contacts(self, robot):
        """Measured ground contacts only; never imply walls/furniture were collision-free."""
        contacts = robot.entity.get_contacts(exclude_self_contact=True)

        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

        a, b = array(contacts["geom_a"]).ravel(), array(contacts["geom_b"]).ravel()
        valid = array(contacts.get("valid_mask", np.ones(a.shape, dtype=bool))).ravel().astype(bool)
        ground_a = (a >= self.entity.geom_start) & (a < self.entity.geom_end)
        ground_b = (b >= self.entity.geom_start) & (b < self.entity.geom_end)
        mask = valid & (ground_a | ground_b)
        forces = array(contacts["force_a"]).reshape(-1, 3)[mask]
        # force_a is the force on geom_a. Report forces on the robot, not on the floor.
        forces = forces * np.where(ground_a[mask], -1, 1)[:, None]
        positions = array(contacts["position"]).reshape(-1, 3)[mask]
        surfaces = []
        if mask.any():
            surfaces.append(
                {
                    "path": "simulation://single-floor",
                    "count": int(mask.sum()),
                    "force_n": float(np.linalg.norm(forces, axis=1).sum()),
                    "horizontal_force_n": float(np.linalg.norm(forces[:, :2], axis=1).sum()),
                    "force_xyz_n": forces.sum(axis=0).tolist(),
                    "position_m": positions.mean(axis=0).tolist(),
                }
            )
        return {
            "count": int(mask.sum()),
            "surfaces": surfaces,
            "scope": "ground_only",
            "wall_furniture_contacts_available": False,
        }
