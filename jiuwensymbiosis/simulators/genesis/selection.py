"""Resolve Nyx's USD hit point to visible USD geometry, without changing assets.

Nyx 0.1.7 reports an empty node name for a USD imported as one Mesh instance.
Use its first-hit position (occlusion comes from the renderer), then verify the
surface against USD triangles. A containing bounding box alone is not a hit.
Heavy USD/numpy imports stay inside the worker and this lazy index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class UsdPickIndex:
    def __init__(self, source: Path):
        from pxr import Usd, UsdGeom

        self.stage = Usd.Stage.Open(str(source))
        self.scale = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.y_up = str(UsdGeom.GetStageUpAxis(self.stage)) == "Y"
        self.bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        self.transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.meshes = []
        self.triangles: dict[str, Any] = {}
        for prim in Usd.PrimRange.Stage(self.stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            image = UsdGeom.Imageable(prim)
            if image.ComputeVisibility() == "invisible" or image.ComputePurpose() not in {"default", "render"}:
                continue
            self.meshes.append((prim, self.world_bounds(prim)))

    def world_bounds(self, prim) -> list[list[float]]:
        box = self.bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        low, high = list(box.GetMin()), list(box.GetMax())
        if self.y_up:
            low, high = [low[0], -high[2], low[1]], [high[0], -low[2], high[1]]
        return [[float(v) * self.scale for v in row] for row in (low, high)]

    def _triangles(self, prim):
        import numpy as np
        from pxr import UsdGeom

        path = str(prim.GetPath())
        if path not in self.triangles:
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            matrix = np.asarray(self.transforms.GetLocalToWorldTransform(prim))
            points = (points @ matrix[:3, :3] + matrix[3, :3]) * self.scale
            if self.y_up:
                points = points[:, [0, 2, 1]] * [1, -1, 1]
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            holes = set(mesh.GetHoleIndicesAttr().Get() or [])
            # The scene assets are triangulated. Fan expansion also covers their
            # convex quads; do not invent triangles across authored hole faces.
            if np.all(counts == 3) and not holes:
                faces = indices.reshape(-1, 3)
            else:
                face_indices = []
                start = 0
                for face, count in enumerate(counts):
                    if face not in holes:
                        face_indices.extend(
                            (indices[start], indices[start + j], indices[start + j + 1]) for j in range(1, count - 1)
                        )
                    start += count
                faces = np.asarray(face_indices, dtype=int).reshape(-1, 3)
            triangles = points[faces].astype(np.float32)
            self.triangles[path] = (triangles, triangles.min(axis=1), triangles.max(axis=1))
        return self.triangles[path]

    def resolve(self, position, *, tolerance: float = 0.002) -> dict:
        """Return the nearest verified visible surface and its USD model bounds."""
        import numpy as np
        import trimesh

        point = np.asarray(position, dtype=float)
        best = None
        best_distance = tolerance
        for prim, (low, high) in self.meshes:
            if np.any(point < np.asarray(low) - tolerance) or np.any(point > np.asarray(high) + tolerance):
                continue
            triangles, lows, highs = self._triangles(prim)
            candidates = triangles[
                np.all(lows <= point + tolerance, axis=1) & np.all(highs >= point - tolerance, axis=1)
            ]
            if candidates.size == 0:
                continue
            closest = trimesh.triangles.closest_point(candidates, np.broadcast_to(point, (len(candidates), 3)))
            distance = float(np.linalg.norm(closest - point, axis=1).min())
            if distance <= best_distance:
                best, best_distance = prim, distance
        if best is None:
            raise ValueError("命中表面未能对应到 USD 网格，请换一个位置点选。")
        model = best
        ancestor = best.GetParent()
        while ancestor and not ancestor.IsPseudoRoot():
            if (
                ancestor.GetMetadata("kind") in {"component", "subcomponent", "assembly"}
                or ancestor.HasAuthoredReferences()
            ):
                model = ancestor
                break
            ancestor = ancestor.GetParent()
        name = model.GetName()
        if name in {"entity", "visual", "body"}:
            name = model.GetParent().GetName()
        return {
            "path": str(model.GetPath()),
            "mesh_path": str(best.GetPath()),
            "name": name,
            "bounds_m": self.world_bounds(model),
            "hit_position_m": [float(v) for v in point],
            "surface_error_m": best_distance,
            "method": "nyx_first_hit+usd_surface",
        }
