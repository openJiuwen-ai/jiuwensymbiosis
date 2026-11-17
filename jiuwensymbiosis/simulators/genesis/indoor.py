"""Read-only room geometry for floor selection and logical navigation admission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh


@dataclass
class RoomMesh:
    path: str
    mesh: trimesh.Trimesh


def read_room_meshes(source: Path) -> list[RoomMesh]:
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    y_up = UsdGeom.GetStageUpAxis(stage) == "Y"
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == "invisible" or image.ComputePurpose() not in {"default", "render"}:
            continue
        usd = UsdGeom.Mesh(prim)
        vertices = np.asarray(usd.GetPointsAttr().Get(), dtype=float)
        if vertices.size == 0:
            continue
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        vertices = (np.column_stack((vertices, np.ones(len(vertices)))) @ matrix)[:, :3] * scale
        if y_up:
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        indices = np.asarray(usd.GetFaceVertexIndicesAttr().Get(), dtype=int)
        holes = set(usd.GetHoleIndicesAttr().Get() or [])
        faces: list[tuple[int, int, int]] = []
        offset = 0
        for i, count in enumerate(usd.GetFaceVertexCountsAttr().Get()):
            end = offset + count
            poly = indices[offset:end]
            if i not in holes:
                faces.extend((poly[0], poly[j], poly[j + 1]) for j in range(1, count - 1))
            offset += count
        if not faces or not np.isfinite(vertices).all():
            raise ValueError(f"室内网格无效：{prim.GetPath()}")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        mesh.fix_normals(multibody=True)
        result.append(RoomMesh(str(prim.GetPath()), mesh))
    if not result:
        raise ValueError("室内场景没有可用的可见网格。")
    return result


def _inside_xy(points, triangles):
    """Point-in-triangle, excluding zero-area projections (vertical walls)."""
    a, b, c = (triangles[:, i, :2] for i in range(3))

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    area = cross(b - a, c - a)
    u = cross(b - a, points - a)
    v = cross(c - b, points - b)
    w = cross(a - c, points - c)
    return (np.abs(area) > 1e-10) & (
        ((u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)) | ((u <= 1e-9) & (v <= 1e-9) & (w <= 1e-9))
    )


def supported_spawn(
    meshes: list[RoomMesh], *, radius: float = 0.42, height: float = 1.95, floor_z: float | None = None
) -> tuple:
    """Choose a flat supported footprint with cylindrical whole-body clearance.

    The caller supplies its conservative body footprint; this is not a navigation map.
    The bounded search fails closed when it cannot establish a supported location.
    """
    low = np.min([item.mesh.bounds[0] for item in meshes], axis=0)
    high = np.max([item.mesh.bounds[1] for item in meshes], axis=0)
    floor_faces = []
    for item in meshes:
        tri = item.mesh.triangles
        flat = np.ptp(tri[:, :, 2], axis=1) < 1e-5
        levels = tri[:, :, 2].mean(axis=1)
        if item.mesh.is_watertight:
            flat &= item.mesh.face_normals[:, 2] > 0.5
        if floor_z is not None:
            flat &= np.abs(levels - floor_z) <= 0.005
        floor_faces.extend(tri[flat])
    floors = np.asarray(floor_faces)
    if floors.size == 0:
        raise ValueError("未找到可确认的水平室内地面。")
    if floor_z is None:
        # Anchor the ground band to upward support, not the underside of a foundation.
        levels = floors[:, :, 2].mean(axis=1)
        floors = floors[levels <= levels.min() + 0.5 + 1e-6]
    angles = np.arange(16) * (2 * np.pi / 16)
    offsets = np.vstack(([0, 0], np.column_stack((np.cos(angles), np.sin(angles))) * radius))
    center = (low[:2] + high[:2]) / 2
    candidates = [center, np.zeros(2)]
    candidates += sorted(
        (
            np.array([x, y])
            for x in np.arange(low[0] + radius, high[0] - radius, 0.2)
            for y in np.arange(low[1] + radius, high[1] - radius, 0.2)
        ),
        key=lambda p: float(np.linalg.norm(p - center)),
    )
    for xy in candidates[:5000]:
        levels = []
        for point in xy + offsets:
            hits = floors[_inside_xy(point, floors)]
            if hits.size == 0:
                break
            levels.append(float(hits[:, :, 2].mean(axis=1).max()))
        if len(levels) != len(offsets) or np.ptp(levels) > 0.005:
            continue
        z = max(levels)
        clear = True
        for item in meshes:
            bounds = item.mesh.bounds
            outside_xy = np.any(xy + radius < bounds[0, :2]) or np.any(xy - radius > bounds[1, :2])
            outside_height = bounds[1, 2] < z + 0.06 or bounds[0, 2] > z + height
            if outside_xy or outside_height:
                continue
            if (
                item.mesh.is_watertight
                and item.mesh.contains([[*xy, z + h] for h in (0.1, height / 2, height - 0.01)]).any()
            ):
                clear = False
                break
            tri = item.mesh.triangles
            tri = tri[(tri[:, :, 2].max(axis=1) > z + 0.06) & (tri[:, :, 2].min(axis=1) < z + height)]
            if tri.size == 0:
                continue
            if _inside_xy(xy, tri).any():
                clear = False
                break
            # Distance to projected triangle edges also handles vertical/degenerate faces.
            for i in range(3):
                a, b = tri[:, i, :2], tri[:, (i + 1) % 3, :2]
                edge = b - a
                t = np.clip(np.sum((xy - a) * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-20), 0, 1)
                if np.any(np.linalg.norm(a + t[:, None] * edge - xy, axis=1) < radius):
                    clear = False
                    break
            if not clear:
                break
        if clear:
            return float(xy[0]), float(xy[1]), z
    raise ValueError("未找到满足全身间距和地面支撑的出生位置。")
