"""Fit visible emitters to authored luminaire diffusers, without editing physics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Diffuser:
    path: str
    vertices: np.ndarray


def read_diffusers(source: Path) -> list[Diffuser]:
    """Find dominant downward planar faces on emissive meshes (Z-up scenes)."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(source))
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise ValueError("灯具贴合目前只支持 Z-up 场景。")
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        shader, _, _ = material.ComputeSurfaceSource("mdl") if material else (None, None, None)
        if not shader or not shader.GetInput("enable_emission") or not shader.GetInput("enable_emission").Get():
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3] * scale
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        triangles = np.array(
            [(s, s + j, s + j + 1) for s, n in zip(starts, counts, strict=True) for j in range(1, n - 1)]
        )
        xyz = world[indices[triangles]]
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            cross = -cross
        area = np.linalg.norm(cross, axis=1) / 2
        down = cross[:, 2] < -1.998 * area
        down &= area > 1e-10
        if not down.any():
            continue
        # 0.1 mm clustering tolerates the small authored transform tilt.
        heights = np.round(xyz.mean(axis=1)[:, 2], 4)
        z = max(np.unique(heights[down]), key=lambda h: area[down & (heights == h)].sum())
        vertices = xyz[down & (heights == z)].reshape(-1, 3)
        result.append(Diffuser(str(prim.GetPath()), vertices))
    return result


def fit_rectangles(rig, diffusers: list[Diffuser], *, inset_m: float = 0.001, offset_m: float = 0.0002):
    """Fit each face, keeping relative lamp power and avoiding native bright speckles.

    This changes the emitting shape to match the visible fixture, not the HDR,
    exposure, physics mesh, or source asset. Ambiguous matches fail explicitly.
    Area-integrated RGB radiance is conserved until a common attenuation is
    needed to keep peak emission below 60000. The attenuation is audited.
    """
    from scipy.spatial import ConvexHull

    if inset_m < 0 or not 0 < offset_m <= 0.001:
        raise ValueError("灯面边距或贴合间隙无效。")
    used, fitted = set(), []
    for rect in rig.rectangles:
        old = np.asarray(rect.corners_m)
        center = old.mean(axis=0)
        candidates = sorted(
            ((float(np.linalg.norm(d.vertices.mean(axis=0) - center)), d) for d in diffusers), key=lambda x: x[0]
        )
        if not candidates or candidates[0][0] > 0.15:
            raise ValueError(f"找不到附近的发光灯具：{rect.path}")
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.02:
            raise ValueError(f"灯具匹配不唯一：{rect.path}")
        _, diffuser = candidates[0]
        if diffuser.path in used:
            raise ValueError("多个灯面不能重复匹配同一个灯具。")
        used.add(diffuser.path)
        u, v = old[1] - old[0], old[3] - old[0]
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        if abs(np.cross(u, v)[2] + 1) > 1e-5:
            raise ValueError("仅支持向下的水平矩形灯。")
        points = diffuser.vertices
        plane, _, rank, _ = np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)
        if rank != 3 or np.max(np.abs(np.c_[points[:, :2], np.ones(len(points))] @ plane - points[:, 2])) > 0.0001:
            raise ValueError("灯具发光面不平整，不能用矩形替代。")
        pu, pv = points @ u, points @ v
        lo, hi = np.array([pu.min(), pv.min()]) + inset_m, np.array([pu.max(), pv.max()]) - inset_m
        if np.any(hi <= lo):
            raise ValueError("灯具开口小于贴合边距。")
        # Keep rectangular emitter corners inside bevelled diffuser boundaries.
        hull = ConvexHull(np.column_stack((pu, pv))).equations
        midpoint = (lo + hi) / 2
        box = np.array([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])

        def inside(scale, midpoint=midpoint, box=box, hull=hull):
            xy = midpoint + (box - midpoint) * scale
            return np.all(xy @ hull[:, :2].T + hull[:, 2] <= 1e-10)

        if not inside(1):
            lower, upper = 0.0, 1.0
            for _ in range(30):
                middle = (lower + upper) / 2
                if inside(middle):
                    lower = middle
                else:
                    upper = middle
            box = midpoint + (box - midpoint) * lower * 0.99999
        corners = np.array([a * u + b * v for a, b in box])
        corners[:, 2] = np.c_[corners[:, :2], np.ones(4)] @ plane - offset_m
        area = float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[3] - corners[0])))
        if not 0.25 < area / rect.area_m2 < 4:
            raise ValueError("灯具和源矩形灯面积差异过大。")
        fitted.append(
            replace(
                rect,
                corners_m=tuple(map(tuple, corners)),
                area_m2=area,
                radiance=tuple(np.asarray(rect.radiance) * rect.area_m2 / area),
                fixture_path=diffuser.path,
                source_area_m2=rect.area_m2,
                center_shift_m=tuple(corners.mean(axis=0) - center),
            )
        )
    # Full-scene GPU controls: moving the face did not remove coloured pixels;
    # reducing radiance did. This is consistent with half-float overflow, but
    # native internals are unavailable. Keep headroom below the half-float max.
    peak = max((max(rect.radiance) for rect in fitted), default=0)
    gain = min(1.0, 60000.0 / peak) if peak > 0 else 1.0
    fitted = [replace(rect, radiance=tuple(np.asarray(rect.radiance) * gain)) for rect in fitted]
    return replace(rig, rectangles=tuple(fitted), emitter_gain=rig.emitter_gain * gain)
