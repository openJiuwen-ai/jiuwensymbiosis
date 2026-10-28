"""USD lighting translated to HDRI and renderer-only emissive rectangles.

Nyx 0.1.7 rejects native Area lights. Emissive meshes retain emitter geometry
and radiance, but do not provide the native area-light sampling strategy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

PROFILE_PATH = Path(__file__).with_name("data") / "genesis_lighting.yaml"


@dataclass(frozen=True)
class RectangleEmitter:
    path: str
    corners_m: tuple[tuple[float, float, float], ...]
    radiance: tuple[float, float, float]
    area_m2: float
    fixture_path: str | None = None
    source_area_m2: float | None = None
    center_shift_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SourceLighting:
    hdr: str
    rotation_rad: float
    multiplier: float
    tint: tuple[float, float, float]
    rectangles: tuple[RectangleEmitter, ...]
    exposure_ev: float
    emitter_gain: float = 1.0

    def summary(self) -> dict:
        """Public scene state, without local asset paths or bulky vertex arrays."""
        return {
            "mode": "usd_hdr_and_emissive_rectangles",
            "rectangle_count": len(self.rectangles),
            "fitted_rectangle_count": sum(r.fixture_path is not None for r in self.rectangles),
            "emitter_gain": self.emitter_gain,
            "source_hdr": True,
            "exposure_ev": self.exposure_ev,
            "native_area_supported": False,
            "inspection_light": False,
            "collision_geometry_added": 0,
        }

    def audit(self) -> dict:
        return {
            **asdict(self),
            **self.summary(),
        }


def load_source_lighting(source: Path, scene_id: str) -> SourceLighting | None:
    """Opt in only scenes whose lighting has been calibrated and tested."""
    import yaml

    with PROFILE_PATH.open() as stream:
        profiles = yaml.safe_load(stream)
    profile = profiles.get("scenes", {}).get(scene_id)
    if profile is None:
        return None
    if set(profile) != {"exposure_ev", "mode"} or profile["mode"] not in {
        "usd_emissive_rectangles",
        "usd_fitted_emitters",
    }:
        raise ValueError("不支持的源场景照明配置。")
    rig = read_source_lighting(source, exposure_ev=float(profile["exposure_ev"]))
    if profile["mode"] == "usd_fitted_emitters":
        if TYPE_CHECKING or __package__:
            from .fixture_lighting import fit_rectangles, read_diffusers
        else:
            from fixture_lighting import fit_rectangles, read_diffusers
        rig = fit_rectangles(rig, read_diffusers(source))
    return rig


def read_source_lighting(source: Path, *, exposure_ev: float) -> SourceLighting:
    """Read composed UsdLux values, including parent transforms and units."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    if not math.isfinite(exposure_ev):
        raise ValueError("曝光必须为有限值。")
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError("无法读取灯光源场景。")
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    if up not in {"Y", "Z"} or not math.isfinite(scale) or scale <= 0:
        raise ValueError("不支持的灯光坐标系或单位。")

    def world_zup(v):
        x, y, z = map(float, v)
        return np.array((x, -z, y) if up == "Y" else (x, y, z))

    domes, rectangles = [], []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        light = UsdLux.LightAPI(prim)
        color = np.array(light.GetColorAttr().Get(), dtype=float)
        if light.GetEnableColorTemperatureAttr().Get():
            color *= np.array(UsdLux.BlackbodyTemperatureAsRgb(light.GetColorTemperatureAttr().Get()))
        power = float(light.GetIntensityAttr().Get()) * 2 ** float(light.GetExposureAttr().Get())
        invalid_color = not np.all(np.isfinite(color)) or np.any(color < 0)
        if invalid_color or not math.isfinite(power) or power < 0:
            raise ValueError(f"灯光数值无效：{prim.GetPath()}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if prim.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(prim)
            asset = dome.GetTextureFileAttr().Get()
            if not asset or not asset.resolvedPath or not Path(asset.resolvedPath).is_file():
                raise ValueError(f"找不到源 HDR：{prim.GetPath()}")
            if str(dome.GetTextureFormatAttr().Get()) not in {"latlong", "automatic"}:
                raise ValueError("当前源照明仅支持经纬 HDR。")
            # Nyx exposes yaw only. Reject tilt instead of silently dropping it.
            axis = (0, 1, 0) if up == "Y" else (0, 0, 1)
            vertical = world_zup(transform.TransformDir(Gf.Vec3d(*axis)))
            if not np.allclose(vertical / np.linalg.norm(vertical), (0, 0, 1), atol=1e-6):
                raise ValueError("当前 Nyx HDR 接口不能保留倾斜的 DomeLight。")
            x_axis = world_zup(transform.TransformDir(Gf.Vec3d(1, 0, 0)))
            domes.append((asset.resolvedPath, math.atan2(x_axis[1], x_axis[0]), power, tuple(color)))
        elif prim.IsA(UsdLux.RectLight):
            rect = UsdLux.RectLight(prim)
            if rect.GetTextureFileAttr().Get():
                raise ValueError("当前矩形发光几何不支持灯光纹理。")
            w, h = float(rect.GetWidthAttr().Get()), float(rect.GetHeightAttr().Get())
            if not math.isfinite(w + h) or min(w, h) <= 0:
                raise ValueError("矩形灯尺寸必须为正数。")
            # USD emits along local -Z: clockwise vertices seen from +Z.
            local = [(-w / 2, -h / 2, 0), (-w / 2, h / 2, 0), (w / 2, h / 2, 0), (w / 2, -h / 2, 0)]
            corners = [world_zup(transform.Transform(Gf.Vec3d(*v))) * scale for v in local]
            area = float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[3] - corners[0])))
            if not math.isfinite(area) or area <= 1e-12:
                raise ValueError("矩形灯变换后的面积无效。")
            # normalize=True divides by world area to keep total power constant.
            # https://openusd.org/25.11/user_guides/schemas/usdLux/LightAPI.html#inputs-normalize
            radiance = color * power / (area if light.GetNormalizeAttr().Get() else 1)
            rectangles.append(
                RectangleEmitter(str(prim.GetPath()), tuple(tuple(v) for v in corners), tuple(radiance), area)
            )
        else:
            raise ValueError(f"当前源照明未实现 {prim.GetTypeName()}，不能静默丢弃灯光。")
    if len(domes) != 1:
        raise ValueError("当前源照明要求恰好一个 DomeLight。")
    return SourceLighting(*domes[0], tuple(rectangles), exposure_ev)


def make_environment(rig: SourceLighting):
    import gs_nyx.nyx_py_sdk as nps

    env = nps.EnvironmentMapAsset()
    env.texture = rig.hdr
    env.layout = nps.EEnvMapLayout.LongLat
    env.rotation = rig.rotation_rad
    env.multiplier = rig.multiplier
    env.tint = nps.float3(*rig.tint)
    return env


def append_emitters(scene_path: Path, directory: Path, rig: SourceLighting) -> None:
    """Add two visual triangles per lamp; never touch the physical world."""
    import gs_nyx.nyx_py_sdk as nps

    asset = nps.SceneAsset()
    if not nps.import_scene_file(str(scene_path), asset):
        raise RuntimeError("无法读取 Nyx 灯光场景。")
    start = asset.instance_size()
    asset.instance_resize(start + len(rig.rectangles))
    for index, rect in enumerate(rig.rectangles):
        mesh = directory / f"source-emitter-{index}.obj"
        # World Z-up -> Nyx Y-up, preserving winding (proper rotation).
        vertices = "".join(f"v {x:.12g} {z:.12g} {-y:.12g}\n" for x, y, z in rect.corners_m)
        mesh.write_text(vertices + "f 1 2 3\nf 1 3 4\n")
        instance = nps.InstanceAsset()
        instance.type = nps.EInstanceType.Mesh
        instance.uuid = nps.generate_uuid()
        instance.mesh_uri = str(mesh.resolve())
        instance.position = nps.float3(0, 0, 0)
        instance.rotation = nps.quaternion(0, 0, 0, 1)
        instance.scale = nps.float3(1, 1, 1)
        instance.convertAxis = False
        instance.enabled = True
        material = nps.MaterialAsset()
        material.albedoColor = nps.float4(0, 0, 0, 1)
        material.emission = nps.float3(*rect.radiance)
        material.sidedness = nps.EMaterialSidedness.Single
        material.set_properties(
            nps.EMaterialProperty.Emission | nps.EMaterialProperty.AlbedoColor | nps.EMaterialProperty.Sidedness
        )
        instance.matOverride = material
        asset.set_instance(start + index, instance)
    nps.export_scene_file(str(scene_path), asset)
    (directory / "lighting-audit.json").write_text(json.dumps(rig.audit(), indent=2))
