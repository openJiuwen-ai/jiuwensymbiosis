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

DEFAULT_EXPOSURE_EV = -8.0


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
    environment_kind: str = "hdr_texture"
    fitting_decisions: tuple[dict, ...] = ()
    native_lights: tuple[dict, ...] = ()
    approximations: tuple[dict, ...] = ()

    def summary(self) -> dict:
        """Public scene state, without local asset paths or bulky vertex arrays."""
        return {
            "mode": "usd_hdr_and_emissive_rectangles",
            "selection": "asset_features",
            "fitting_preserved_count": sum(d["action"] == "preserve_source" for d in self.fitting_decisions),
            "rectangle_count": len(self.rectangles),
            "point_count": sum(e["type"] == "point" for e in self.native_lights),
            "directional_count": sum(e["type"] == "directional" for e in self.native_lights),
            "approximation_count": len(self.approximations),
            "fitted_rectangle_count": sum(r.fixture_path is not None for r in self.rectangles),
            "emitter_gain": self.emitter_gain,
            "source_hdr": self.environment_kind == "hdr_texture",
            "environment_kind": self.environment_kind,
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


def load_source_lighting(source: Path, *, directory: Path, exposure_ev: float = DEFAULT_EXPOSURE_EV) -> SourceLighting:
    """Apply the same source-light and material-driven fitting policy to every USD."""
    if TYPE_CHECKING or __package__:
        from .fixture_lighting import auto_fit_rectangles, read_diffusers
    else:
        from fixture_lighting import auto_fit_rectangles, read_diffusers
    rig = read_source_lighting(source, exposure_ev=exposure_ev, directory=directory)
    return auto_fit_rectangles(rig, read_diffusers(source)) if rig.rectangles else rig


def read_source_lighting(source: Path, *, exposure_ev: float, directory: Path | None = None) -> SourceLighting:
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

    domes, rectangles, native_lights, approximations = [], [], [], []
    environment_kind = "hdr_texture"
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
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
        if float(light.GetSpecularAttr().Get()) != 1 or float(light.GetDiffuseAttr().Get()) != 1:
            approximations.append(
                {"path": str(prim.GetPath()), "reason": "Nyx cannot preserve separate diffuse/specular multipliers"}
            )

        if prim.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(prim)
            asset = dome.GetTextureFileAttr().Get()
            if not asset or not asset.path:
                if directory is None:
                    raise ValueError("Constant DomeLight requires a runtime output directory")
                import cv2

                environment_kind = "constant_color"
                directory.mkdir(parents=True, exist_ok=True)
                hdr = directory / "constant-environment.hdr"
                if not cv2.imwrite(str(hdr), np.ones((8, 16, 3), dtype=np.float32)):
                    raise RuntimeError("Cannot write constant environment HDR")
                hdr_path = str(hdr.resolve())
            elif not asset.resolvedPath or not Path(asset.resolvedPath).is_file():
                raise ValueError(f"找不到源 HDR：{prim.GetPath()}")
            else:
                hdr_path = asset.resolvedPath
            if str(dome.GetTextureFormatAttr().Get()) not in {"latlong", "automatic"}:
                raise ValueError("当前源照明仅支持经纬 HDR。")
            # Nyx exposes yaw only. Reject tilt instead of silently dropping it.
            axis = (0, 1, 0) if up == "Y" else (0, 0, 1)
            vertical = world_zup(transform.TransformDir(Gf.Vec3d(*axis)))
            if not np.allclose(vertical / np.linalg.norm(vertical), (0, 0, 1), atol=1e-6):
                raise ValueError("当前 Nyx HDR 接口不能保留倾斜的 DomeLight。")
            x_axis = world_zup(transform.TransformDir(Gf.Vec3d(1, 0, 0)))
            domes.append((hdr_path, math.atan2(x_axis[1], x_axis[0]), power, tuple(color)))
        elif prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.DistantLight):
            shaping = UsdLux.ShapingAPI(prim)
            if prim.HasAPI(UsdLux.ShapingAPI):
                # The native point/directional adapter cannot preserve angular shaping.
                for attr in (shaping.GetShapingFocusAttr(), shaping.GetShapingConeSoftnessAttr()):
                    if attr and attr.Get() not in (None, 0):
                        raise ValueError(f"Unsupported native-light shaping: {prim.GetPath()}")
                cone = shaping.GetShapingConeAngleAttr().Get()
                if cone is not None and cone < 180 or shaping.GetShapingIesFileAttr().Get():
                    raise ValueError(f"Unsupported native-light cone/IES: {prim.GetPath()}")
            entry = {"path": str(prim.GetPath()), "color": color.tolist()}
            if prim.IsA(UsdLux.SphereLight):
                sphere = UsdLux.SphereLight(prim)
                axes = np.array([world_zup(transform.TransformDir(Gf.Vec3d(*v))) for v in np.eye(3)])
                lengths = np.linalg.norm(axes, axis=1) * scale
                authored_radius = float(sphere.GetRadiusAttr().Get())
                radius = authored_radius * float(lengths.mean())
                if not math.isfinite(radius) or radius < 0 or np.any(lengths <= 0):
                    raise ValueError(f"Invalid sphere scale: {prim.GetPath()}")
                area = 4 * math.pi * radius**2
                nonuniform = not np.allclose(lengths, lengths[0], rtol=1e-4)
                if nonuniform:
                    import trimesh

                    mesh = trimesh.creation.icosphere(subdivisions=3, radius=authored_radius)
                    mesh.vertices = mesh.vertices @ axes * scale
                    area = float(mesh.area)
                    approximations.append(
                        {
                            "path": str(prim.GetPath()),
                            "reason": "nonuniform sphere area estimated from subdivided mesh; directional distribution averaged",
                        }
                    )
                # Cauchy mean projected area = surface area/4 for a convex emitter.
                point = bool(sphere.GetTreatAsPointAttr().Get())
                intensity = power / 4 if light.GetNormalizeAttr().Get() or point else power * area / 4
                entry.update(
                    type="point",
                    pos=(world_zup(transform.ExtractTranslation()) * scale).tolist(),
                    intensity=intensity,
                    radius_m=radius,
                    source_area_m2=area,
                )
                reason = "finite sphere approximated by point; no finite-size soft shadows"
            else:
                angle = float(UsdLux.DistantLight(prim).GetAngleAttr().Get())
                if not math.isfinite(angle) or not 0 <= angle <= 180:
                    raise ValueError(f"Invalid distant angular diameter: {prim.GetPath()}")
                direction = world_zup(transform.TransformDir(Gf.Vec3d(0, 0, -1)))
                direction /= np.linalg.norm(direction)
                factor = math.pi * math.sin(math.radians(angle) / 2) ** 2 if angle else 1
                entry.update(
                    type="directional",
                    direction=direction.tolist(),
                    intensity=power if light.GetNormalizeAttr().Get() else power * factor,
                    angle_deg=angle,
                )
                reason = "finite angular disk approximated by directional ray; soft shadows not retained"
            native_lights.append(entry)
            approximations.append({"path": str(prim.GetPath()), "reason": reason})
        elif prim.IsA(UsdLux.RectLight):
            rect = UsdLux.RectLight(prim)
            shaping = UsdLux.ShapingAPI(prim)
            cone_value = shaping.GetShapingConeAngleAttr().Get()
            softness_value = shaping.GetShapingConeSoftnessAttr().Get()
            focus_value = shaping.GetShapingFocusAttr().Get()
            cone = 90.0 if cone_value is None else float(cone_value)
            softness = 0.0 if softness_value is None else float(softness_value)
            focus = 0.0 if focus_value is None else float(focus_value)
            # USD RectLight emits on its front hemisphere only. A neutral
            # cutoff anywhere from 90 to 180 degrees clips none of that light.
            # Softened/focused/IES emission requires an angular shader, which
            # plain emissive geometry cannot preserve.
            if (
                not all(math.isfinite(value) for value in (cone, softness, focus))
                or not 90 <= cone <= 180
                or softness != 0
                or focus != 0
                or shaping.GetShapingIesFileAttr().Get()
            ):
                raise ValueError("Nyx emissive rectangles require unshaped hemispherical emission")
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
    if len(domes) > 1:
        raise ValueError("Multiple DomeLights require environment compositing; not supported")
    if not domes:
        if directory is None:
            raise ValueError("A scene without DomeLight requires a runtime output directory")
        import cv2

        directory.mkdir(parents=True, exist_ok=True)
        hdr = directory / "no-environment.hdr"
        if not cv2.imwrite(str(hdr), np.zeros((8, 16, 3), dtype=np.float32)):
            raise RuntimeError("Cannot write zero environment HDR")
        domes.append((str(hdr.resolve()), 0.0, 0.0, (1.0, 1.0, 1.0)))
        environment_kind = "none"

    return SourceLighting(
        *domes[0],
        tuple(rectangles),
        exposure_ev,
        environment_kind=environment_kind,
        native_lights=tuple(native_lights),
        approximations=tuple(approximations),
    )


def make_environment(rig: SourceLighting):
    import gs_nyx.nyx_py_sdk as nps

    env = nps.EnvironmentMapAsset()
    env.texture = rig.hdr
    env.layout = nps.EEnvMapLayout.LongLat
    env.rotation = math.degrees(rig.rotation_rad)
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
    start_light = asset.light_size()
    asset.light_resize(start_light + len(rig.native_lights))
    for index, entry in enumerate(rig.native_lights):
        light = nps.LightAsset()
        light.color = nps.float3(*entry["color"])
        light.intensity = entry["intensity"]
        light.shadow = True
        if entry["type"] == "point":
            light.type = nps.ELightType.Point
            light.unit = nps.ELightUnit.Candela
            light.point_position = nps.float3_z_up_to_y_up_a(nps.float3(*entry["pos"]))
            light.point_range = 100
        else:
            light.type = nps.ELightType.Directional
            light.unit = nps.ELightUnit.Lux
            light.directional_direction = nps.float3_z_up_to_y_up_a(nps.float3(*entry["direction"]))
        asset.set_light(start_light + index, light)
    nps.export_scene_file(str(scene_path), asset)
    (directory / "lighting-audit.json").write_text(json.dumps(rig.audit(), indent=2))
