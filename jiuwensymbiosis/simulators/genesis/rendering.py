"""Rebuildable Nyx rendering, independent of the Genesis physics scene."""

from __future__ import annotations

import math
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .material_repair import MaterialRules


class SceneRenderer:
    """Own the Nyx renderer; caller owns the already-built Genesis scene."""

    def __init__(
        self,
        scene,
        source: Path | None,
        directory: Path,
        settings,
        pose: dict,
        env,
        *,
        exposure: float = 0,
        light_pose: dict | None = None,
        visual_urdfs: dict[Path, Path] | None = None,
        source_lighting=None,
        robot=None,
        cameras=None,
        startup=None,
        material_rules: MaterialRules | None = None,
        material_directory: Path | None = None,
    ):
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as nps

        if TYPE_CHECKING or __package__:
            from .appearance import TONE_MAPPERS, load_appearance_profile
        else:
            from appearance import TONE_MAPPERS, load_appearance_profile
        from gs_nyx_plugin.nyx_renderer import NyxPyRenderer
        from gs_nyx_plugin.nyx_scene_exporter import NyxSceneExporter

        self._scene = scene
        self._renderer: Any = None
        fov = 60 if source is not None else 42
        self.fov_y = fov
        self.width, self.height = settings.width, settings.height
        self.entity_pairs: list = []
        self.cameras = cameras or {}
        self._links = {name: robot.get_link(camera.link_name) for name, camera in self.cameras.items()}
        self._grids = {}
        self._rectified_grids = {}
        self._projection = {}
        self._camera_indices = {"observer": 0}
        self.last_camera: dict = {}
        self.render_counts = dict.fromkeys(("observer", *self.cameras), 0)
        appearance_profile = load_appearance_profile(settings.scene_id)
        base_exposure_ev = source_lighting.exposure_ev if source_lighting else exposure
        color_management = appearance_profile.color_management.effective_parameters(base_exposure_ev)
        profile = nps.CameraProfileAsset()
        profile.set_fov_y(math.radians(fov))
        profile.colorManagement.toneMapper = getattr(nps.EToneMapper, TONE_MAPPERS[color_management["tone_mapper"]])
        profile.sensor.exposure.exposureCompensation = color_management["effective_exposure_ev"]
        profile_path = str(directory / "usd-camera.json")
        nps.export_camera_profile_file(profile_path, profile)
        camera = {
            "res": (settings.width, settings.height),
            "pos": pose["pos"],
            "lookat": pose["lookat"],
            "up": pose["up"],
            "fov": fov,
            "spp": settings.spp,
            "denoise": True,
            "camera_profile": profile_path,
        }
        camera_defs = [camera]
        for name, sensor in self.cameras.items():
            from importlib.metadata import version

            import torch
            from rgbd import effective_focal, rectified_grid

            size, sensor_fov, grid = sensor.render_map()
            nyx_version, plugin_version = version("gs-nyx"), version("gs-nyx-plugin")
            actual_focal = effective_focal(size, sensor_fov, nyx_version, plugin_version)
            nominal_focal = size[1] / (2 * math.tan(math.radians(sensor_fov) / 2))
            grid = grid * (actual_focal / nominal_focal)
            self._rectified_grids[name] = torch.as_tensor(
                rectified_grid(sensor, size, actual_focal), device="cuda"
            ).unsqueeze(0)
            self._projection[name] = {
                "nyx_version": nyx_version,
                "plugin_version": plugin_version,
                "native_focal_px": actual_focal,
                "projection_correction": "validated_fov_radians_float16",
            }
            sensor_pose = sensor.pose(self._links[name])
            profile.set_fov_y(math.radians(sensor_fov))
            sensor_profile = str(directory / f"camera-{len(camera_defs)}.json")
            nps.export_camera_profile_file(sensor_profile, profile)
            self._camera_indices[name] = len(camera_defs)
            self._grids[name] = torch.as_tensor(grid, device="cuda").unsqueeze(0)
            camera_defs.append(
                {
                    **camera,
                    "res": size,
                    **sensor_pose,
                    "fov": sensor_fov,
                    "near": sensor.clipping[0],
                    "far": sensor.clipping[1],
                    "camera_profile": sensor_profile,
                }
            )
        path = str(directory / "usd-scene.json")
        # USD mesh import does not translate every UsdLux light. An explicitly
        # reported inspection light makes enclosed, non-emissive rooms visible.
        # It is a fixed preview light, not a claim to reproduce the source rig.
        fill = nps.LightAsset()
        fill.type = nps.ELightType.Point
        fill.point_position = nps.float3_z_up_to_y_up_a(nps.float3(*(light_pose or pose)["pos"]))
        fill.point_range = 100.0
        fill.unit = nps.ELightUnit.Lumen
        fill.color = nps.float3(1, 1, 1)
        fill.intensity = 30.0 * 2 ** (-exposure)
        fill.shadow = True
        exporter = NyxSceneExporter(
            scene,
            str(directory),
            cameras=camera_defs,
            env_maps=[env],
            lights=[fill] if source is not None and source_lighting is None else [],
        )
        with startup.phase("nyx_scene_export") if startup and not startup.finished else nullcontext():
            exporter.export_to_file(path)
        if visual_urdfs:
            # Change only the renderer's asset, retaining physical URDF link names
            # and UUIDs so Nyx's existing pose updates and picking still apply.
            asset = nps.SceneAsset()
            if not nps.import_scene_file(path, asset):
                raise RuntimeError("无法读取 Nyx 机器人场景。")
            for index in range(asset.instance_size()):
                instance = asset.get_instance(index)
                if instance.type == nps.EInstanceType.SubScene:
                    replacement = visual_urdfs.get(Path(instance.subscene_uri))
                    if replacement is not None:
                        instance.subscene_uri = str(replacement)
                        asset.set_instance(index, instance)
            nps.export_scene_file(path, asset)
        # Nyx 0.1.9 exposes this bridge only as an internal attribute.
        entity_pairs = getattr(exporter, "_entity_uuid_pairs", None)
        if entity_pairs is None:
            raise RuntimeError("Nyx exporter bridge unavailable; requires gs-nyx-plugin 0.1.9")
        self.entity_pairs = entity_pairs
        self.appearance = {"color_management": color_management}
        if source is not None:
            # Without rules prepare_materials converts nothing and returns the
            # source unchanged, so skip it: the default preview path must not
            # require pxr just to report what it would leave untouched.
            if material_rules:
                source = self._repair_materials(source, directory, material_rules, material_directory)
            asset = nps.SceneAsset()
            if not nps.import_scene_file(path, asset):
                raise RuntimeError("无法创建 Nyx 背景场景。")
            background_index = asset.instance_size()
            asset.instance_resize(background_index + 1)
            instance = asset.get_instance(background_index)
            instance.type = nps.EInstanceType.Mesh
            instance.mesh_uri = str(source)
            instance.uuid = nps.generate_uuid()
            instance.position = nps.float3(0, 0, 0)
            instance.rotation = nps.quaternion(0, 0, 0, 1)
            instance.scale = nps.float3(1, 1, 1)
            instance.convertAxis = False  # USD carries its own upAxis/metersPerUnit.
            instance.enabled = True
            asset.set_instance(background_index, instance)
            nps.export_scene_file(path, asset)
        if source_lighting is not None:
            if TYPE_CHECKING or __package__:
                from .source_lighting import append_emitters
            else:
                from source_lighting import append_emitters

            append_emitters(Path(path), directory, source_lighting)
        self._renderer = NyxPyRenderer(
            scene,
            path,
            max(camera["res"][0] for camera in camera_defs),
            max(camera["res"][1] for camera in camera_defs),
            n_envs=1,
            camera_defs=camera_defs,
            render_mode=(
                npr.ERenderMode.RefPathTracer if settings.render_mode == "reference" else npr.ERenderMode.FastPathTracer
            ),
        )
        try:
            with startup.phase("nyx_renderer_build") if startup and not startup.finished else nullcontext():
                self._renderer.build(self.entity_pairs)
            self.update_camera_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
            for name in self.cameras:
                self._update_sensor(name)
        except BaseException:
            self.close()
            raise

    def _repair_materials(
        self, source: Path, directory: Path, rules: MaterialRules, material_directory: Path | None
    ) -> Path:
        """Return the material-repaired source layer and record a compact summary."""
        if TYPE_CHECKING or __package__:
            from .material_repair import prepare_materials
        else:
            from material_repair import prepare_materials

        # Baked textures are content-addressed (source digest + parameters):
        # a persistent material_directory lets renderer rebuilds reuse them
        # instead of re-baking a full texture session on every resolution change.
        materials_dir = Path(material_directory) if material_directory is not None else directory / "materials"
        repaired, repair = prepare_materials(source, materials_dir, rules=rules)
        # RuntimeScene.info crosses the worker IPC boundary. Keep per-prim
        # paths and bake manifests on disk instead of sending megabytes.
        self.appearance["material_repair"] = {
            "converted": repair["converted"],
            "textures": repair["textures"],
            "remaining": len(repair["remaining"]),
            "issues": dict(Counter(item["kind"] for item in repair["issues"])),
            "baked_textures": len(repair["baked_textures"]),
            "report_path": str(materials_dir / "material-repair.json") if repair["converted"] else None,
        }
        return repaired

    def update_camera_pose(self, *, pos, lookat, up):
        self._renderer.update_camera_tensor(0, pos, lookat, up)

    def pick(self, u: float, v: float):
        hit = self._renderer.pick_pixel(
            self.entity_pairs, 0, min(self.width - 1, int(u * self.width)), min(self.height - 1, int(v * self.height))
        )
        if hit is None:
            return None
        entity, node, point = hit
        return entity, node, [float(point.x), float(point.y), float(point.z)]

    def _update_sensor(self, name: str) -> None:
        sensor = self.cameras[name]
        pose = sensor.pose(self._links[name])
        self._renderer.update_camera_tensor(self._camera_indices[name], pose["pos"], pose["lookat"], pose["up"])
        self.last_camera = {**sensor.metadata(), **pose}

    def read_rgb(self, camera_name: str = "observer", *, rectified: bool = False):
        if camera_name not in self._camera_indices:
            raise ValueError(f"当前场景没有相机：{camera_name}")
        if camera_name != "observer":
            self._update_sensor(camera_name)
        # Match NyxCameraSensor: physical link poses and visual geometry poses
        # are separate buffers. The direct renderer consumes the latter.
        self._scene.visualizer.update_visual_states()
        self._renderer.update_scene()
        self._renderer.update()
        # The plugin's render() batches every camera. The pinned native bridge
        # applies the same uploaded poses but renders only this one camera.
        native = getattr(self._renderer, "_renderer", None)
        if native is None:
            raise RuntimeError("Nyx native renderer unavailable; requires a built gs-nyx-plugin 0.1.9 renderer")
        rgb = native.render_frame(self._camera_indices[camera_name])
        self.render_counts[camera_name] += 1
        if camera_name == "observer":
            return rgb
        from torch.nn.functional import grid_sample

        image = rgb.permute(2, 0, 1).unsqueeze(0).float()
        grid = self._rectified_grids[camera_name] if rectified else self._grids[camera_name]
        warped = grid_sample(image, grid, align_corners=False)
        self.last_camera.update(self._projection[camera_name])
        if rectified:
            self.last_camera.update(D=[0.0] * 5, distortion_model="rectified_pinhole")
        return warped[0].permute(1, 2, 0).round().clamp_(0, 255).to(rgb.dtype).contiguous()

    def read_rgbd(self, camera_name: str, *, stride: int = 8, check=lambda: None):
        """One state, one calibrated rectified RGB image and optical-Z depth map."""
        import numpy as np
        from rgbd import sample_depth

        if camera_name not in self.cameras:
            raise ValueError("RGB-D requires a calibrated robot camera")
        check()
        rgb = self.read_rgb(camera_name, rectified=True).cpu().numpy().copy()
        # gs-nyx-plugin 0.1.9 has no public accessor for the native depth ray API.
        # Keep this bridge explicit; an incompatible SDK must fail, not invent depth.
        # pylint: disable=protected-access
        depth = sample_depth(
            self._renderer._renderer,  # noqa: SLF001 -- pinned native bridge
            self.cameras[camera_name],
            np.asarray(self.last_camera["T_world_camera"]),
            stride=stride,
            check=check,
        )
        # pylint: enable=protected-access
        self.last_camera.update(depth_type="optical_z", depth_unit="m", depth_sampling_stride=stride)
        return rgb, depth

    def close(self):
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None
        self._grids.clear()
        self._rectified_grids.clear()


class RasterRenderer:
    """Observer-only OpenGL rendering for the portable Genesis demo."""

    def __init__(self, scene, pose: dict, *, fov_y: float = 42):
        if TYPE_CHECKING or __package__:
            from .camera import PREVIEW_RESOLUTIONS
        else:
            from camera import PREVIEW_RESOLUTIONS

        self.fov_y = fov_y
        self.appearance = {"renderer": "Genesis Rasterizer", "source_material_fidelity": False}
        self._observers = {
            size: scene.add_camera(
                res=size, fov=fov_y, GUI=False, **{key: pose[key] for key in ("pos", "lookat", "up")}
            )
            for size in PREVIEW_RESOLUTIONS
        }
        self._camera = self._observers[(960, 540)]
        self.render_counts = {"observer": 0}
        self.last_camera: dict = {}

    def configure(self, settings, pose: dict) -> None:
        camera = self._observers.get((settings.width, settings.height))
        if camera is None:
            raise ValueError("预览支持 540p、720p、1080p。")
        self._camera = camera
        self.update_camera_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])

    def update_camera_pose(self, *, pos, lookat, up):
        self._camera.set_pose(pos=pos, lookat=lookat, up=up)

    def read_rgb(self, camera_name: str = "observer"):
        import torch

        if camera_name != "observer":
            raise ValueError("Mac Genesis 演示仅提供 observer 观察视角。")
        rgb, *_ = self._camera.render(rgb=True)
        self.render_counts[camera_name] += 1
        return torch.from_numpy(rgb.copy())

    @staticmethod
    def pick(u: float, v: float):
        raise ValueError("macOS 最小预览暂不支持表面点选；可使用相机环绕、平移和缩放。")

    def close(self):
        # Cameras belong to the scene and remain available across resizing.
        pass
