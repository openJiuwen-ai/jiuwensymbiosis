"""Rebuildable Nyx rendering, independent of the Genesis physics scene."""

from __future__ import annotations

import json
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
        presentation=None,
        robot=None,
        cameras=None,
        camera_pose_provider=None,
        startup=None,
        material_rules: MaterialRules | None = None,
        material_directory: Path | None = None,
    ):
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as nps

        if TYPE_CHECKING or __package__:
            from .appearance import TONE_MAPPERS, load_appearance_profile
            from .source_lighting import load_source_lighting, make_environment
        else:
            from appearance import TONE_MAPPERS, load_appearance_profile
            from source_lighting import load_source_lighting, make_environment
        from gs_nyx_plugin.nyx_renderer import NyxPyRenderer
        from gs_nyx_plugin.nyx_scene_exporter import NyxSceneExporter

        if source is not None and source_lighting is None:
            source_lighting = load_source_lighting(source, directory=directory)
            env = make_environment(source_lighting)
        self._scene = scene
        self._renderer: Any = None
        fov = presentation["fov_y"] if presentation is not None else (60 if source is not None else 42)
        self.fov_y = fov
        self.width, self.height = settings.width, settings.height
        self.entity_pairs: list = []
        self.cameras = cameras or {}
        self._camera_pose_provider = camera_pose_provider
        self._links = (
            {}
            if camera_pose_provider
            else {name: robot.get_link(camera.link_name) for name, camera in self.cameras.items()}
        )
        self._grids = {}
        self._rectified_grids = {}
        self._projection = {}
        self._camera_indices = {"observer": 0}
        self.last_camera: dict = {}
        self.render_counts = dict.fromkeys(("observer", *self.cameras), 0)
        appearance_profile = load_appearance_profile()
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
        if presentation is not None:
            camera.update(near=presentation["near"], far=presentation["far"])
        camera_defs = [camera]
        for name, sensor in self.cameras.items():
            from importlib.metadata import version

            import torch

            from .rgbd import effective_focal, rectified_grid

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
            sensor_pose = self._sensor_pose(name)
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
        exporter = NyxSceneExporter(
            scene,
            str(directory),
            cameras=camera_defs,
            env_maps=[env],
            lights=[],
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
        # The plugin exposes entity/UUID correspondence only as an internal attribute.
        entity_pairs = getattr(exporter, "_entity_uuid_pairs", None)
        if entity_pairs is None:
            raise RuntimeError("Nyx exporter bridge unavailable; requires gs-nyx-plugin 0.1.10")
        self.entity_pairs = entity_pairs
        self.appearance = {"color_management": color_management}
        if presentation is not None:
            self.appearance["source_camera"] = {
                "path": presentation["camera_path"],
                "fov_y": fov,
                "near": presentation["near"],
                "far": presentation["far"],
                "source_aspect": presentation["aspect"],
                "resolution": [settings.width, settings.height],
                "spp": settings.spp,
                "denoise": True,
                "quality_policy": presentation["quality_policy"],
            }
        if source is not None:
            with startup.phase("material_prepare") if startup and not startup.finished else nullcontext():
                source = self._repair_materials(
                    source, directory, material_rules, material_directory, validate=presentation is not None
                )
            if TYPE_CHECKING or __package__:
                from .native_glass import append_glass_instances, prepare_glass_layers
            else:
                from native_glass import append_glass_instances, prepare_glass_layers
            source, glass_groups, glass_audit = prepare_glass_layers(source, directory / "native-glass")
            self.appearance["native_glass"] = glass_audit
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
            append_glass_instances(asset, glass_groups)
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
            render_mode=(npr.ERenderMode.FastPathTracer),
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
        self,
        source: Path,
        directory: Path,
        rules: MaterialRules | None,
        material_directory: Path | None,
        *,
        validate: bool = False,
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
        materials_dir.mkdir(parents=True, exist_ok=True)
        (materials_dir / "material-repair.json").write_text(json.dumps(repair, ensure_ascii=False, indent=2))
        if validate:
            if TYPE_CHECKING or __package__:
                from .material_repair import require_supported_materials
            else:
                from material_repair import require_supported_materials
            self.appearance["material_validation"] = require_supported_materials(repaired, repair)
        # RuntimeScene.info crosses the worker IPC boundary. Keep per-prim
        # paths and bake manifests on disk instead of sending megabytes.
        self.appearance["material_repair"] = {
            "converted": repair["converted"],
            "textures": repair["textures"],
            "remaining": len(repair["remaining"]),
            "issues": dict(Counter(item["kind"] for item in repair["issues"])),
            "baked_textures": len(repair["baked_textures"]),
            "report_path": str(materials_dir / "material-repair.json"),
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

    def _sensor_pose(self, name):
        if self._camera_pose_provider is not None:
            return self._camera_pose_provider(name)
        return self.cameras[name].pose(self._links[name])

    def _update_sensor(self, name: str) -> None:
        sensor = self.cameras[name]
        pose = self._sensor_pose(name)
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
        rgb = self.render_camera(camera_name)
        return self.project_rgb(rgb, camera_name, rectified=rectified)

    def render_camera(self, camera_name: str):
        """Render only the selected camera into a borrowed RGB view (env 0).

        The pinned native SDK's single-camera output starts at byte zero, even
        for a nonzero camera index. The plugin's batch views have other offsets
        and must not be used here. Consume this view before the next render.
        """
        index = self._camera_indices[camera_name]
        plugin = self._renderer
        native = plugin._renderer  # noqa: SLF001 -- pinned SDK's native single-camera API
        plugin.update()
        # Single-camera output invalidates the plugin's packed batch views.
        plugin._batch_dirty = True  # noqa: SLF001 -- native output invalidates plugin batch cache
        if not native.render_frame(index):
            raise RuntimeError(f"Nyx 单相机渲染失败：{camera_name}")
        width, height = native.camera_resolution(index)
        size = width * height * 3
        output = plugin._color_output  # noqa: SLF001 -- pinned shared CameraColor allocation
        if output is None or len(output) < size:
            raise RuntimeError("Nyx 单相机 RGB 输出缓冲不足")
        self.render_counts[camera_name] += 1
        return output[:size].reshape(height, width, 3)

    def project_rgb(self, rgb, camera_name, *, rectified=False):
        """Apply the calibrated lens to an already-rendered, stable RGB view."""
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

    def read_depth(self, camera_name, *, stride=8, check=lambda: None, region=None):
        """Read depth from the same rendered state, before any next frame."""
        import numpy as np

        from .rgbd import sample_depth

        if camera_name not in self.cameras:
            raise ValueError("RGB-D requires a calibrated robot camera")
        # gs-nyx-plugin 0.1.9 has no public accessor for the native depth ray API.
        # Keep this bridge explicit; an incompatible SDK must fail, not invent depth.
        # pylint: disable=protected-access
        depth = sample_depth(
            self._renderer._renderer,  # noqa: SLF001 -- pinned native bridge
            self.cameras[camera_name],
            np.asarray(self.last_camera["T_world_camera"]),
            stride=stride,
            check=check,
            region=region,
        )
        # pylint: enable=protected-access
        self.last_camera.update(depth_type="optical_z", depth_unit="m", depth_sampling_stride=stride)
        if region is not None:
            self.last_camera.update(depth_sampling_region=region, depth_sampling_stride=None)
        return depth

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
            from ..common.camera import PREVIEW_RESOLUTIONS
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
