"""Rebuildable Nyx rendering, independent of the Genesis physics scene."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any


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
            import torch

            size, sensor_fov, grid = sensor.render_map()
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
            # Keep authored materials and texture references for the renderer to interpret.
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
            self._renderer.build(self.entity_pairs)
            self.update_camera_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
            for name in self.cameras:
                self._update_sensor(name)
        except BaseException:
            self.close()
            raise

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

    def read_rgb(self, camera_name: str = "observer"):
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
        warped = grid_sample(image, self._grids[camera_name], align_corners=False)
        return warped[0].permute(1, 2, 0).round().clamp_(0, 255).to(rgb.dtype).contiguous()

    def close(self):
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None
        self._grids.clear()
