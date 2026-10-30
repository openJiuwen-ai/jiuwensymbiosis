"""Worker-only ownership of physics and replaceable rendering resources."""

from __future__ import annotations

import tempfile
import uuid
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jiuwensymbiosis.simulators.genesis.robot_cameras import RobotCamera

from camera import OrbitCamera, PreviewSettings
from rendering import SceneRenderer
from scene_catalog import resolve_scene, validate_material_bundle
from selection import UsdPickIndex
from source_lighting import load_source_lighting, make_environment
from usd_scene import initial_camera, prepare_static_stage


class RuntimeScene:
    """One physical world; resizing never replaces this world or its state."""

    def __init__(self, settings: PreviewSettings, directory: Path, revision: int = 0, *, startup=None):
        self._startup = startup
        self.settings = settings
        self.scene: Any = None
        self.renderer: Any = None
        self.bodies: list[Any] = []
        self.robot: Any = None
        self.robot_asset: Any = None
        self.cameras: dict[str, RobotCamera] = {}
        self.indoor = None
        self.motion: Any = None
        self.asset: Any = None
        self.steps = self.cycle = 0
        self._directory = tempfile.TemporaryDirectory(prefix="scene-", dir=directory)
        self.directory = Path(self._directory.name)
        self._render_directory: tempfile.TemporaryDirectory | None = None
        self.source = None
        self.lighting = None
        self.selection: dict | None = None
        self.selected_entity = None
        self.pick_index: Any = None
        self.info = {"id": "demo", "mode": "dynamic_demo", "collision": True, "reset_interval_steps": 480}
        try:
            self.asset = resolve_scene(settings.scene_id) if settings.scene_id != "demo" else None
            mobile = settings.robot_mode == "moz1_wheels"
            self.robot_asset = resolve_scene(settings.robot_id) if settings.robot_id else self.asset
            if settings.robot_id and (self.asset is None or self.asset.asset.preview_mode != "static_usd"):
                raise ValueError("组合场景需要 USD 背景资产。")
            if mobile and (self.robot_asset is None or self.robot_asset.asset.preview_mode != "urdf"):
                raise ValueError("底盘物理模式需要选择 MOZ1 URDF 机器人。")
            if self.asset and self.asset.asset.preview_mode == "urdf":
                self.info = {
                    "id": self.asset.asset.id,
                    "mode": "robot_mobile" if mobile else "robot_preview",
                    "collision": mobile,
                    "reset_interval_steps": None,
                }
                camera = OrbitCamera(yaw=-0.7, pitch=0.15, distance=3.4, target=(0, 0, 0.95))
            elif settings.scene_id != "demo":
                self.source = self.directory / "background.usda"
                audit = prepare_static_stage(self.asset.preview_path, self.source)
                self.info = {
                    "id": settings.scene_id,
                    "mode": "robot_mobile" if mobile else "static_background",
                    "collision": mobile,
                    "reset_interval_steps": None,
                    **audit,
                }
                camera = OrbitCamera(
                    **initial_camera(self.asset.asset.category + "/" + self.asset.asset.name, audit["bounds_m"])
                )
            else:
                camera = OrbitCamera()
            self.info["instance_id"] = uuid.uuid4().hex
            self.initial_pose = replace(camera, revision=revision).snapshot()
            self.pose = self.initial_pose
            if self.source is not None:
                self.lighting = load_source_lighting(self.asset.preview_path, settings.scene_id)
                if self.lighting is not None:
                    self.info["lighting"] = self.lighting.summary()
                    self.info["preview_exposure_ev"] = self.lighting.exposure_ev
            with self._phase("physics_build"):
                self._build_physics()
            if self.robot is not None:
                from robot_cameras import load_robot_cameras

                self.cameras = load_robot_cameras(
                    self.robot_asset.paths["source"], {link.name for link in self.robot.entity.links}
                )
            self.frame_size(settings.camera_name)
            if mobile:
                from base_motion import RelativeMotion

                self.motion = RelativeMotion(self.robot)
            if self.robot is not None:
                self.info["robot_bounds_m"] = self.robot.bounds
                if self.source is None:
                    self.info["bounds_m"] = self.robot.bounds
                camera = camera.focus(
                    self.robot.bounds, fov_y=60 if self.source else 42, aspect=settings.width / settings.height
                )
                self.initial_pose = replace(camera, revision=revision).snapshot()
                self.pose = self.initial_pose
            self.build_renderer(settings)
        except BaseException:
            self.close()
            raise

    def _phase(self, name: str):
        return self._startup.phase(name) if self._startup is not None and not self._startup.finished else nullcontext()

    def _build_physics(self) -> None:
        import cv2
        import genesis as gs
        import gs_nyx.nyx_py_sdk as nps
        import numpy as np

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.settings.dt,
                substeps=4 if self.info["mode"] == "robot_mobile" else 1,
                gravity=(0, 0, 0) if self.info["mode"] == "robot_preview" else (0, 0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(enable_self_collision=self.info["mode"] != "robot_mobile"),
            show_viewer=False,
        )
        if self.info["mode"] == "robot_mobile":
            from omni_base import Moz1Base

            spawn = (0.0, 0.0, 0.0)
            if self.source is not None:
                from indoor import IndoorCollision, read_room_meshes, supported_spawn

                meshes = read_room_meshes(self.asset.preview_path)
                spawn = supported_spawn(meshes)
                self.indoor = IndoorCollision(self.scene, meshes, self.directory)
                self.info.update(
                    background_id=self.asset.asset.id,
                    robot_id=self.robot_asset.asset.id,
                    spawn_ground_m=spawn,
                    collision_meshes=len(meshes),
                    collision_method="fixed_mesh_convex_decomposition",
                )
            else:
                self.scene.add_entity(morph=gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.32, 0.36, 0.4)))
            self.robot = Moz1Base(
                self.scene,
                self.robot_asset.preview_path,
                self.directory,
                name=self.robot_asset.asset.name,
                render_urdf=validate_material_bundle(self.robot_asset),
                spawn=spawn,
            )
        elif self.info["mode"] == "robot_preview":
            from robot_preview import RobotPreview

            self.scene.add_entity(morph=gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.32, 0.36, 0.4)))
            self.robot = RobotPreview(
                self.scene,
                self.asset.preview_path,
                name=self.asset.asset.name,
                render_urdf=validate_material_bundle(self.asset),
                file_meshes_are_zup=self.asset.asset.options.get("file_meshes_are_zup", False),
            )
        elif self.source is None:
            self.scene.add_entity(morph=gs.morphs.Plane(), surface=gs.surfaces.Rough(color=(0.45, 0.49, 0.54)))
            self.scene.add_entity(
                morph=gs.morphs.Box(size=(1.6, 1.1, 0.12), pos=(0, 0, 0.06), fixed=True),
                surface=gs.surfaces.Rough(color=(0.12, 0.20, 0.27)),
            )
            self.bodies = [
                self.scene.add_entity(
                    morph=gs.morphs.Sphere(radius=0.13, pos=(-0.34, 0, 1.1)), surface=gs.surfaces.Gold()
                ),
                self.scene.add_entity(
                    morph=gs.morphs.Box(size=(0.22, 0.22, 0.22), pos=(0, 0, 1.45)),
                    surface=gs.surfaces.Rough(color=(0.1, 0.48, 0.78)),
                ),
                self.scene.add_entity(
                    morph=gs.morphs.Sphere(radius=0.11, pos=(0.34, 0.05, 1.8)), surface=gs.surfaces.Copper()
                ),
            ]
        with self._phase("scene_build"):
            self.scene.build(n_envs=1)
        if self.robot is not None:
            self.robot.build()
            if self.info["mode"] == "robot_mobile":
                for _ in range(round(2 / self.settings.dt)):
                    self.robot.before_step(self.settings.dt)
                    self.scene.step()
                    self.steps += 1
                if self.indoor is not None:
                    from base_motion import measure

                    observed = measure(self.robot)
                    expected = np.asarray(self.info["spawn_ground_m"]) + [0, 0, self.robot.layout.radius_m]
                    contacts = self.indoor.contacts(self.robot)
                    if (
                        np.linalg.norm(np.asarray(observed["position_m"]) - expected) > 0.02
                        or np.linalg.norm(observed["linear_velocity_m_s"]) > 0.02
                        or sum(item["force_n"] for item in contacts["surfaces"]) < 1
                    ):
                        raise ValueError("室内机器人出生后未能确认稳定地面支撑，请检查场景碰撞网格。")
                    self.info["spawn_support_verified"] = True
        if self.lighting is not None:
            self.env = make_environment(self.lighting)
            return
        u, v = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 256))
        light = 0.3 + 7 * np.exp(-(((u + 0.35) / 0.16) ** 8) - ((v + 0.3) / 0.3) ** 8)
        light += 2 * np.exp(-(((u - 0.5) / 0.25) ** 4) - ((v + 0.2) / 0.25) ** 4)
        hdr = self.directory / "studio.hdr"
        if not cv2.imwrite(str(hdr), np.repeat(light[..., None], 3, axis=2).astype(np.float32)):
            raise RuntimeError("无法写入 Nyx 环境光纹理。")
        self.env = nps.EnvironmentMapAsset()
        self.env.texture = str(hdr)
        self.env.layout = nps.EEnvMapLayout.LongLat

    def build_renderer(self, settings: PreviewSettings) -> None:
        """Re-export the current world, retaining its physical and camera state."""
        import torch

        # Physics construction leaves cached Torch scratch buffers. Nyx allocates
        # through a different GPU allocator and cannot reuse those reservations.
        torch.cuda.empty_cache()
        self._render_directory = tempfile.TemporaryDirectory(prefix="render-", dir=self.directory)
        self.renderer = SceneRenderer(
            self.scene,
            self.source,
            Path(self._render_directory.name),
            settings,
            self.pose,
            self.env,
            exposure=self.info.get("preview_exposure_ev", 0),
            light_pose=self.initial_pose,
            visual_urdfs={self.robot.path: self.robot.render_urdf} if self.robot and self.robot.render_urdf else None,
            source_lighting=self.lighting,
            robot=self.robot.entity if self.robot is not None else None,
            cameras=self.cameras,
            startup=self._startup,
        )
        self.info["appearance"] = self.renderer.appearance
        self.settings = settings

    def frame_size(self, camera_name: str) -> tuple[int, int]:
        if camera_name == "observer":
            return self.settings.width, self.settings.height
        if camera_name not in self.cameras:
            raise ValueError(f"当前机器人没有相机：{camera_name}")
        return self.cameras[camera_name].resolution

    def pick(self, u: float, v: float) -> dict | None:
        hit = self.renderer.pick(u, v)
        if hit is None:
            self.selection = None
        elif self.source is not None and (self.robot is None or hit[0] is not self.robot.entity):
            self.selected_entity = None
            if self.pick_index is None:
                self.pick_index = UsdPickIndex(self.source)
            self.selection = self.pick_index.resolve(hit[2])
        else:
            entity, _, point = hit
            if entity is None:
                raise ValueError("命中对象未在场景中注册。")
            box = (
                self.robot.bounds
                if self.robot is not None and entity is self.robot.entity
                else entity.get_AABB()[0].cpu().tolist()
            )
            self.selected_entity = entity
            labels = (
                ["地面", self.robot.name]
                if self.robot is not None
                else ["地面", "平台", "金色球", "蓝色方块", "铜色球"]
            )
            index = list(self.scene.entities).index(entity)
            self.selection = {
                "path": f"{self.settings.scene_id}/entity/{index}",
                "name": self.robot.name if self.robot is not None and entity is self.robot.entity else labels[index],
                "bounds_m": box,
                "hit_position_m": point,
                "method": "nyx_entity",
            }
        if self.selection is not None:
            self.selection["scene_instance_id"] = self.info["instance_id"]
        return self.selection

    def selection_snapshot(self) -> dict | None:
        if self.selection is not None and self.selected_entity is not None:
            self.selection["bounds_m"] = (
                self.robot.bounds
                if self.robot is not None and self.selected_entity is self.robot.entity
                else self.selected_entity.get_AABB()[0].cpu().tolist()
            )
        return self.selection

    def close_renderer(self) -> None:
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.renderer = None
            if self._render_directory is not None:
                self._render_directory.cleanup()
                self._render_directory = None

    def step(self) -> None:
        if self.bodies and self.steps % 480 == 0:
            self.cycle += 1
            for i, body in enumerate(self.bodies):
                body.set_pos((-0.34 + i * 0.34, 0.04 * (i % 2), 1.1 + i * 0.35), zero_velocity=True)
        if self.info["mode"] == "robot_mobile":
            self.motion.tick(self.settings.dt, self.steps)
            self.robot.before_step(self.settings.dt)
        self.scene.step()
        if self.info["mode"] == "robot_preview":
            self.robot.hold()
        self.steps += 1

    def update_camera(self, pose: dict) -> None:
        self.renderer.update_camera_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
        self.pose = pose

    def close(self) -> None:
        try:
            self.close_renderer()
        finally:
            if self.scene is not None:
                self.scene.destroy()
                self.scene = None
            self.bodies = []
            self._directory.cleanup()
