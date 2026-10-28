"""USD camera calibration and link mounts; rendering stays in the scene owner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RobotCamera:
    name: str
    link_name: str
    resolution: tuple[int, int]
    intrinsic: np.ndarray
    distortion: np.ndarray
    link_from_camera: np.ndarray
    clipping: tuple[float, float]

    def metadata(self) -> dict:
        """Calibration describes the delivered, distorted RGB image in pixels."""
        return {
            "name": self.name,
            "link": self.link_name,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "K": self.intrinsic.tolist(),
            "D": self.distortion.tolist(),
            "distortion_model": "opencv_pinhole",
            "optical_frame": "x_right_y_down_z_forward",
            "clipping_range_m": list(self.clipping),
        }

    def pose(self, link) -> dict:
        """Compose the current physical link pose with the authored USD mount."""
        from scipy.spatial.transform import Rotation

        world_from_link = np.eye(4)
        world_from_link[:3, :3] = Rotation.from_quat(link.get_quat()[0].cpu().numpy(), scalar_first=True).as_matrix()
        world_from_link[:3, 3] = link.get_pos()[0].cpu().numpy()
        world_from_camera = world_from_link @ self.link_from_camera
        position = world_from_camera[:3, 3]
        return {
            "pos": position.tolist(),
            "lookat": (position + world_from_camera[:3, 2]).tolist(),
            "up": (-world_from_camera[:3, 1]).tolist(),
            "T_world_camera": world_from_camera.tolist(),
        }

    def render_map(self) -> tuple[tuple[int, int], float, np.ndarray]:
        """Inverse distortion maps output pixels to an overscanned pinhole image.

        The virtual focal length never undersamples either calibrated axis.
        Nyx pixel centres are at half pixels; grid_sample uses align_corners=False.
        All map construction is done once, not on the physics/render hot path.
        """
        import cv2

        width, height = self.resolution
        pixels = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1).astype(np.float64)
        rays = cv2.undistortPointsIter(
            pixels.reshape(-1, 1, 2),
            self.intrinsic,
            self.distortion,
            np.eye(3),
            np.eye(3),
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
        ).reshape(height, width, 2)
        focal = max(self.intrinsic[0, 0], self.intrinsic[1, 1])
        extent = np.max(np.abs(rays), axis=(0, 1))
        if not np.isfinite(rays).all() or float(max(extent)) * focal + 2 > 2048:
            raise ValueError(f"相机 {self.name} 的标定超出渲染范围。")
        # A finite inverse can still be unconverged. Check the delivered pixels
        # with the forward model, without allocating a per-pixel Jacobian.
        x, y = rays[..., 0], rays[..., 1]
        r2 = x * x + y * y
        k1, k2, p1, p2, k3 = self.distortion
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        projected = np.stack(
            [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y],
            axis=-1,
        )
        projected = projected * self.intrinsic.diagonal()[:2] + self.intrinsic[:2, 2]
        if not np.allclose(projected, pixels, rtol=0, atol=1e-4):
            raise ValueError(f"相机 {self.name} 的畸变无法可靠反解。")
        width, height = (int(math.ceil(float(v) * focal + 2)) * 2 for v in extent)
        size = (width, height)
        grid = (rays * (2 * focal / np.asarray(size))).astype(np.float32)
        return size, math.degrees(2 * math.atan(size[1] / (2 * focal))), grid


def load_robot_cameras(source: Path, link_names: set[str]) -> dict[str, RobotCamera]:
    """Read composed camera prims, never historical source-position annotations."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError("无法读取机器人相机资产。")
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    cameras = {}
    prefix = "omni:lensdistortion:opencvPinhole:"
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera) or not prim.GetAttribute(prefix + "fx").HasAuthoredValueOpinion():
            continue
        name = prim.GetName()
        if name == "observer" or name in cameras:
            raise ValueError(f"相机名称重复或保留：{name}")
        parent = prim.GetParent()
        if parent.GetName() not in link_names:
            raise ValueError(f"相机 {name} 的挂载连杆未载入。")
        camera = UsdGeom.Camera(prim)
        resolution = tuple(prim.GetAttribute(prefix + "imageSize").Get() or ())
        values = [
            prim.GetAttribute(prefix + key).Get() for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
        ]
        if (
            len(resolution) != 2
            or any(type(v) is not int or not 1 <= v <= 1920 for v in resolution)
            or any(v is None or not math.isfinite(v) for v in values)
        ):
            raise ValueError(f"相机 {name} 的分辨率或标定无效。")
        if min(values[:2]) <= 0 or camera.GetProjectionAttr().Get() != "perspective":
            raise ValueError(f"相机 {name} 的分辨率或标定无效。")
        mount = UsdGeom.Xformable(prim)
        local = np.asarray(mount.GetLocalTransformation()).T.copy()
        local[:3, 3] *= unit
        if (
            mount.GetResetXformStack()
            or not np.isfinite(local).all()
            or not np.allclose(local[3], [0, 0, 0, 1], atol=1e-5)
        ):
            raise ValueError(f"相机 {name} 的挂载变换必须是相对连杆的刚体变换。")
        rotation = local[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError(f"相机 {name} 的挂载变换必须是相对连杆的刚体变换。")
        # USD cameras look along -Z with +Y up; returned poses use OpenCV axes.
        local = local @ np.diag([1, -1, -1, 1])
        near, far = (float(v) * unit for v in camera.GetClippingRangeAttr().Get())
        if not 0 < near < far or not math.isfinite(far):
            raise ValueError(f"相机 {name} 的裁剪范围无效。")
        fx, fy, cx, cy, *distortion = values
        cameras[name] = RobotCamera(
            name,
            parent.GetName(),
            resolution,
            np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            np.asarray(distortion),
            local,
            (near, far),
        )
    return cameras
