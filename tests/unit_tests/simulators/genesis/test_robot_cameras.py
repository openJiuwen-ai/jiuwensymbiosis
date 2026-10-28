"""Camera mounts and calibrated pixels must agree with the delivered image."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.robot_cameras import RobotCamera, load_robot_cameras


@pytest.fixture
def camera():
    return RobotCamera(
        "head_cam",
        "head",
        (640, 480),
        np.array([[389.88, 0, 332.75], [0, 389.93, 252.76], [0, 0, 1]]),
        np.array([-0.041943, 0.039136, 0.00582, 0.00367, 0]),
        np.eye(4),
        (0.03, 50),
    )


def test_remap_pixels_match_opencv_projection_and_have_full_coverage(camera):
    cv2 = pytest.importorskip("cv2")
    size, _, grid = camera.render_map()
    assert np.max(np.abs(grid)) < 1
    # An independent forward projection must recover every destination pixel,
    # including the principal point shift, tangential distortion and corners.
    rays = grid.astype(float) * np.array(size) / (2 * max(camera.intrinsic[0, 0], camera.intrinsic[1, 1]))
    xyz = np.concatenate([rays, np.ones((*rays.shape[:2], 1))], axis=-1)
    projected, _ = cv2.projectPoints(xyz.reshape(-1, 3), np.zeros(3), np.zeros(3), camera.intrinsic, camera.distortion)
    expected = np.stack(np.meshgrid(np.arange(640), np.arange(480)), axis=-1)
    assert np.max(np.abs(projected.reshape(480, 640, 2) - expected)) < 0.0001
    assert camera.metadata()["D"] == camera.distortion.tolist()


def test_nonconvergent_distortion_is_rejected(camera):
    pytest.importorskip("cv2")
    camera = replace(
        camera,
        intrinsic=np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]]),
        distortion=np.array([-0.3, 0, 0, 0, 0]),
    )
    with pytest.raises(ValueError, match="畸变"):
        camera.render_map()


def test_mount_follows_live_link_translation_and_rotation(camera):
    def tensor(row):
        return [SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(row)))]

    local = np.eye(4)
    local[:3, 3] = [0.1, 0.2, 0.3]
    camera = replace(camera, link_from_camera=local)
    link = SimpleNamespace(get_pos=lambda: tensor([1, 2, 3]), get_quat=lambda: tensor([2**-0.5, 0, 0, 2**-0.5]))
    pose = camera.pose(link)
    np.testing.assert_allclose(pose["pos"], [0.8, 2.1, 3.3])
    np.testing.assert_allclose(pose["lookat"], [0.8, 2.1, 4.3])
    np.testing.assert_allclose(pose["up"], [1, 0, 0], atol=1e-12)


@pytest.fixture
def camera_usd(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.Xform.Define(stage, "/robot/head")
    camera = UsdGeom.Camera.Define(stage, "/robot/head/head_cam")
    camera.CreateClippingRangeAttr((0.03, 50))
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(-0.07, 0.015, 0.035))
    p = camera.GetPrim()
    p.CreateAttribute("spiritai:sourcePositionM", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0.07, 0.015, 0.035))
    prefix = "omni:lensdistortion:opencvPinhole:"
    p.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(640, 480))
    for name, value in {
        "fx": 400,
        "fy": 405,
        "cx": 323,
        "cy": 263,
        "k1": -0.03,
        "k2": 0.03,
        "k3": 0,
        "p1": 0.01,
        "p2": 0.003,
    }.items():
        p.CreateAttribute(prefix + name, Sdf.ValueTypeNames.Float).Set(value)
    stage.GetRootLayer().Save()
    return path


def test_loader_uses_composed_mount_instead_of_historical_annotation(camera_usd, tmp_path):
    from pxr import Gf, Usd, UsdGeom

    variant = Usd.Stage.CreateNew(str(tmp_path / "variant.usda"))
    variant.GetRootLayer().subLayerPaths = [str(camera_usd)]
    variant.GetPrimAtPath("/robot/head/head_cam").GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.08, 0.02, 0.04))
    UsdGeom.SetStageMetersPerUnit(variant, 1)
    variant.GetRootLayer().Save()
    camera = load_robot_cameras(tmp_path / "variant.usda", {"head"})["head_cam"]
    np.testing.assert_allclose(camera.link_from_camera[:3, 3], [-0.08, 0.02, 0.04])
    np.testing.assert_allclose(camera.link_from_camera[:3, :3], np.diag([1, -1, -1]))
    assert camera.resolution == (640, 480)
    assert camera.link_name == "head"


def test_missing_mount_is_not_silently_replaced_by_observer(camera_usd):
    with pytest.raises(ValueError, match="挂载连杆"):
        load_robot_cameras(camera_usd, {"other"})


@pytest.mark.parametrize("kind", ["reflection", "reset_stack", "projective"])
def test_unsupported_mount_is_rejected(camera_usd, kind):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(camera_usd))
    mount = UsdGeom.Xformable(stage.GetPrimAtPath("/robot/head/head_cam"))
    if kind == "reflection":
        mount.AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
    elif kind == "reset_stack":
        mount.SetResetXformStack(True)
    else:
        matrix = Gf.Matrix4d(1)
        matrix[0, 3] = 0.5
        mount.AddTransformOp().Set(matrix)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="挂载变换"):
        load_robot_cameras(camera_usd, {"head"})


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_intrinsics_are_rejected(camera_usd, value):
    from pxr import Usd

    stage = Usd.Stage.Open(str(camera_usd))
    stage.GetPrimAtPath("/robot/head/head_cam").GetAttribute("omni:lensdistortion:opencvPinhole:fx").Set(value)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="标定无效"):
        load_robot_cameras(camera_usd, {"head"})
