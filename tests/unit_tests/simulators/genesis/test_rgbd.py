"""RGB-D calibration conventions at the shared perception boundary."""

import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview
from jiuwensymbiosis.simulators.genesis.rgbd import base_body_bounds, base_from_camera, effective_focal, rectified_grid


@pytest.fixture
def ray_sdk(monkeypatch):
    """Match the native float32 carrier and its signed axis permutations."""

    class Float3:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = map(float, np.asarray([x, y, z], dtype=np.float32))

    sdk = ModuleType("gs_nyx.nyx_py_sdk")
    sdk.float3 = Float3
    sdk.float2 = lambda x, y: tuple(map(float, np.asarray([x, y], dtype=np.float32)))
    sdk.float3_z_up_to_y_up_a = lambda p: Float3(p.y, p.z, p.x)
    sdk.float3_y_up_a_to_z_up = lambda p: Float3(p.z, p.x, p.y)
    renderer = ModuleType("gs_nyx.nyx_py_renderer")
    renderer.SceneReference = object
    package = ModuleType("gs_nyx")
    package.nyx_py_sdk, package.nyx_py_renderer = sdk, renderer
    for name, module in (("gs_nyx", package), (sdk.__name__, sdk), (renderer.__name__, renderer)):
        monkeypatch.setitem(sys.modules, name, module)
    return sdk


@pytest.mark.parametrize("angle", [0.0, 0.45, -1.2])
def test_sample_depth_is_optical_z_in_metres_with_sparse_holes(ray_sdk, angle):
    from jiuwensymbiosis.simulators.genesis.rgbd import sample_depth

    camera = SimpleNamespace(
        resolution=(17, 13), intrinsic=np.array([[12, 0, 8], [0, 12, 6], [0, 0, 1]]), clipping=(0.1, 4.0)
    )
    transform = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    transform[:3, 3] = [1.3, -2.4, 0.7]
    calls = []

    def trace(origin, direction, limits, hit, reference):
        world_origin = ray_sdk.float3_y_up_a_to_z_up(origin)
        world_direction = ray_sdk.float3_y_up_a_to_z_up(direction)
        vector = np.array([world_direction.x, world_direction.y, world_direction.z])
        optical_z = float(vector @ transform[:3, 2])
        np.testing.assert_allclose(limits, np.array(camera.clipping) / optical_z, atol=1e-6)
        origin_array = np.array([world_origin.x, world_origin.y, world_origin.z])
        np.testing.assert_allclose(origin_array, transform[:3, 3], atol=1e-6)
        calls.append(1)
        if len(calls) == 2:
            return False
        point = origin_array + vector * (2.0 / optical_z)
        native_point = ray_sdk.float3_z_up_to_y_up_a(ray_sdk.float3(*point))
        hit.x, hit.y, hit.z = native_point.x, native_point.y, native_point.z
        return True

    depth = sample_depth(SimpleNamespace(trace_ray=trace), camera, transform, stride=4, check=lambda: None)
    assert len(calls) == 20
    assert depth.dtype == np.float32
    assert np.isfinite(depth).sum() == 19
    assert np.isnan(depth[0, 4]) and np.isnan(depth[1, 1])
    np.testing.assert_allclose(depth[np.isfinite(depth)], 2.0, atol=5e-7)


def test_sample_depth_cancellation_remains_bounded_between_ray_blocks(ray_sdk):
    from jiuwensymbiosis.simulators.genesis.rgbd import sample_depth

    camera = SimpleNamespace(resolution=(65, 65), intrinsic=np.eye(3), clipping=(0.1, 4.0))
    calls, checks = [], []

    def check():
        checks.append(1)
        if len(checks) == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        sample_depth(SimpleNamespace(trace_ray=lambda *args: calls.append(1)), camera, np.eye(4), stride=4, check=check)
    assert len(calls) == 32


def test_validated_fov_correction_and_unknown_versions_fail_closed():
    focal = effective_focal((684, 524), math.degrees(1.1832457862453918), "0.1.7", "0.1.9")
    assert focal == pytest.approx(389.7874463923982)
    with pytest.raises(ValueError, match="validation"):
        effective_focal((684, 524), 60, "0.2.0", "0.1.9")


def test_rectified_grid_roundtrips_pinhole_pixels_without_distortion():
    camera = SimpleNamespace(resolution=(640, 480), intrinsic=np.array([[390, 0, 332], [0, 390, 252], [0, 0, 1]]))
    focal = effective_focal((684, 524), math.degrees(1.1832457862453918), "0.1.7", "0.1.9")
    grid = rectified_grid(camera, (684, 524), focal)
    rays = grid.astype(float) * [684, 524] / (2 * focal)
    pixels = rays * [390, 390] + [332, 252]
    expected = np.stack(np.meshgrid(np.arange(640), np.arange(480)), axis=-1)
    np.testing.assert_allclose(pixels, expected, atol=3e-5)


def test_live_base_transform_converts_metres_to_mm_and_body_forward_axes():
    world_camera = np.eye(4)
    world_camera[:3, 3] = [1, 3, 1.2]
    result = base_from_camera(world_camera, {"position_m": [1, 2, 0.2], "body_yaw_rad": math.pi / 2})
    np.testing.assert_allclose(result[:3, 3], [1000, 0, 1000], atol=1e-10)
    np.testing.assert_allclose(result[:3, :3] @ [0, 1, 0], [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(result[3], [0, 0, 0, 1])


def test_body_envelope_is_robot_geometry_in_base_metres():
    result = base_body_bounds(
        np.array([[0.7, 1.6, 0.1], [1.3, 2.4, 2.1]]), {"position_m": [1, 2, 0.2], "body_yaw_rad": math.pi / 2}
    )
    np.testing.assert_allclose(result, [[-0.4, -0.3, -0.1], [0.4, 0.3, 1.9]], atol=1e-10)


@pytest.mark.parametrize(
    "camera,stride", [("observer", 8), ("head_cam", True), ("head_cam", 3), ("head_cam", 33), ("head_cam", 8.0)]
)
def test_invalid_rgbd_request_is_rejected_before_worker_access(tmp_path, camera, stride):
    preview = GenesisPreview(state_dir=tmp_path)
    with pytest.raises(ValueError, match="robot camera and sampling stride"):
        preview.capture_image(camera, include_depth=True, depth_stride=stride)
