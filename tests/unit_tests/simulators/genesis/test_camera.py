"""Camera geometry and preview settings, without a renderer or worker."""

import math
from itertools import product

import pytest

from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01, True, "1"])
def test_camera_rejects_invalid_input_without_changing_pose(value):
    camera = OrbitCamera()
    with pytest.raises(ValueError):
        camera.command("orbit", dx=value)
    assert camera.revision == 0


def test_camera_orbit_pan_zoom_and_reset_in_world_metres():
    camera = OrbitCamera()
    rotated = camera.command("orbit", dx=0.2, dy=0.1)
    assert rotated.snapshot()["pos"] != camera.snapshot()["pos"]
    assert rotated.target == camera.target
    panned = rotated.command("pan", dx=0.1, dy=0.1)
    assert panned.target != rotated.target
    zoomed = panned.command("zoom", delta=-0.3)
    assert zoomed.distance < panned.distance
    for _ in range(100):
        zoomed = zoomed.command("zoom", delta=-1).command("orbit", dy=1).command("pan", dy=1)
    assert zoomed.distance == 0.02
    assert zoomed.pitch == 1.48
    assert zoomed.snapshot()["pos"][2] <= 20
    reset = zoomed.command("reset")
    assert reset.snapshot()["pos"] == camera.snapshot()["pos"]
    assert reset.revision == zoomed.revision + 1


def test_fly_looks_from_fixed_eye_and_orbit_keeps_new_target():
    camera = OrbitCamera()
    looked = camera.command("fly", dx=0.25, dy=-0.6)
    assert looked.snapshot()["pos"] == pytest.approx(camera.snapshot()["pos"])
    assert looked.target != camera.target
    assert looked.pitch < 0  # Can look up, beyond the old orbit hemisphere.
    orbited = looked.command("orbit", dx=0.1)
    assert orbited.target == looked.target
    assert orbited.pitch == looked.pitch
    assert orbited.snapshot()["pos"] != looked.snapshot()["pos"]


def test_fly_translation_uses_camera_heading_and_world_up():
    camera = OrbitCamera(yaw=math.pi / 2, pitch=0, scene_scale=3)
    moved = camera.command("fly", forward=0.2, strafe=0.1, lift=0.05)
    expected = [-0.3, -0.6, 0.15]
    assert [moved.snapshot()["pos"][i] - camera.snapshot()["pos"][i] for i in range(3)] == pytest.approx(expected)
    assert [moved.target[i] - camera.target[i] for i in range(3)] == pytest.approx(expected)
    assert (moved.yaw, moved.pitch, moved.distance) == (camera.yaw, camera.pitch, camera.distance)
    combined = camera.command("fly", dx=0.5, forward=0.2)
    # After a quarter turn, forward follows -X instead of the old -Y heading.
    assert combined.snapshot()["pos"][0] == pytest.approx(camera.snapshot()["pos"][0] - 0.6)
    assert combined.snapshot()["pos"][1] == pytest.approx(camera.snapshot()["pos"][1])


@pytest.mark.parametrize("field", ["forward", "strafe", "lift"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, True, "1"])
def test_fly_rejects_non_finite_or_unbounded_distance(field, value):
    with pytest.raises(ValueError):
        OrbitCamera().command("fly", **{field: value})


def test_fly_limits_translation_and_pitch_without_a_pole_singularity():
    camera = OrbitCamera()
    for _ in range(100):
        camera = camera.command("fly", dy=-1, forward=1, strafe=1, lift=1)
    assert camera.pitch == -1.48
    assert all(abs(v) <= 20 + 1e-10 for v in camera.snapshot()["pos"])
    assert all(math.isfinite(v) for v in camera.snapshot()["pos"])
    assert camera.command("fly", dx=0.1).snapshot()["pos"] == pytest.approx(camera.snapshot()["pos"])


def test_orbit_eye_outside_flight_bounds_can_return_without_snapping():
    camera = OrbitCamera(yaw=0, pitch=0, target=(20, 0, 0))
    assert camera.snapshot()["pos"][0] == 23
    assert camera.command("fly", lift=0.1).snapshot()["pos"][0] == 23
    assert camera.command("fly", forward=0.1).snapshot()["pos"][0] == pytest.approx(22.9)
    assert camera.command("fly", forward=-0.1).snapshot()["pos"][0] == 23


@pytest.mark.parametrize(
    "values",
    [
        {"fps": 0},
        {"fps": True},
        {"fps": 1000},
        {"width": 8192},
        {"spp": 0},
        {"dt": float("nan")},
        {"render_mode": "invalid"},
    ],
)
def test_preview_settings_reject_invalid_budgets(values):
    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings

    with pytest.raises(ValueError):
        PreviewSettings(**values)


@pytest.mark.parametrize("aspect", [16 / 9, 9 / 16])
def test_focus_fits_all_corners_and_sets_orbit_center(aspect):
    box = [[-2, -0.4, 0], [2, 0.4, 1.3]]
    camera = OrbitCamera().focus(box, fov_y=60, aspect=aspect)
    assert camera.target == pytest.approx([0, 0, 0.65])
    original_eye = OrbitCamera().snapshot()["pos"]
    original_direction = [original_eye[i] - camera.target[i] for i in range(3)]
    assert camera.yaw == pytest.approx(math.atan2(original_direction[1], original_direction[0]))
    eye = camera.snapshot()["pos"]
    outward = [
        math.cos(camera.pitch) * math.cos(camera.yaw),
        math.cos(camera.pitch) * math.sin(camera.yaw),
        math.sin(camera.pitch),
    ]
    right = [-math.sin(camera.yaw), math.cos(camera.yaw), 0]
    up = [
        -math.cos(camera.yaw) * math.sin(camera.pitch),
        -math.sin(camera.yaw) * math.sin(camera.pitch),
        math.cos(camera.pitch),
    ]
    for corner in product(*zip(*box, strict=True)):
        relative = [corner[i] - eye[i] for i in range(3)]
        depth = -sum(a * b for a, b in zip(relative, outward, strict=True))
        assert depth > 0
        assert abs(sum(a * b for a, b in zip(relative, right, strict=True))) < depth * math.tan(math.pi / 6) * aspect
        assert abs(sum(a * b for a, b in zip(relative, up, strict=True))) < depth * math.tan(math.pi / 6)
    assert camera.command("orbit", dx=0.2).target == camera.target


def test_small_focused_object_can_zoom_closer_without_jumping_back():
    camera = OrbitCamera().focus([[-0.02] * 3, [0.02] * 3], fov_y=60, aspect=16 / 9)
    assert camera.distance < 0.2
    assert camera.command("zoom", delta=-0.1).distance < camera.distance


def test_reset_preserves_scene_context_and_subsequent_bounds():
    camera = OrbitCamera(scene_scale=3, center=(5, 0, 0)).command("orbit", dx=0.2)
    reset = camera.command("reset")
    assert reset.scene_scale == 3
    assert reset.center == (5, 0, 0)
    assert reset.revision == camera.revision + 1
    assert reset.snapshot()["pos"] == OrbitCamera().snapshot()["pos"]
    zoomed = reset
    flown = reset
    for _ in range(100):
        zoomed = zoomed.command("zoom", delta=1)
        flown = flown.command("fly", strafe=1, lift=1)
    assert zoomed.distance == 24
    assert flown.snapshot()["pos"][0] == pytest.approx(65)
    assert flown.snapshot()["pos"][2] == pytest.approx(60)
