"""Single-floor physics cannot silently reintroduce house mesh collision."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_indoor import box

from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.single_floor import SingleFloor
from jiuwensymbiosis.simulators.genesis.virtual_camera import optical_pose


def test_one_invisible_plane_at_supported_height_no_mesh_export(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "genesis",
        SimpleNamespace(
            morphs=SimpleNamespace(Plane=lambda **kw: ("plane", kw)),
            materials=SimpleNamespace(Rigid=lambda **kw: kw),
            surfaces=SimpleNamespace(Collision=lambda: "collision-only"),
        ),
    )
    entity = SimpleNamespace(geom_start=5, geom_end=6)

    def add(**kw):
        calls.append(kw)
        return entity

    meshes = [box((4, 4, 0.14), (0, 0, 0.07), "floor"), box((0.5, 0.5, 1), (0, 0, 0.64), "cupboard")]
    monkeypatch.setattr(
        type(meshes[0].mesh),
        "export",
        lambda *a, **kw: pytest.fail("House geometry must never be exported for physics"),
    )
    room = SingleFloor(SimpleNamespace(add_entity=add), meshes)
    assert len(calls) == 1
    assert calls[0]["morph"] == ("plane", {"pos": (0, 0, pytest.approx(0.14)), "visualization": False})
    assert room.summary()["house_collision_entities"] == 0
    assert room.summary()["wall_furniture_contacts_available"] is False
    assert room.meshes is meshes
    contacts = {
        "geom_a": np.array([1, 5, 1]),
        "geom_b": np.array([5, 1, 99]),
        "force_a": np.array([[0, 0, 10], [0, 0, -20], [0, 0, 30]]),
        "position": np.zeros((3, 3)),
        "valid_mask": np.array([True, True, True]),
    }
    report = room.contacts(SimpleNamespace(entity=SimpleNamespace(get_contacts=lambda **kw: contacts)))
    assert report["count"] == 2
    assert report["surfaces"][0]["force_xyz_n"] == [0, 0, 30]
    assert report["scope"] == "ground_only"


def test_explicit_height_requires_source_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace())
    with pytest.raises(ValueError, match="地面"):
        SingleFloor(None, [box((4, 4, 0.1), (0, 0, 0.05), "floor")], floor_z=10)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0"])
def test_settings_reject_invalid_floor_height(value):
    with pytest.raises(ValueError):
        PreviewSettings(floor_height_m=value)


def test_virtual_pose_preserves_horizontal_optical_axes():
    p = optical_pose({"pos": [2, 3, 1.2], "lookat": [3, 3, 1.2], "up": [0, 0, 1]})
    t = np.array(p["T_world_camera"])
    np.testing.assert_allclose(t[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    np.testing.assert_allclose(t[:3, 3], [2, 3, 1.2])
    assert np.linalg.det(t[:3, :3]) == pytest.approx(1)


def test_web_configuration_preserves_virtual_mode_and_explicit_floor():
    from jiuwensymbiosis.simulators.genesis.stream import _preview_configuration
    from jiuwensymbiosis.simulators.genesis.web.routes import StartOptions

    options = StartOptions(robot_mode="virtual_camera", camera_name="scan_cam", floor_height_m=0.14)
    config = _preview_configuration(options.preview_options(), 1280, 720)
    assert config["robot_mode"] == "virtual_camera"
    assert config["floor_height_m"] == 0.14
    assert "floor_height_m" not in StartOptions().preview_options()
    assert StartOptions(floor_height_m=None).preview_options()["floor_height_m"] is None
    for value in [True, float("nan"), "0"]:
        with pytest.raises(ValueError):
            StartOptions(floor_height_m=value)
