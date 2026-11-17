"""Source radiance and transforms survive translation without a GPU."""

import math

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis import source_lighting


@pytest.fixture
def rig_source(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    hdr = tmp_path / "sky.hdr"
    hdr.touch()
    dome = UsdLux.DomeLight.Define(stage, "/Dome")
    dome.CreateTextureFileAttr(Sdf.AssetPath("sky.hdr"))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(10)
    dome.CreateExposureAttr(2)
    dome.AddRotateZOp().Set(90)
    parent = UsdGeom.Xform.Define(stage, "/Room")
    parent.AddTranslateOp().Set((100, 200, 300))
    parent.AddRotateZOp().Set(90)
    rect = UsdLux.RectLight.Define(stage, "/Room/Lamp")
    rect.CreateWidthAttr(200)
    rect.CreateHeightAttr(100)
    rect.CreateIntensityAttr(100)
    rect.CreateExposureAttr(1)
    rect.CreateColorAttr((0.5, 0.25, 1))
    stage.GetRootLayer().Save()
    return source


def test_world_transform_units_radiance_and_emission_direction(rig_source):
    rig = source_lighting.read_source_lighting(rig_source, exposure_ev=-10)
    assert rig.multiplier == 40
    assert rig.rotation_rad == pytest.approx(math.pi / 2)
    assert rig.hdr == str(rig_source.parent / "sky.hdr")
    rect = rig.rectangles[0]
    corners = np.array(rect.corners_m)
    assert corners.mean(axis=0) == pytest.approx([1, 2, 3])
    assert np.ptp(corners, axis=0) == pytest.approx([1, 2, 0])
    assert rect.area_m2 == pytest.approx(2)
    assert rect.radiance == pytest.approx([100, 50, 200])
    assert np.cross(corners[1] - corners[0], corners[2] - corners[0])[2] < 0
    assert rig.audit()["collision_geometry_added"] == 0
    assert rig.audit()["inspection_light"] is False
    assert "hdr" not in rig.summary()
    assert "rectangles" not in rig.summary()
    assert rig.summary()["rectangle_count"] == 1


def test_normalized_lamp_divides_world_area(rig_source):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    UsdLux.LightAPI(stage.GetPrimAtPath("/Room/Lamp")).CreateNormalizeAttr(True)
    stage.GetRootLayer().Save()
    rig = source_lighting.read_source_lighting(rig_source, exposure_ev=-10)
    assert rig.rectangles[0].radiance == pytest.approx([50, 25, 100])


def test_missing_hdr_and_unsupported_light_fail_explicitly(rig_source):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    UsdLux.DiskLight.Define(stage, "/Unsupported")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="DiskLight"):
        source_lighting.read_source_lighting(rig_source, exposure_ev=-10)
    stage.RemovePrim("/Unsupported")
    stage.GetRootLayer().Save()
    (rig_source.parent / "sky.hdr").unlink()
    with pytest.raises(ValueError, match="HDR"):
        source_lighting.read_source_lighting(rig_source, exposure_ev=-10)


def test_tilt_cannot_be_silently_lost(rig_source):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(rig_source))
    UsdGeom.Xformable(stage.GetPrimAtPath("/Dome")).AddRotateXOp().Set(30)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="倾斜"):
        source_lighting.read_source_lighting(rig_source, exposure_ev=-10)


def test_loading_uses_asset_features_without_profile(rig_source, tmp_path):
    rig = source_lighting.load_source_lighting(rig_source, directory=tmp_path)
    assert rig.summary()["selection"] == "asset_features"
    assert len(rig.rectangles) == 1
    assert rig.fitting_decisions[0]["action"] == "preserve_source"
    assert rig.exposure_ev == -8


def test_no_dome_does_not_invent_fill_light(rig_source, tmp_path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(rig_source))
    stage.RemovePrim("/Dome")
    stage.GetRootLayer().Save()
    rig = source_lighting.load_source_lighting(rig_source, directory=tmp_path)
    assert rig.environment_kind == "none"
    assert rig.multiplier == 0
    assert not rig.summary()["inspection_light"]
    assert len(rig.rectangles) == 1


def test_y_up_converts_to_genesis_z_up_and_hidden_lights_are_ignored(rig_source):
    from pxr import Usd, UsdGeom, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    UsdGeom.SetStageUpAxis(stage, "Y")
    dome = UsdGeom.Xformable(stage.GetPrimAtPath("/Dome"))
    dome.ClearXformOpOrder()
    dome.AddRotateYOp().Set(90)
    hidden = UsdLux.SphereLight.Define(stage, "/Hidden")
    UsdGeom.Imageable(hidden.GetPrim()).MakeInvisible()
    stage.GetRootLayer().Save()
    rig = source_lighting.read_source_lighting(rig_source, exposure_ev=-8)
    assert rig.rotation_rad == pytest.approx(math.pi / 2)
    assert np.mean(rig.rectangles[0].corners_m, axis=0) == pytest.approx([1, -3, 2])
    assert len(rig.rectangles) == 1


@pytest.mark.parametrize("normalize", [None, False, True])
@pytest.mark.parametrize("scale_x, area", [(0.1, 0.2), (3.0, 6.0)])
def test_normalization_uses_world_area_only_when_enabled(rig_source, normalize, scale_x, area):
    # OpenUSD 25.11 LightAPI: sizeFactor is 1 when normalize is off.
    from pxr import Usd, UsdGeom, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    prim = stage.GetPrimAtPath("/Room/Lamp")
    UsdGeom.Xformable(prim).AddScaleOp().Set((scale_x, 1, 1))
    light = UsdLux.LightAPI(prim)
    if normalize is not None:
        light.CreateNormalizeAttr(normalize)
    stage.GetRootLayer().Save()
    original = rig_source.read_bytes()
    rectangle = source_lighting.read_source_lighting(rig_source, exposure_ev=-10).rectangles[0]
    assert rectangle.area_m2 == pytest.approx(area)
    base = np.array([100, 50, 200])
    expected = base / area if normalize else base
    assert rectangle.radiance == pytest.approx(expected)
    assert np.array(rectangle.radiance) * area == pytest.approx(base if normalize else base * area)
    assert rig_source.read_bytes() == original


def test_environment_rotation_uses_sdk_degrees(monkeypatch):
    """Keep USD's internal radians while honoring Nyx 0.1.9's degree boundary."""
    import sys
    from types import ModuleType, SimpleNamespace

    package = ModuleType("gs_nyx")
    sdk = ModuleType("gs_nyx.nyx_py_sdk")
    sdk.EnvironmentMapAsset = SimpleNamespace
    sdk.EEnvMapLayout = SimpleNamespace(LongLat="longlat")
    sdk.float3 = lambda *values: values
    package.nyx_py_sdk = sdk
    monkeypatch.setitem(sys.modules, "gs_nyx", package)
    monkeypatch.setitem(sys.modules, "gs_nyx.nyx_py_sdk", sdk)
    rig = source_lighting.SourceLighting("sky.hdr", math.pi / 2, 2, (1, 1, 1), (), 0)
    env = source_lighting.make_environment(rig)
    assert env.rotation == pytest.approx(90)
    assert env.texture == "sky.hdr"
    assert env.multiplier == 2


@pytest.mark.parametrize("angle", [90, 120, 180])
def test_neutral_hemisphere_shaping_is_supported(rig_source, angle):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    shaping = UsdLux.ShapingAPI.Apply(stage.GetPrimAtPath("/Room/Lamp"))
    shaping.CreateShapingConeAngleAttr(angle)
    stage.GetRootLayer().Save()
    assert len(source_lighting.read_source_lighting(rig_source, exposure_ev=0).rectangles) == 1


@pytest.mark.parametrize("field,value", [("ConeAngle", 45), ("Focus", 1), ("ConeSoftness", 0.5)])
def test_directional_shaping_is_not_silently_discarded(rig_source, field, value):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    shaping = UsdLux.ShapingAPI.Apply(stage.GetPrimAtPath("/Room/Lamp"))
    getattr(shaping, "CreateShaping" + field + "Attr")(value)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="hemispherical"):
        source_lighting.read_source_lighting(rig_source, exposure_ev=0)


@pytest.mark.parametrize("normalized", [False, True])
def test_native_sphere_and_distant_energy_and_transforms(rig_source, normalized):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    sphere = UsdLux.SphereLight.Define(stage, "/Sphere")
    sphere.CreateRadiusAttr(100)  # fixture centimetres -> one metre
    sphere.CreateIntensityAttr(100)
    sphere.CreateNormalizeAttr(normalized)
    sphere.AddTranslateOp().Set((100, 200, 300))
    sun = UsdLux.DistantLight.Define(stage, "/Sun")
    sun.CreateIntensityAttr(200)
    sun.CreateAngleAttr(60)
    sun.CreateNormalizeAttr(normalized)
    stage.GetRootLayer().Save()
    rig = source_lighting.read_source_lighting(rig_source, exposure_ev=-8)
    point, directional = rig.native_lights
    assert point["pos"] == pytest.approx([1, 2, 3])
    assert point["intensity"] == pytest.approx(25 if normalized else 100 * math.pi)
    assert directional["direction"] == pytest.approx([0, 0, -1])
    assert directional["intensity"] == pytest.approx(200 if normalized else 50 * math.pi)
    assert rig.summary()["point_count"] == rig.summary()["directional_count"] == 1
    assert len(rig.approximations) == 2


def test_nonuniform_sphere_is_explicitly_approximated(rig_source):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.Open(str(rig_source))
    sphere = UsdLux.SphereLight.Define(stage, "/Sphere")
    sphere.AddScaleOp().Set((1, 2, 1))
    stage.GetRootLayer().Save()
    rig = source_lighting.read_source_lighting(rig_source, exposure_ev=-8)
    assert rig.native_lights[0]["source_area_m2"] > 0
    assert any("nonuniform" in item["reason"] for item in rig.approximations)
