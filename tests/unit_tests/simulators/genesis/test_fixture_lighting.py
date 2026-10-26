"""Luminaire geometry fitting and radiance conservation without a GPU."""

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.fixture_lighting import Diffuser, fit_rectangles, read_diffusers
from jiuwensymbiosis.simulators.genesis.source_lighting import RectangleEmitter, SourceLighting


def sample():
    corners = ((-0.2, -0.5, 2.48), (-0.2, 0.5, 2.48), (0.2, 0.5, 2.48), (0.2, -0.5, 2.48))
    rect = RectangleEmitter("/Lamp", corners, (100, 90, 80), 0.4)
    rig = SourceLighting("original.hdr", -2, 4924, (1, 1, 1), (rect,), -8)
    diffuser = Diffuser(
        "/Fixture",
        np.array([[-0.195, -0.275, 2.505], [-0.195, 0.275, 2.505], [0.195, 0.275, 2.505], [0.195, -0.275, 2.505]]),
    )
    return rig, diffuser


def test_fit_is_inside_diffuser_with_submillimetre_gap_and_conserves_flux():
    rig, diffuser = sample()
    fitted = fit_rectangles(rig, [diffuser])
    rect = fitted.rectangles[0]
    corners = np.array(rect.corners_m)
    assert corners[:, 2] == pytest.approx([2.5048] * 4)
    assert np.all(corners[:, :2].min(axis=0) > diffuser.vertices[:, :2].min(axis=0))
    assert np.all(corners[:, :2].max(axis=0) < diffuser.vertices[:, :2].max(axis=0))
    assert np.cross(corners[1] - corners[0], corners[3] - corners[0])[2] < 0
    assert np.array(rect.radiance) * rect.area_m2 == pytest.approx(np.array(rig.rectangles[0].radiance) * 0.4)
    assert fitted.hdr == rig.hdr and fitted.multiplier == rig.multiplier and fitted.exposure_ev == rig.exposure_ev
    assert fitted.summary()["fitted_rectangle_count"] == 1
    assert rig.rectangles[0].fixture_path is None


def test_missing_ambiguous_and_duplicate_matches_fail():
    from dataclasses import replace

    rig, diffuser = sample()
    with pytest.raises(ValueError, match="找不到"):
        fit_rectangles(rig, [])
    with pytest.raises(ValueError, match="不唯一"):
        fit_rectangles(rig, [diffuser, Diffuser("/Other", diffuser.vertices + [0.001, 0, 0])])
    with pytest.raises(ValueError, match="重复"):
        fit_rectangles(replace(rig, rectangles=rig.rectangles * 2), [diffuser])


def test_bright_fitted_emitters_share_audited_attenuation_without_changing_hdr():
    from dataclasses import replace

    rig, diffuser = sample()
    rig = replace(rig, rectangles=(replace(rig.rectangles[0], radiance=(80000, 60000, 40000)),))
    fitted = fit_rectangles(rig, [diffuser])
    rect = fitted.rectangles[0]
    assert max(rect.radiance) == pytest.approx(60000)
    assert 0 < fitted.emitter_gain < 1
    assert np.asarray(rect.radiance) * rect.area_m2 == pytest.approx(
        np.asarray(rig.rectangles[0].radiance) * 0.4 * fitted.emitter_gain
    )
    assert fitted.hdr == rig.hdr and fitted.multiplier == rig.multiplier and fitted.exposure_ev == rig.exposure_ev


def test_fitted_corners_do_not_cover_bevelled_fixture_frame():
    from scipy.spatial import ConvexHull

    rig, _ = sample()
    xy = np.array(
        [
            [-0.195, -0.24],
            [-0.16, -0.275],
            [0.16, -0.275],
            [0.195, -0.24],
            [0.195, 0.24],
            [0.16, 0.275],
            [-0.16, 0.275],
            [-0.195, 0.24],
        ]
    )
    diffuser = Diffuser("/Bevelled", np.c_[xy, np.full(8, 2.505)])
    fitted = fit_rectangles(rig, [diffuser]).rectangles[0]
    hull = ConvexHull(xy).equations
    corners = np.asarray(fitted.corners_m)
    assert np.all(corners[:, :2] @ hull[:, :2].T + hull[:, 2] < 0)
    assert fitted.area_m2 < (0.39 - 0.002) * (0.55 - 0.002)
    assert np.asarray(fitted.radiance) * fitted.area_m2 == pytest.approx([40, 36, 32])


def test_diffuser_reads_composed_units_and_ignores_nonemissive_meshes(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    p = tmp_path / "fixture.usda"
    s = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(s, "Z")
    UsdGeom.SetStageMetersPerUnit(s, 0.01)
    parent = UsdGeom.Xform.Define(s, "/Fixture")
    parent.AddTranslateOp().Set((0, 0, 250))
    material = UsdShade.Material.Define(s, "/Material")
    shader = UsdShade.Shader.Define(s, "/Material/Shader")
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    for name in ["Light", "Plain"]:
        m = UsdGeom.Mesh.Define(s, "/Fixture/" + name)
        m.CreatePointsAttr([(-20, -30, 0), (-20, 30, 0), (20, 30, 0), (20, -30, 0)])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        if name == "Light":
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    s.GetRootLayer().Save()
    ds = read_diffusers(p)
    assert len(ds) == 1
    assert ds[0].vertices.min(axis=0) == pytest.approx([-0.2, -0.3, 2.5])
    assert ds[0].vertices.max(axis=0) == pytest.approx([0.2, 0.3, 2.5])
