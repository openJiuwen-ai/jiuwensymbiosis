"""MJCF registration stays local and does not grant robot control."""

import pytest

from jiuwensymbiosis.simulators.assets import AssetError
from jiuwensymbiosis.simulators.assets.config import RootSpec
from jiuwensymbiosis.simulators.assets.scanners import scan_root


def test_mjcf_tracks_includes_and_rejects_escaping_references(tmp_path):
    (tmp_path / "scene.xml").write_text('<mujoco><include file="robot.xml"/></mujoco>')
    (tmp_path / "robot.xml").write_text("<mujocoinclude><worldbody/></mujocoinclude>")
    spec = RootSpec(id="local", path=str(tmp_path), scanner="mjcf", entries=["scene.xml"])
    asset = scan_root(spec, tmp_path).assets[0]
    assert asset.preview_mode == "mjcf" and asset.kind == "environment"
    assert set(asset.representations.values()) == {"scene.xml", "robot.xml"}
    (tmp_path / "robot.xml").unlink()
    (tmp_path / "robot.xml").symlink_to(__file__)
    with pytest.raises(AssetError, match="离开资产根目录"):
        scan_root(spec, tmp_path)


@pytest.mark.parametrize("contents", ['<mujoco><include file="scene.xml"/></mujoco>', "<robot/>"])
def test_mjcf_rejects_cycles_and_other_xml_formats(tmp_path, contents):
    (tmp_path / "scene.xml").write_text(contents)
    spec = RootSpec(id="local", path=str(tmp_path), scanner="mjcf", entries=["scene.xml"])
    with pytest.raises(AssetError):
        scan_root(spec, tmp_path)


def test_mjcf_requires_explicit_relative_entries():
    for entries in ([], ["../outside.xml"], ["https://example.org/scene.xml"], ["model.urdf"]):
        with pytest.raises(ValueError):
            RootSpec(id="local", path=".", scanner="mjcf", entries=entries)
