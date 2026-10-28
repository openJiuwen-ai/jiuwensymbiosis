"""Camera color settings reach the SDK without loading GPU libraries."""

import sys
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.simulators.genesis import appearance, source_lighting
from jiuwensymbiosis.simulators.genesis.rendering import SceneRenderer


@pytest.mark.parametrize("tone_mapper, expected", [("reinhard", 11), ("neutral", 22), ("aces_1_3", 33)])
@pytest.mark.parametrize("source_exposure", [None, -8.0])
@pytest.mark.parametrize("with_source", [False, True])
def test_renderer_exports_effective_color_without_changing_lights(
    monkeypatch, tmp_path, tone_mapper, expected, source_exposure, with_source
):
    config = tmp_path / "appearance.yaml"
    config.write_text(
        f"scenes:\n  indoor:\n    color_management:\n      tone_mapper: {tone_mapper}\n      exposure_adjust_ev: 0.5\n"
    )
    monkeypatch.setattr(appearance, "PROFILE_PATH", config)
    monkeypatch.setattr(source_lighting, "append_emitters", lambda *args: None)
    captured = {}

    class SceneAsset:
        def __init__(self):
            self.instances = []

        def instance_size(self):
            return len(self.instances)

        def instance_resize(self, count):
            self.instances.extend(SimpleNamespace() for _ in range(count - len(self.instances)))

        def get_instance(self, index):
            return self.instances[index]

        def set_instance(self, index, instance):
            self.instances[index] = instance

    class CameraProfile:
        def __init__(self):
            self.colorManagement = SimpleNamespace()
            self.sensor = SimpleNamespace(exposure=SimpleNamespace())

        def set_fov_y(self, fov):
            self.fov = fov

    class Exporter:
        def __init__(self, *args, **kwargs):
            captured["exporter"] = kwargs
            self._entity_uuid_pairs = []

        def export_to_file(self, path):
            pass

    class Renderer:
        def __init__(self, *args, **kwargs):
            captured["renderer"] = kwargs

        def build(self, pairs):
            assert pairs == []

        def destroy(self):
            captured["destroyed"] = captured.get("destroyed", 0) + 1

        def update_camera_tensor(self, *args):
            pass

    def export_profile(path, profile):
        captured["profile"] = profile
        # A second config read during the same build would now fail.
        config.write_text("invalid: document")

    sdk = SimpleNamespace(
        CameraProfileAsset=CameraProfile,
        EToneMapper=SimpleNamespace(Reinhard=11, Neutral=22, ACES_1_3=33),
        ELightType=SimpleNamespace(Point=1),
        ELightUnit=SimpleNamespace(Lumen=1),
        LightAsset=lambda: captured.setdefault("fill", SimpleNamespace()),
        float3=lambda *args: args,
        quaternion=lambda *args: args,
        float3_z_up_to_y_up_a=lambda point: point,
        export_camera_profile_file=export_profile,
        SceneAsset=SceneAsset,
        EInstanceType=SimpleNamespace(Mesh=1),
        import_scene_file=lambda *args: True,
        export_scene_file=lambda path, asset: captured.update(background=asset),
        generate_uuid=lambda: "background-id",
    )
    modes = SimpleNamespace(ERenderMode=SimpleNamespace(FastPathTracer=1, RefPathTracer=2))
    modules = {
        "gs_nyx": SimpleNamespace(nyx_py_renderer=modes, nyx_py_sdk=sdk),
        "gs_nyx.nyx_py_renderer": modes,
        "gs_nyx.nyx_py_sdk": sdk,
        "gs_nyx_plugin": SimpleNamespace(),
        "gs_nyx_plugin.nyx_renderer": SimpleNamespace(NyxPyRenderer=Renderer),
        "gs_nyx_plugin.nyx_scene_exporter": SimpleNamespace(NyxSceneExporter=Exporter),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    settings = SimpleNamespace(width=1280, height=720, spp=8, scene_id="indoor", render_mode="fast")
    source = None
    if with_source:
        source = tmp_path / "source.usda"
        source.write_text(
            '#usda 1.0\n(defaultPrim = "World")\n'
            'def Xform "World" {\n'
            '    def Shader "Material" {\n'
            '        uniform token info:id = "UsdPreviewSurface"\n'
            "        float inputs:roughness = 0.37\n"
            "        float inputs:opacity = 0.2\n"
            "        asset inputs:texture = @texture.bin@\n"
            "    }\n}\n"
        )
        (tmp_path / "texture.bin").write_bytes(b"original texture data")
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path != config}
    renderer = SceneRenderer(
        object(),
        source,
        tmp_path,
        settings,
        {"pos": (0, 0, 1), "lookat": (0, 0, 0), "up": (0, 1, 0)},
        object(),
        exposure=-3.0,
        source_lighting=None if source_exposure is None else SimpleNamespace(exposure_ev=source_exposure),
    )
    base = -3.0 if source_exposure is None else source_exposure
    profile = captured["profile"]
    assert profile.colorManagement.toneMapper == expected
    assert profile.sensor.exposure.exposureCompensation == base + 0.5
    assert renderer.appearance["color_management"] == {
        "tone_mapper": tone_mapper,
        "base_exposure_ev": base,
        "exposure_adjust_ev": 0.5,
        "effective_exposure_ev": base + 0.5,
    }
    assert captured["exporter"]["lights"] == ([captured["fill"]] if with_source and source_exposure is None else [])
    assert captured["fill"].intensity == 240.0
    assert captured["renderer"]["camera_defs"][0]["denoise"] is True
    if with_source:
        instance = captured["background"].instances[0]
        assert instance.mesh_uri == str(source)
        assert not hasattr(instance, "matOverride")
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path != config} == original_files

    renderer.close()
    renderer.close()
    assert captured["destroyed"] == 1
