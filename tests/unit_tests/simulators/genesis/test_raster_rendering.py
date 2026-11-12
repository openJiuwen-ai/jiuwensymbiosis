"""The portable Genesis demo keeps its observer camera across resizing."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.camera import OrbitCamera, PreviewSettings
from jiuwensymbiosis.simulators.genesis.rendering import RasterRenderer


def test_raster_observer_resize_reuses_scene_cameras_and_rejects_robot_views():
    pytest.importorskip("torch")
    created = []

    class Camera:
        def __init__(self, **options):
            self.options, self.renders, self.pose = options, 0, {}
            self.width, self.height = options["res"]

        def set_pose(self, **pose):
            self.pose = pose

        def render(self, **kwargs):
            self.renders += 1
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), None, None, None

    def add_camera(**kwargs):
        camera = Camera(**kwargs)
        created.append(camera)
        return camera

    pose = OrbitCamera().snapshot()
    renderer = RasterRenderer(SimpleNamespace(add_camera=add_camera), pose)
    renderer.configure(PreviewSettings(), pose)
    assert renderer.read_rgb().shape == (720, 1280, 3)
    renderer.close()
    renderer.configure(replace(PreviewSettings(), width=960, height=540), pose)
    assert renderer.read_rgb().shape == (540, 960, 3)
    assert len(created) == 3
    assert renderer.render_counts == {"observer": 2}
    with pytest.raises(ValueError, match="observer"):
        renderer.read_rgb("head_cam")


def test_unsupported_raster_size_is_rejected_without_replacing_camera():
    from unittest.mock import Mock

    scene = SimpleNamespace(add_camera=Mock(side_effect=lambda **kwargs: Mock()))
    pose = OrbitCamera().snapshot()
    renderer = RasterRenderer(scene, pose)
    original = renderer._camera
    with pytest.raises(ValueError, match="540p"):
        renderer.configure(SimpleNamespace(width=1024, height=768), pose)
    assert renderer._camera is original
    assert scene.add_camera.call_count == 3
    with pytest.raises(ValueError, match="540p"):
        PreviewSettings(width=1024, height=768)
