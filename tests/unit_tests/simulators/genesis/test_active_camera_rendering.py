"""Selected-camera rendering and view changes must preserve physical ownership."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common.preview import SimulationPreview
from jiuwensymbiosis.simulators.genesis.rendering import SceneRenderer


def test_single_camera_uses_native_buffer_start_for_every_camera():
    calls = []
    output = np.zeros(120, dtype=np.uint8)
    sizes = [(4, 3), (2, 2)]

    def render_frame(index):
        calls.append(index)
        width, height = sizes[index]
        output[: width * height * 3] = index + 1
        return True

    def render_batch():
        pytest.fail("A selected view must not render the full camera batch")

    renderer = SceneRenderer.__new__(SceneRenderer)
    renderer._camera_indices = {"observer": 0, "head_cam": 1}
    renderer.render_counts = {"observer": 0, "head_cam": 0}
    renderer._renderer = SimpleNamespace(
        _renderer=SimpleNamespace(
            render_frame=render_frame, render_batch=render_batch, camera_resolution=lambda index: sizes[index]
        ),
        _color_output=output,
        _batch_dirty=False,
        update=lambda: None,
        render=render_batch,
    )
    head = renderer.render_camera("head_cam")
    assert head.shape == (2, 2, 3)
    assert np.all(head == 2)
    observer = renderer.render_camera("observer")
    assert observer.shape == (3, 4, 3)
    assert np.all(observer == 1)
    assert calls == [1, 0]
    assert renderer.render_counts == {"observer": 1, "head_cam": 1}
    assert renderer._renderer._batch_dirty  # Batch views were not refreshed.


def test_camera_only_change_keeps_motion_binding_and_rejects_old_video(tmp_path):
    preview = SimulationPreview(state_dir=tmp_path)
    observed = {
        "state": "ready",
        "config_revision": 0,
        "stream_epoch": 0,
        "settings": asdict(preview.settings),
        "scene": {"instance_id": "scene"},
        "cameras": [{"name": "head_cam"}],
    }
    commands = []
    preview._observed_revision = 0
    preview._process = SimpleNamespace(poll=lambda: None, pid=1)
    preview._mailbox = SimpleNamespace(read_status=lambda: observed, write_control=commands.append)
    preview._motion_owner = "owner"
    preview._motion_binding = ("scene", 0)
    preview._motion_request = {"id": "ongoing-grasp"}
    preview._base_request = {"id": "ongoing-base-command"}
    preview._frame = {"sequence": 1, "captured_at": 0}

    state = preview.configure(
        width=preview.settings.width, height=preview.settings.height, fps=preview.settings.fps, camera_name="head_cam"
    )

    assert state["state"] == "ready"
    assert state["settings"]["camera_name"] == "head_cam"
    assert preview._motion_owner == "owner"
    assert preview._motion_binding == ("scene", 0)
    assert commands[-1]["base_motion"] == {"id": "ongoing-grasp"}
    assert commands[-1]["base_command"] == {"id": "ongoing-base-command"}
    assert commands[-1]["config_revision"] == 0
    assert commands[-1]["stream_epoch"] == 1
    assert commands[-1]["keyframe_request"] == 1
    assert preview._frame is None
    # A delayed old status must not restore the previous selected camera.
    assert preview.status()["settings"]["camera_name"] == "head_cam"
    with pytest.raises(RuntimeError, match="控制权"):
        preview.configure(width=1280, height=720, fps=30, camera_name="head_cam")
