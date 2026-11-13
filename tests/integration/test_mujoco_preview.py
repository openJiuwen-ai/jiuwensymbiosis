"""Opt-in real MuJoCo 4 physics/cameras and sequential backend ownership."""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("JIUWEN_MUJOCO_INTEGRATION") != "1", reason="Opt-in native OpenGL acceptance"),
]


def ready(preview):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = preview.status()
        assert state["state"] != "failed", state.get("message")
        if state["state"] == "ready":
            return state
        time.sleep(0.1)
    raise AssertionError(f"Backend did not become ready: {preview.status()}")


def test_native_demo_mjcf_cameras_and_reset(tmp_path, monkeypatch):
    import yaml

    from jiuwensymbiosis.simulators.assets import build_index, index, load_config
    from jiuwensymbiosis.simulators.common.camera import PreviewSettings
    from jiuwensymbiosis.simulators.common.preview import SimulationPreview
    from jiuwensymbiosis.simulators.mujoco.scene_runtime import DEMO_XML

    evidence = Path(os.environ.get("JIUWEN_MUJOCO_EVIDENCE", tmp_path / "evidence"))
    evidence.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JIUWENSYMBIOSIS_SIMULATOR_HOME", str(tmp_path))
    catalog = tmp_path / "outputs/simulator-assets/index.sqlite"
    monkeypatch.setattr(index, "DEFAULT_INDEX_PATH", catalog)
    (tmp_path / "scene.xml").write_text(DEMO_XML)
    config = tmp_path / "assets.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "index": str(catalog),
                "roots": [{"id": "local", "scanner": "mjcf", "path": str(tmp_path), "entries": ["scene.xml"]}],
            }
        )
    )
    build_index(load_config(config))
    preview = SimulationPreview(
        state_dir=tmp_path / "preview", settings=PreviewSettings(backend="mujoco", width=960, height=540, fps=30)
    )
    try:
        preview.start()
        state = ready(preview)
        assert state["backend"] == "mujoco" and state["robot"] is None
        for name in ("observer", "scene_cam"):
            image = preview.capture_image(name)
            (evidence / f"{name}.jpg").write_bytes(image["jpeg"])
        before = state["scene"]["instance_id"]
        preview.reset_scene()
        state = ready(preview)
        assert state["scene"]["instance_id"] != before
        preview.configure(backend="mujoco", width=960, height=540, fps=30, scene_id="library://local/scene")
        state = ready(preview)
        assert state["scene"]["id"] == "library://local/scene" and state["robot"] is None
        (evidence / "demo-mjcf.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))
    finally:
        preview.stop()


@pytest.mark.skipif(importlib.util.find_spec("genesis") is None, reason="Dual backend environment required")
def test_backend_switch_restarts_worker_and_revokes_old_scene(tmp_path):
    from jiuwensymbiosis.simulators.common.camera import PreviewSettings
    from jiuwensymbiosis.simulators.common.preview import SimulationPreview

    preview = SimulationPreview(
        state_dir=tmp_path, settings=PreviewSettings(backend="mujoco", width=960, height=540, fps=30)
    )
    try:
        preview.start()
        previous = ready(preview)
        for backend in ("genesis", "mujoco"):
            preview.configure(backend=backend, width=960, height=540, fps=30)
            state = ready(preview)
            assert state["pid"] != previous["pid"] and state["backend"] == backend
            assert state["scene"]["instance_id"] != previous["scene"]["instance_id"]
            with pytest.raises(ValueError, match="场景已经切换"):
                preview.select(
                    action="clear",
                    u=0,
                    v=0,
                    scene_instance_id=previous["scene"]["instance_id"],
                    config_revision=previous["config_revision"],
                    camera_revision=previous["camera"]["revision"],
                )
            previous = state
    finally:
        preview.stop()
