"""GPU worker cancellation, ownership, scene changes and real physical stop."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("JIUWEN_GENESIS_INTEGRATION") != "1", reason="requires Genesis CUDA/Nyx runtime"),
]


def wait_for(check, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for physical worker")


def test_cancel_lease_and_scene_replacement(tmp_path):
    from jiuwensymbiosis.simulators.assets import AssetLibrary
    from jiuwensymbiosis.simulators.genesis.agent import make_session
    from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
    from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

    robot = next(
        a
        for a in AssetLibrary().search(kind="robot", query="moz1").items
        if a.root_id == "spiritai" and a.preview_mode == "urdf"
    )
    preview = GenesisPreview(state_dir=tmp_path, settings=PreviewSettings(scene_id=robot.id, robot_mode="moz1_wheels"))
    evidence = {}
    try:
        preview.start()
        wait_for(lambda: preview.status()["state"] == "ready", 240)
        cancelled = threading.Event()
        session = make_session(preview, cancelled=cancelled)
        session.connect()
        with pytest.raises(RuntimeError, match="另一个"):
            preview.acquire_motion()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(session.api.navigate_relative, 0.8)
            wait_for(lambda: preview.status().get("base_motion", {}).get("status") == "running")
            with pytest.raises(RuntimeError, match="Agent"):
                preview.base_command(
                    vx=0.15, vy=0, wz=0, scene_instance_id=preview.status()["scene"]["instance_id"], config_revision=0
                )
            preview.set_streaming(False)
            time.sleep(0.4)
            assert not future.done(), "video disconnect must not cancel an action"
            cancelled.set()
            result = future.result(timeout=15)
            assert not result["ok"] and result["settled"]
            assert result["reason"] == "cancelled_or_lease_expired"
            evidence["cancel"] = result
        session.disconnect()
        assert session.cleanup_report().released

        # Simulate a crashed producer: one request, no renewal of its lease.
        owner = preview.acquire_motion()
        state = preview.status()
        preview._motion_request = {
            "id": "expired-lease",
            "scene_instance_id": state["scene"]["instance_id"],
            "config_revision": 0,
            "lease_until": time.monotonic() + 0.3,
            "cancelled": False,
            "dx": 0.8,
            "dy": 0,
            "dyaw": 0,
        }
        preview._send_control()
        result = wait_for(
            lambda: (
                (r if r.get("id") == "expired-lease" and r.get("status") == "completed" else None)
                if (r := preview.status().get("base_motion", {}))
                else None
            )
        )
        assert not result["ok"] and result["settled"]
        evidence["lease_expiry"] = result
        preview.release_motion(owner)

        session = make_session(preview)
        session.connect()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(session.api.navigate_relative, 0, 0.8)
            wait_for(lambda: preview.status().get("base_motion", {}).get("status") == "running")
            old_scene = preview.status()["scene"]["instance_id"]
            with pytest.raises(RuntimeError, match="持有控制权"):
                preview.configure(width=1280, height=720, fps=60, scene_id="demo")
            assert preview.status()["scene"]["instance_id"] == old_scene
            evidence["switch_rejected_while_owned"] = True
            session.env.driver.cancelled.set()
            stopped = future.result(timeout=15)
            assert not stopped["ok"] and stopped["settled"]
        session.disconnect()
        assert session.cleanup_report().released

        # Reset must restore the physical checkpoint and motor targets, and
        # publish a new-world frame before declaring completion.
        before_reset = preview.status()
        reset = preview.reset_scene()
        assert reset["reset_result"]["complete"]
        assert reset["reset_result"]["stop_confirmation"] == "measured"
        assert reset["scene"]["instance_id"] != before_reset["scene"]["instance_id"]
        assert reset["pid"] == before_reset["pid"]
        assert reset["robot"]["upper_body_state"]["max_abs_error_rad"] < 0.0174533
        evidence["reset"] = reset["reset_result"]
        evidence["upper_body_after_reset"] = reset["robot"]["upper_body_state"]
        preview.configure(width=1280, height=720, fps=60, scene_id="demo", robot_mode="inspection")
        wait_for(lambda: preview.status()["state"] == "ready", 90)
        assert preview.status()["scene"]["instance_id"] != reset["scene"]["instance_id"]
        with pytest.raises(RuntimeError):
            session.env.get_observation()
        evidence["scene_replaced"] = True
        evidence["passed"] = True
    finally:
        preview.stop()
        out = Path(os.environ.get("JIUWEN_GENESIS_EVIDENCE", tmp_path))
        out.mkdir(parents=True, exist_ok=True)
        (out / "lifecycle.json").write_text(json.dumps(evidence, indent=2))
