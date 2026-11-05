"""Slow RGB-D sampling refreshes measurements without advancing physics."""

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.simulators.genesis.capture_status import refresh_capture_status


@pytest.mark.parametrize("scene", ["current", "old"])
def test_status_refresh_requires_current_scene_and_remeasures(scene):
    state = {"state": "ready", "scene": {"instance_id": scene}, "measured_at": 0}
    published = []
    mailbox = SimpleNamespace(read_status=lambda: state.copy(), write_status=published.append)
    runtime = SimpleNamespace(
        info={"instance_id": "current"},
        robot=SimpleNamespace(snapshot=lambda: {"pose": "measured"}),
        indoor=None,
        bodies=[],
        motion=SimpleNamespace(result={"status": "running"}),
        steps=10,
        settings=SimpleNamespace(dt=0.01),
    )
    updated = refresh_capture_status(runtime, mailbox, measure_base=lambda robot: {"x": 1}, now=lambda: 5)
    assert updated is (scene == "current")
    if updated:
        assert published[0]["base_observation"] == {"x": 1}
        assert published[0]["robot"] == {"pose": "measured"}
        assert published[0]["measured_at"] == 5 and published[0]["steps"] == 10
    else:
        assert published == []
