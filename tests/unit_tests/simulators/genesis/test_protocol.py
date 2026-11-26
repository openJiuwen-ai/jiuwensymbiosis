# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shared wire contract keeps one field definition for parent and worker."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jiuwensymbiosis.simulators.common import protocol


def _capture(**overrides):
    values = {
        "request_id": "capture-1",
        "scene_instance_id": "scene",
        "config_revision": 3,
        "camera_revision": 7,
        "camera_name": "observer",
        "include_depth": False,
        "depth_stride": 8,
        "expires_at": 12.5,
    }
    values.update(overrides)
    return protocol.build_capture_request(values.pop("request_id"), **values)


def test_capture_request_wire_shape_and_documented_defaults():
    request = _capture()
    assert request == {
        "id": "capture-1",
        "scene_instance_id": "scene",
        "config_revision": 3,
        "camera_revision": 7,
        "camera_name": "observer",
        "include_depth": False,
        "depth_stride": 8,
        "expires_at": 12.5,
    }
    minimal = {
        "id": "capture-1",
        "scene_instance_id": "scene",
        "config_revision": 3,
        "camera_revision": 7,
        "expires_at": 12.5,
    }
    assert protocol.parse_capture_request(minimal) == {**minimal, **request}
    assert protocol.parse_capture_request({**request, "future_field": 1}) == request
    assert json.loads(json.dumps(request)) == request


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"config_revision": -1}, "config_revision"),
        ({"expires_at": float("nan")}, "expires_at"),
        ({"depth_stride": 2, "include_depth": True, "camera_name": "wrist"}, "stride"),
        ({"camera_name": "observer", "include_depth": True}, "robot camera"),
    ],
)
def test_capture_construction_rejects_invalid_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        _capture(**overrides)


@pytest.mark.parametrize(
    "overrides, invalid",
    [
        ({"scene_instance_id": "other"}, True),
        ({"camera_revision": 6}, True),
        ({"camera_name": "wrist", "camera_revision": 6}, False),
        ({"expires_at": 1.0}, True),
    ],
)
def test_capture_request_invalid_covers_binding_view_and_deadline(overrides, invalid):
    request = _capture(**overrides)
    assert (
        protocol.capture_request_invalid(
            request,
            config_revision=3,
            scene_instance_id="scene",
            camera_revision=7,
            now=10.0,
        )
        is invalid
    )


def test_capture_reply_shape_and_freshness():
    reply = protocol.build_capture_reply(
        "capture-1",
        captured_at=11.0,
        captured_utc="2026-01-01T00:00:00.000+00:00",
        scene_instance_id="scene",
        config_revision=3,
        camera_revision=7,
        camera_name="observer",
        camera={"pos": [0, 0, 0]},
        step=9,
        width=1280,
        height=720,
        capture_duration_ms=3.5,
    )
    assert not {"base_sample", "body_bounds_m", "motion_id"} & reply.keys()
    request = _capture()
    assert protocol.capture_reply_matches(reply, request, now=12.0)
    assert not protocol.capture_reply_matches({**reply, "camera_revision": 6}, request, now=12.0)
    assert not protocol.capture_reply_matches({**reply, "captured_at": 9.0}, request, now=12.0)

    rgbd = protocol.build_capture_reply(
        "capture-1",
        captured_at=11.0,
        captured_utc="2026-01-01T00:00:00.000+00:00",
        scene_instance_id="scene",
        config_revision=3,
        camera_revision=7,
        camera_name="wrist",
        camera={"pos": [0, 0, 0]},
        step=9,
        width=1280,
        height=720,
        capture_duration_ms=3.5,
        base_sample={"position_m": [0, 0, 0]},
        body_bounds_m=[[-1, -1, 0], [1, 1, 1]],
        motion_id="motion-1",
    )
    assert rgbd["base_sample"] == {"position_m": [0, 0, 0]}
    assert rgbd["body_bounds_m"] == [[-1, -1, 0], [1, 1, 1]]
    assert rgbd["motion_id"] == "motion-1"


def test_pick_request_round_trip_and_binding_reasons():
    request = protocol.build_pick_request(
        "pick-1",
        action="pick",
        u=0.25,
        v=0.75,
        scene_instance_id="scene",
        config_revision=3,
        camera_revision=7,
    )
    assert protocol.parse_pick_request(request) == request
    assert protocol.pick_binding_error(request, config_revision=3, scene_instance_id="scene", camera_revision=7) is None
    assert (
        protocol.pick_binding_error(request, config_revision=4, scene_instance_id="scene", camera_revision=7) == "scene"
    )
    assert (
        protocol.pick_binding_error(request, config_revision=3, scene_instance_id="scene", camera_revision=8)
        == "camera"
    )
    clear = {**request, "action": "clear"}
    assert protocol.pick_binding_error(clear, config_revision=3, scene_instance_id="scene", camera_revision=8) is None
    with pytest.raises(ValueError, match="within"):
        protocol.build_pick_request(
            "pick-2",
            action="pick",
            u=1.5,
            v=0.0,
            scene_instance_id="scene",
            config_revision=0,
            camera_revision=0,
        )


def test_base_command_validity_requires_streaming_live_binding():
    request = protocol.build_base_command_request(
        "pulse",
        vx=0.1,
        vy=0.0,
        wz=0.0,
        scene_instance_id="scene",
        config_revision=3,
        expires_at=10.0,
    )
    assert protocol.base_command_valid(request, config_revision=3, scene_instance_id="scene", streaming=True, now=9.0)
    assert not protocol.base_command_valid(
        request, config_revision=3, scene_instance_id="scene", streaming=False, now=9.0
    )
    assert not protocol.base_command_valid(
        request, config_revision=3, scene_instance_id="scene", streaming=True, now=10.5
    )
    assert not protocol.base_command_valid({}, config_revision=3, scene_instance_id="scene", streaming=True, now=9.0)
    assert protocol.build_base_command_reply("pulse", applied=False, error="nope") == {
        "id": "pulse",
        "applied": False,
        "error": "nope",
    }


def test_base_motion_envelope_round_trip_and_rejections():
    request = protocol.build_base_motion_request(
        "motion-1",
        scene_instance_id="scene",
        config_revision=3,
        lease_until=10.0,
        kind="arm",
        targets=[0.1] * 7,
    )
    parsed = protocol.parse_base_motion_request({**request, "future": "payload"})
    assert parsed["kind"] == "arm" and parsed["future"] == "payload" and parsed["cancelled"] is False
    assert protocol.base_motion_binding(parsed) == ("scene", 3)
    with pytest.raises(ValueError, match="redefine"):
        protocol.build_base_motion_request(
            "motion-2",
            scene_instance_id="scene",
            config_revision=3,
            lease_until=10.0,
            id="override",
        )
    for broken in (
        {},
        [],
        {**request, "cancelled": 1},
        {**request, "lease_until": "soon"},
        {**request, "id": ""},
        {**request, "config_revision": -1},
    ):
        with pytest.raises(ValueError):
            protocol.parse_base_motion_request(broken)


def test_scene_query_and_reset_round_trip():
    query = protocol.build_scene_query_request(
        "query",
        operation="objects",
        params={"kind": "cup"},
        scene_instance_id="scene",
        config_revision=3,
        expires_at=20.0,
    )
    assert protocol.parse_scene_query_request(query) == query
    assert protocol.binding_mismatch(query, config_revision=4, scene_instance_id="scene")
    assert not protocol.binding_mismatch(query, config_revision=3, scene_instance_id="scene")
    reset = protocol.build_reset_request("reset", scene_instance_id="scene", config_revision=3)
    assert protocol.parse_reset_request(reset) == reset


def test_control_message_slots_are_centralised():
    payload = {
        "streaming": True,
        "camera": {"pos": [0, 0, 0]},
        "settings": {"fps": 30},
        "config_revision": 1,
        "stream_epoch": 2,
        "keyframe_request": 3,
        "pick": {"id": "pick"},
        "base_command": {"id": "base"},
        "base_motion": {"id": "motion"},
        "capture": {"id": "capture"},
        "scene_query": {"id": "query"},
        "reset": {"id": "reset"},
    }
    message = protocol.build_control_message(**payload)
    assert tuple(message) == (
        "streaming",
        "camera",
        "settings",
        "config_revision",
        "stream_epoch",
        "keyframe_request",
        "pick",
        "base_command",
        "base_motion",
        "capture",
        "scene_query",
        "reset",
    )
    assert message == payload


def test_protocol_import_is_standalone_and_stdlib_only():
    common = Path(protocol.__file__).parent
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(common)!r})\n"
        "import protocol\n"
        "loaded = sorted(name for name in sys.modules if name.startswith(('openjiuwen', 'jiuwensymbiosis')))\n"
        "assert not loaded, loaded\n"
        "assert protocol.parse_capture_request({'id':'c','scene_instance_id':'s','config_revision':0,"
        "'camera_revision':0,'expires_at':1.0})['camera_name'] == 'observer'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_parsers_reject_malformed_messages_and_predicates_tolerate_missing_fields():
    parsers = (
        protocol.parse_capture_request,
        protocol.parse_pick_request,
        protocol.parse_base_command_request,
        protocol.parse_scene_query_request,
        protocol.parse_reset_request,
        protocol.parse_base_motion_request,
    )
    for parser in parsers:
        for broken in ({}, [], "not-an-object", None):
            with pytest.raises(ValueError):
                parser(broken)
    assert protocol.capture_request_invalid(
        {}, config_revision=0, scene_instance_id="scene", camera_revision=0, now=1.0
    )
    assert protocol.pick_binding_error({}, config_revision=0, scene_instance_id="scene", camera_revision=0) == "scene"
    assert not protocol.binding_mismatch(
        {"scene_instance_id": "scene", "config_revision": 0}, config_revision=0, scene_instance_id="scene"
    )
