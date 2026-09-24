# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Remote failures must stop tracking/motion and keep machine-readable causes."""

import json
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.fast import runner
from jiuwensymbiosis.agent.fast.realtime.servo import ServoResult
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker
from jiuwensymbiosis.agent.fast.runner import _detect_once, direct_executor
from jiuwensymbiosis.agent.fast.sequence import TRACK_DETECT, TRACK_GRASP, ActionStep
from jiuwensymbiosis.errors import InferenceServiceError
from jiuwensymbiosis.motion.approach import coarse_approach, redetect
from jiuwensymbiosis.perception.config import parse_detector_config
from jiuwensymbiosis.perception.detector_client import create_detector_client
from jiuwensymbiosis.perception.vision import default_get_grasp_info_simple
from jiuwensymbiosis.utils.service_http import HttpServiceClient


@pytest.mark.parametrize("op", [TRACK_DETECT, TRACK_GRASP])
def test_late_detection_failure_preserves_motion_error(monkeypatch, op):
    token = CancelToken()
    entered, release = threading.Event(), threading.Event()
    calls, actions = [], []

    def detect(name):
        calls.append(name)
        if len(calls) == 1:
            return {"ok": True, "position": [100.0, 100.0, 100.0], "grasp_z": 80.0}
        entered.set()
        assert release.wait(5)
        raise InferenceServiceError("late detector failure", code="inference_timeout")

    real_stop = BackgroundTracker.stop

    def stop(tracker, **kwargs):
        assert entered.wait(1)
        release.set()
        return real_stop(tracker, **kwargs)

    monkeypatch.setattr(BackgroundTracker, "stop", stop)
    monkeypatch.setattr(runner, "_prescan", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_run_servo_phase",
        lambda *args, **kwargs: ServoResult(
            False, "servo_error", 1, 0.01, None, None, error="bounds rejected", error_code="cartesian_bounds_rejected"
        ),
    )
    session = SimpleNamespace(
        api=SimpleNamespace(
            get_grasp_info_simple=detect,
            get_pose=lambda: {"x": 0.0, "y": 0.0, "z": 100.0, "r": 0.0},
            home=lambda: None,
            open_gripper=lambda: None,
        ),
        env=SimpleNamespace(capabilities=frozenset({"motion.servo"}), holding_payload=False),
        cancel_token=token,
    )
    try:
        result = runner.run_sequence(
            session,
            [ActionStep(op, {"object_name": "box", "approach_mm": 20}), ActionStep("next_motion", {})],
            executor=lambda action, params: actions.append(action) or {"ok": True},
            config=runner.SkillExecConfig(detect_hz=100),
        )
        assert result["ok"] is False
        assert result["steps"][0]["error_code"] == "cartesian_bounds_rejected"
        assert "bounds rejected" in result["steps"][0]["reason"]
        assert actions == []
        assert not token.pending_work
    finally:
        release.set()
        assert token.wait_for_idle(2)


@pytest.mark.parametrize("op,cached", [(TRACK_DETECT, False), (TRACK_DETECT, True), (TRACK_GRASP, False)])
@pytest.mark.parametrize("structured", [False, True])
def test_service_failure_during_tracker_shutdown_aborts_sequence(monkeypatch, op, cached, structured):
    token = CancelToken()
    entered, release = threading.Event(), threading.Event()
    calls, actions = [], []

    def detect(name):
        calls.append(name)
        if op == TRACK_DETECT and len(calls) == 1:  # Real home pre-scan.
            if cached:
                return {"ok": True, "position": [100.0, 100.0, 100.0], "grasp_z": 80.0}
            return {"ok": False, "reason": "no_detection"}
        entered.set()
        assert release.wait(5)
        if structured:
            return {"ok": False, "reason": "detector_unavailable", "error_code": "inference_timeout"}
        raise InferenceServiceError("late inference failure", code="inference_timeout")

    real_stop = BackgroundTracker.stop

    def stop(tracker, **kwargs):
        # Complete the pending inference only after the first-target wait timed out.
        assert entered.wait(1)
        release.set()
        return real_stop(tracker, **kwargs)

    monkeypatch.setattr(BackgroundTracker, "stop", stop)
    session = SimpleNamespace(
        api=SimpleNamespace(
            get_grasp_info_simple=detect,
            get_pose=lambda: {"x": 0.0, "y": 0.0, "z": 100.0, "r": 0.0},
            home=lambda: None,
            open_gripper=lambda: None,
        ),
        env=SimpleNamespace(capabilities=frozenset({"motion.servo"}), holding_payload=False),
        cancel_token=token,
    )
    try:
        result = runner.run_sequence(
            session,
            [
                ActionStep(op, {"object_name": "box", "approach_mm": 20}, bind="box"),
                ActionStep("next_motion", {}),
            ],
            executor=lambda action, params: actions.append(action) or {"ok": True, "result": {"ok": True}},
            config=runner.SkillExecConfig(first_target_timeout_s=0.02),
        )
        assert result["ok"] is False
        assert result["steps"][0]["error_code"] == "inference_timeout"
        assert actions == []
        assert not token.pending_work
    finally:
        release.set()
        assert token.wait_for_idle(2)


def test_successful_tracking_waits_for_healthy_slow_detection(monkeypatch):
    token = CancelToken()
    motions, calls = [], []

    def detect(name):
        calls.append(name)
        time.sleep(2.3)  # Within the 8-second tracking budget, beyond the old 2-second join.
        return {"ok": True, "position": [100.0, 100.0, 100.0], "grasp_z": 80.0}

    session = SimpleNamespace(
        api=SimpleNamespace(
            get_grasp_info_simple=detect,
            get_pose=lambda: {"x": 0.0, "y": 0.0, "z": 100.0, "r": 0.0},
            home=lambda: motions.append("home"),
            open_gripper=lambda: motions.append("release"),
        ),
        env=SimpleNamespace(capabilities=frozenset({"motion.servo"}), holding_payload=False),
        cancel_token=token,
    )
    monkeypatch.setattr(runner, "_prescan", lambda *args: {})
    try:
        result = runner.run_sequence(
            session,
            [ActionStep(TRACK_DETECT, {"object_name": "box"}, bind="box")],
            executor=lambda *args: {"ok": True},
        )
        assert len(calls) >= 2
        assert result["ok"] is True
        assert motions == []
        assert not token.pending_work
        assert not token.is_set()
    finally:
        assert token.wait_for_idle(4)


@pytest.mark.parametrize("op", [TRACK_DETECT, TRACK_GRASP])
def test_cancelled_tracking_never_recovers_while_capture_is_pending(monkeypatch, op):
    token = CancelToken()
    release = threading.Event()
    motions = []
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://unused"}}})
    )
    client.bind_cancel_token(token)

    def grab_frames():
        token.set()
        release.wait(10)
        return np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10))

    env = SimpleNamespace(low_level=SimpleNamespace(grab_frames=grab_frames), holding_payload=False)
    api = SimpleNamespace(
        env=env,
        open_gripper=lambda: motions.append("open_gripper"),
        home=lambda: motions.append("home"),
    )
    api.get_grasp_info_simple = lambda name: default_get_grasp_info_simple(
        api, name, seg_fn=client, pose_to_tf=lambda pose: np.eye(4)
    )
    session = SimpleNamespace(api=api, env=env, cancel_token=token)
    monkeypatch.setattr(runner, "_prescan", lambda *args: {})
    monkeypatch.setattr(
        runner, "ServoBinding", lambda session: SimpleNamespace(read_pose=lambda: {"x": 0, "y": 0, "z": 0})
    )
    try:
        with pytest.raises(RunCancelled):
            runner.run_sequence(
                session,
                [ActionStep(op, {"object_name": "box", "approach_mm": 20})],
                executor=lambda *args: {"ok": True},
            )
        assert motions == []
        assert token.pending_work  # The blocked capture must still prevent resource release.
    finally:
        release.set()
        assert token.wait_for_idle(2)
        client.close()


def test_action_failure_concurrent_with_cancellation_never_recovers():
    token = CancelToken()
    motions = []
    session = SimpleNamespace(
        api=SimpleNamespace(home=lambda: motions.append("home"), open_gripper=lambda: motions.append("open_gripper")),
        env=SimpleNamespace(holding_payload=False),
        cancel_token=token,
    )

    def fail_after_cancel(*args):
        token.set()
        raise RuntimeError("operation interrupted")

    with pytest.raises(RunCancelled):
        runner.run_sequence(session, [ActionStep("action", {})], executor=fail_after_cancel)
    assert motions == []


def test_tracker_does_not_turn_structured_service_error_into_a_miss():
    api = SimpleNamespace(
        get_grasp_info_simple=lambda obj: {
            "ok": False,
            "reason": "detector_unavailable",
            "error_code": "inference_timeout",
        }
    )
    with pytest.raises(InferenceServiceError) as exc:
        _detect_once(api, "box")
    assert exc.value.code == "inference_timeout"


@pytest.mark.parametrize("field", ["box", "score"])
def test_tracker_does_not_turn_overflowing_http_response_into_a_miss(field):
    client = create_detector_client(
        parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://vision"}}})
    )

    def handle(request):
        item = {"box": [0, 0, 10, 10], "score": 0.9, "label": "box"}
        if field == "box":
            item["box"][2] = 10**400
        else:
            item["score"] = 10**400
        payload = json.loads(request.content)
        response = {
            "schema_version": 1,
            "request_id": payload["request_id"],
            "result": {"frame_id": payload["frame_id"], "width": 10, "height": 10, "detections": [item]},
        }
        return httpx.Response(200, json=response)

    client._http = HttpServiceClient(client.endpoint, service="vision", transport=httpx.MockTransport(handle))
    api = SimpleNamespace(get_grasp_info_simple=lambda name: client(np.zeros((10, 10, 3), dtype=np.uint8), name))
    try:
        with pytest.raises(InferenceServiceError) as exc:
            _detect_once(api, "box")
        assert exc.value.code == "inference_protocol_error"
    finally:
        client.close()


def test_grounded_service_failure_never_retries_ungrounded():
    calls = []

    def detect(*args, **kwargs):
        calls.append(kwargs)
        return {"ok": False, "reason": "detector_unavailable", "error_code": "inference_unavailable"}

    with pytest.raises(InferenceServiceError):
        redetect(SimpleNamespace(locate_for_grasp=detect), "box", "table")
    assert len(calls) == 1


def test_active_approach_drive_stops_on_remote_failure():
    stops = []
    driver = SimpleNamespace(
        start_base_drive=lambda: "drive",
        base_drive_running=lambda handle: True,
        stop_base_drive=lambda handle: stops.append(handle),
    )
    api = SimpleNamespace(env=driver, rotate_base=lambda angle: {"ok": True})

    def unavailable(obj):
        return {"ok": False, "reason": "detector_unavailable", "error_code": "inference_timeout"}

    with pytest.raises(InferenceServiceError):
        coarse_approach(api, unavailable, "box", 0.0)
    assert stops == ["drive"]


def test_direct_executor_preserves_cancellation():
    def cancelled():
        raise RunCancelled

    with pytest.raises(RunCancelled):
        direct_executor({"detect": cancelled})("detect", {})
