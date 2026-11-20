"""Thread ownership, completed-state lifetime and clock coherence without GPUs."""

import threading
import weakref
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.simulators.common.independent_video import CompletedFrame, CompletedStates, IndependentVideoLoop
from jiuwensymbiosis.simulators.common.split_threads import run_split


def frame(**overrides):
    values = {
        "geometry": object(),
        "step": 42,
        "dt": 1 / 120,
        "completed_at": 9.0,
        "ready_at": 9.01,
        "scene_instance_id": "scene",
        "config_revision": 1,
        "stream_epoch": 2,
        "reset_id": None,
        "reset_generation": 0,
        "camera_revision": 3,
    }
    return CompletedFrame(**(values | overrides))


def test_exchange_drops_backlog_but_keeps_in_use_state_alive():
    class Geometry:
        pass

    states = CompletedStates()
    states.publish(frame(geometry=Geometry()))
    in_use, _ = states.read()
    held = weakref.ref(in_use.geometry)
    states.publish(frame(geometry=Geometry(), step=43))
    discarded = weakref.ref(states.read()[0].geometry)
    states.publish(frame(geometry=Geometry(), step=44))
    assert discarded() is None
    assert held() is not None
    assert states.read()[0].step == 44
    del in_use
    assert held() is None
    with pytest.raises(ValueError, match="regressed"):
        states.publish(frame(step=0))
    states.clear()
    states.publish(frame(step=0, reset_generation=1))
    assert states.read()[0].sim_time == 0


def harness():
    clock = SimpleNamespace(now=10.0)
    command = {
        "config_revision": 1,
        "stream_epoch": 2,
        "streaming": True,
        "keyframe_request": 0,
        "camera": {"revision": 3},
    }
    packets, threads = [], []
    done = threading.Event()

    def render(state, **kwargs):
        threads.append(("render", threading.get_ident()))
        return state

    def encode(rgb, metadata, **kwargs):
        threads.append(("readback_encode", threading.get_ident()))
        clock.now += 0.04
        return [{**metadata, "h264": b"encoded", "keyframe": True}]

    def write(metadata, payload):
        threads.append(("publish", threading.get_ident()))
        packets.append(metadata)
        if len(packets) >= 3:
            done.set()

    return SimpleNamespace(
        clock=clock,
        command=command,
        packets=packets,
        threads=threads,
        done=done,
        mailbox=SimpleNamespace(
            stopping=lambda: False,
            read_control=lambda: command.copy(),
            acknowledged=lambda: 100000,
            write_frame=write,
        ),
        renderer=SimpleNamespace(render_completed=render, output_context=nullcontext),
        encoder=SimpleNamespace(encode=encode, copy_ms=1.0, encode_ms=2.0, close=lambda: None),
        settings=SimpleNamespace(fps=30, camera_name="observer"),
    )


def test_readback_encoding_and_publication_continue_without_physics_and_keep_sim_clock():
    h = harness()
    main = threading.get_ident()
    stats = {}

    def physical(dispatch):
        loop = dispatch.call(
            IndependentVideoLoop,
            dispatch,
            h.mailbox,
            h.renderer,
            h.encoder,
            h.settings,
            monotonic=lambda: h.clock.now,
        )
        try:
            loop.states.publish(frame())
            assert h.done.wait(1), "Readback/encoding stalled with the physics owner"
            dispatch.check_error()
            stats.update(loop.snapshot())
        finally:
            loop.close()

    run_split(physical)
    assert {t for _, t in h.threads} == {main}
    assert all(p["sim_time"] == 42 / 120 and p["step"] == 42 for p in h.packets)
    assert len({p["captured_at"] for p in h.packets}) == len(h.packets)
    assert all(p["state_age_ms"] >= 1000 for p in h.packets)
    assert stats["video_schedule"]["fresh_state_fps"] == 0
    assert stats["video_schedule"]["recent_source_fps"] > 0


def test_configuration_change_during_encoding_cannot_publish_old_frame():
    h = harness()
    dispatch = SimpleNamespace(set_tick=lambda _: None, frame_sequence=0)
    loop = IndependentVideoLoop(dispatch, h.mailbox, h.renderer, h.encoder, h.settings, monotonic=lambda: h.clock.now)
    original = h.encoder.encode

    def encode(*args, **kwargs):
        result = original(*args, **kwargs)
        h.command["config_revision"] = 2
        return result

    h.encoder.encode = encode
    loop.states.publish(frame())
    loop.tick()
    assert not h.packets


def test_state_identity_includes_reset_and_stream_epochs():
    h = harness()
    state = frame()
    assert state.matches(h.command)
    assert not replace(state, stream_epoch=0).matches(h.command)
    assert not replace(state, config_revision=0).matches(h.command)
    assert not replace(state, reset_id="new").matches(h.command)
