"""The renderer stays on main while the sole physics owner can progress."""

import threading
import weakref

import pytest

from jiuwensymbiosis.simulators.common.split_threads import run_split


def test_main_thread_render_does_not_block_physics_owner():
    main = threading.get_ident()
    entered = threading.Event()
    advanced = threading.Event()
    observed = {}

    def physical(dispatch):
        observed["physical"] = threading.get_ident()

        def render():
            assert threading.get_ident() == main
            entered.set()
            assert advanced.wait(1)
            return "image"

        frame = dispatch.submit(render)
        assert entered.wait(1)
        dispatch.physics_steps += 3
        advanced.set()
        assert frame.result(timeout=1) == "image"
        assert dispatch.call(threading.get_ident) == main
        observed.update(dispatch.snapshot())

    run_split(physical)
    assert observed["physical"] != main
    assert observed["physics_thread_id"] != observed["render_thread_id"]


def test_render_failure_reaches_owner_and_cleanup_stays_on_main():
    main = threading.get_ident()
    closed = []

    def physical(dispatch):
        try:
            dispatch.call(lambda: (_ for _ in ()).throw(ValueError("render failed")))
        finally:
            dispatch.call(lambda: closed.append(threading.get_ident()))

    with pytest.raises(ValueError, match="render failed"):
        run_split(physical)
    assert closed == [main]


def test_only_one_waiting_render_command_is_allowed():
    # A blocked main-thread command cannot accumulate unbounded future work.
    entered = threading.Event()
    release = threading.Event()

    def physical(dispatch):
        first = dispatch.submit(lambda: (entered.set(), release.wait(1)))
        assert entered.wait(1)
        second = dispatch.submit(lambda: None)
        with pytest.raises(RuntimeError, match="queue is full"):
            dispatch.submit(lambda: None)
        release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    run_split(physical)


def test_completed_image_is_released_before_renderer_teardown():
    class Image:
        pass

    def physical(dispatch):
        future = dispatch.submit(Image)
        image = future.result(timeout=1)
        reference = weakref.ref(image)
        del image, future
        dispatch.call(lambda: None)
        assert reference() is None

    run_split(physical)


def test_render_loop_ticks_while_physics_is_blocked_without_submitting_frames():
    """A slow physics step must not prevent the independent renderer ticking."""
    rendered = threading.Event()
    calls = []
    main = threading.get_ident()

    def physical(dispatch):
        def tick():
            calls.append(threading.get_ident())
            if len(calls) >= 3:
                rendered.set()
            return 0.001

        dispatch.call(dispatch.set_tick, tick)
        try:
            # Model a blocking SDK step: no per-frame submissions from owner.
            assert rendered.wait(1), "Rendering waited for the physics owner"
        finally:
            dispatch.call(dispatch.set_tick, None)

    run_split(physical)
    assert len(calls) >= 3
    assert set(calls) == {main}


def test_independent_render_failure_reaches_physics_owner():
    failed = threading.Event()

    def physical(dispatch):
        def tick():
            failed.set()
            raise ValueError("independent render failed")

        dispatch.call(dispatch.set_tick, tick)
        assert failed.wait(1)
        dispatch.call(lambda: None)
        dispatch.check_error()

    with pytest.raises(ValueError, match="independent render failed"):
        run_split(physical)
