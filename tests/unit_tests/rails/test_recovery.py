# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.rails.recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jiuwensymbiosis.rails.recovery import RecoveryRail, recover_session
from tests.helpers import FakeCtx, RecordingRailSink, make_mock_session


@pytest.fixture
def mock_session():
    return make_mock_session()


class TestIsWatched:
    @pytest.mark.parametrize("tool_name", ["home", "close_gripper"], ids=["motion", "grasp"])
    def test_default_tags_watched(self, mock_session, tool_name):
        rail = RecoveryRail(mock_session)
        assert rail._is_watched(tool_name, tool_args=None) is True

    def test_other_tool_not_watched_without_tags(self, mock_session):
        rail = RecoveryRail(mock_session)
        assert rail._is_watched("get_pose", tool_args=None) is False


class TestRecoveryRail:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_args", "expected_call"),
        [
            ("close_gripper", {}, "home"),
            ("goto_xyzr", {"x": 100, "y": 0, "z": 300, "r": 0}, "home"),
        ],
        ids=["grasp", "motion"],
    )
    async def test_recovery_on_watched_exception(self, mock_session, tool_name, tool_args, expected_call):
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args)
        await rail.on_tool_exception(ctx)
        assert any(expected_call in c for c in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_typed_safe_failure_skips_recovery(self, mock_session):
        class SafeFailure(RuntimeError):
            skip_recovery = True

        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name="goto_xyzr", exception=SafeFailure("not reached"))
        await rail.on_tool_exception(ctx)
        assert mock_session.api._call_log == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("goto_xyzr", {"x": 100, "y": 0, "z": 300}),
            ("robot_control", {"action": "goto_xyzr", "params": {"x": 100, "y": 0, "z": 300}}),
        ],
        ids=["direct", "robot-control"],
    )
    async def test_wrapped_typed_safe_failure_skips_recovery(self, mock_session, tool_name, tool_args):
        class SafeFailure(RuntimeError):
            skip_recovery = True

        original = SafeFailure("IK rejected before dispatch")
        wrapped = RuntimeError("AbilityExecutionError")
        wrapped.__cause__ = original
        wrapped.cause = original
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args, exception=wrapped)

        await rail.on_tool_exception(ctx)

        assert mock_session.api._call_log == []

    @pytest.mark.asyncio
    async def test_implicit_exception_context_does_not_skip_real_failure_recovery(self, mock_session):
        class SafeFailure(RuntimeError):
            skip_recovery = True

        real_failure = None
        try:
            try:
                raise SafeFailure("pre-dispatch request rejected")
            except SafeFailure:
                # Deliberately implicit: this regression test proves that a
                # prior safe error in __context__ cannot suppress recovery.
                raise RuntimeError("transport failed while handling rejection")  # noqa: B904
        except RuntimeError as exc:
            real_failure = exc
            assert isinstance(exc.__context__, SafeFailure)

        rail = RecoveryRail(mock_session)
        assert real_failure is not None
        ctx = FakeCtx(tool_name="goto_xyzr", exception=real_failure)
        await rail.on_tool_exception(ctx)

        assert any(call == "home" for call in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_adapter_recovery_home_is_preferred(self, mock_session):
        mock_session.api.recovery_home = lambda: mock_session.api._call_log.append("recovery_home")
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 300})
        await rail.on_tool_exception(ctx)
        assert "recovery_home" in mock_session.api._call_log
        assert not any(call == "home" for call in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_motion_failure_preserves_confirmed_payload_during_home(self, mock_session):
        mock_session.env.holding_payload = True
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 300})

        await rail.on_tool_exception(ctx)

        assert any(call == "home" for call in mock_session.api._call_log)
        assert not any(call in {"deactivate_suction", "open_gripper"} for call in mock_session.api._call_log)

    @pytest.mark.asyncio
    async def test_reported_empty_payload_still_releases_before_home(self, mock_session):
        mock_session.env.holding_payload = False
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 100, "y": 0, "z": 300})

        await rail.on_tool_exception(ctx)

        assert any(call == "home" for call in mock_session.api._call_log)
        assert any(call in {"deactivate_suction", "open_gripper"} for call in mock_session.api._call_log)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "tool_args"),
        [
            ("home", {}),
            ("robot_control", {"action": "home", "params": {}}),
        ],
        ids=["direct", "robot-control"],
    )
    async def test_failed_home_releases_but_does_not_retry_home(self, mock_session, tool_name, tool_args):
        rail = RecoveryRail(mock_session)
        ctx = FakeCtx(tool_name=tool_name, tool_args=tool_args)

        await rail.on_tool_exception(ctx)

        assert any(call in {"deactivate_suction", "open_gripper"} for call in mock_session.api._call_log)
        assert not any(call == "home" for call in mock_session.api._call_log)


class TestReleaseEffectorHook:
    """A body whose end effector is neither suction nor a parallel gripper."""

    @staticmethod
    def _session(calls, *, release_effector):
        class _Api:
            def home(self):
                calls.append("home")

        class _Env:
            def set_end_effector(self, closed):
                calls.append(f"set_end_effector({closed})")

        api = _Api()
        if release_effector is not None:
            api.release_effector = release_effector
        return SimpleNamespace(api=api, env=_Env())

    def test_release_effector_is_tried_before_the_env_fallback(self):
        calls = []
        session = self._session(calls, release_effector=lambda: calls.append("release_effector"))

        released_ok, home_ok = recover_session(session, release=True, home=True, log_prefix="test")

        assert (released_ok, home_ok) == (True, True)
        assert calls == ["release_effector", "home"]

    def test_env_fallback_still_runs_when_release_effector_fails(self):
        calls = []

        def _boom():
            calls.append("release_effector")
            raise RuntimeError("paddle servo offline")

        session = self._session(calls, release_effector=_boom)

        released_ok, _ = recover_session(session, release=True, home=True, log_prefix="test")

        assert released_ok is True
        assert calls == ["release_effector", "set_end_effector(False)", "home"]


class TestRecoveryRailTraceSink:
    @pytest.mark.asyncio
    async def test_recover_notifies_sink_with_result(self, mock_session):
        sink = RecordingRailSink()
        rail = RecoveryRail(mock_session, trace_sink=sink)
        ctx = FakeCtx(tool_name="goto_xyzr", tool_args={"x": 1, "y": 2, "z": 3})
        await rail.on_tool_exception(ctx)
        assert sink.events
        detail = sink.events[0][2]
        assert "home_ok" in detail
        assert detail["holding_payload"] is None
        assert detail["payload_preserved"] is False
        assert sink.events[0][3] is True  # mock home succeeds
