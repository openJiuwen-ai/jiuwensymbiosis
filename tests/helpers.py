# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared test doubles for agent/rail lifecycle tests."""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


class FakeInputs:
    """Small stand-in for openjiuwen callback input objects."""

    def __init__(
        self,
        *,
        tool_name: str = "",
        tool_args: Any = None,
        tool_result: Any = None,
        conversation_id: str = "",
        query: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = {} if tool_args is None else tool_args
        self.tool_result = tool_result
        self.tool_msg = None
        self.conversation_id = conversation_id
        self.query = query


class FakeCtx:
    """Minimal AgentCallbackContext shape used by rails."""

    def __init__(
        self,
        inputs: FakeInputs | None = None,
        *,
        tool_name: str = "",
        tool_args: Any = None,
        tool_result: Any = None,
        conversation_id: str = "",
        query: str | None = None,
        context: Any = None,
        extra: dict[str, Any] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.inputs = inputs or FakeInputs(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            conversation_id=conversation_id,
            query=query,
        )
        self.context = context
        self.extra = {} if extra is None else extra
        self.exception = exception


class RecordingModelContext:
    """Stand-in for openjiuwen SessionModelContext."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    async def add_messages(self, message: Any) -> list[Any]:
        self.added.append(message)
        return self.added


class RecordingRailSink:
    """Trace sink double for tests that only need recorded rail events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, Any, bool]] = []

    def record_rail_event(self, *, rail_name: str, kind: str, detail: Any, success: bool) -> None:
        self.events.append((rail_name, kind, detail, success))


def make_mock_session(
    *,
    name: str = "test",
    env: MockArmEnv | None = None,
    api: MockApi | None = None,
    api_kwargs: dict[str, Any] | None = None,
    **session_kwargs: Any,
) -> RobotSession:
    env = env or MockArmEnv()
    api = api or MockApi(env, **(api_kwargs or {}))
    return RobotSession(env=env, api=api, name=name, **session_kwargs)
