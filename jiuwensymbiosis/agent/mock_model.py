# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Offline mock LLM for ``--mock`` runs.

When a demo is driven by hardware-free ``MockArmEnv`` (``--mock``), there is no
real LLM to talk to and the YAML ``model.api_key`` is a placeholder. Building a
real client (``build_model → Model(...) → create_model_client``) then trips
``_validate_config`` on the empty api_key before the agent ever invokes.

This module short-circuits that by handing ``build_robot_agent`` a pre-built
``Model`` whose client never opens a connection. It is the LLM counterpart of
``jiuwensymbiosis.env.mock.MockArmEnv``: a runtime (non-test) mock that lives in
the production package so ``examples/*`` can import it after ``pip install``.

Registration: ``MockModelClient`` sets ``__client_name__ = "mock"`` /
``__client_type__ = "llm"``. ``BaseModelClient.__init_subclass__`` auto-registers
it the moment this module is imported, so ``create_model_client``'s registry
fallback (used for any provider not in ``ProviderType``) picks it up for
``client_provider="mock"``. No monkeypatch, no openjiuwen edit.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Optional, Union

from jiuwensymbiosis.agent.abstractions import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)
from openjiuwen.core.foundation.llm.model_clients.base_model_client import (
    BaseModelClient,
)
from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
from openjiuwen.core.foundation.llm.schema.message_chunk import (
    AssistantMessageChunk,
)

__all__ = ["MockModelClient", "build_mock_model"]

# Fixed assistant reply: the agent loop treats a content-only message as a
# final answer, so a ``--mock`` run with ``--max-iter 1`` exits cleanly after
# one turn without any tool call or network round-trip.
_MOCK_REPLY = "mock: no real model, task skipped"


class MockModelClient(BaseModelClient):
    """A ``BaseModelClient`` that answers from memory — never contacts a server."""

    __client_name__ = "mock"
    __client_type__ = "llm"

    def _validate_config(self) -> None:
        # No api_key / api_base to check — this client is offline.
        return None

    async def invoke(
        self,
        messages: Union[str, List[Any], List[dict]],
        *,
        tools: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[str] = None,
        output_parser: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        return AssistantMessage(content=_MOCK_REPLY)

    async def stream(
        self,
        messages: Union[str, List[Any], List[dict]],
        *,
        tools: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[str] = None,
        output_parser: Optional[Any] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AssistantMessageChunk]:
        # Agent loop uses invoke(); streaming is not exercised under --mock.
        if False:  # pragma: no cover - makes this an empty async generator
            yield AssistantMessageChunk(content="")

    async def generate_image(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("MockModelClient does not generate images")

    async def generate_speech(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("MockModelClient does not generate speech")

    async def generate_video(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("MockModelClient does not generate video")


def build_mock_model() -> Model:
    """Return an offline ``Model`` wired to ``MockModelClient``.

    Importing ``MockModelClient`` (the class reference above) runs
    ``BaseModelClient.__init_subclass__``, which registers ``llm_mock`` in the
    client registry. ``create_model_client`` then resolves
    ``client_provider="mock"`` via the registry fallback rather than failing on
    the unknown provider.
    """
    return Model(
        model_client_config=ModelClientConfig(
            client_provider="mock",
            api_key="mock",
            api_base="mock",
        ),
        model_config=ModelRequestConfig(
            model_name="mock",
            temperature=0.0,
            max_tokens=64,
        ),
    )
