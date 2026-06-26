# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.mock_model."""

from __future__ import annotations

import asyncio

from jiuwensymbiosis.agent.mock_model import MockModelClient, build_mock_model


class TestBuildMockModel:
    def test_returns_model_without_raising(self):
        # The whole point: building a mock model must NOT trip _validate_config
        # on a placeholder api_key the way build_model() does.
        from jiuwensymbiosis.agent.abstractions import Model

        m = build_mock_model()
        assert isinstance(m, Model)

    def test_registered_as_llm_mock(self):
        # MockModelClient must auto-register so create_model_client's registry
        # fallback resolves client_provider="mock".
        from openjiuwen.core.common.clients import get_client_registry

        assert "llm_mock" in get_client_registry().list_clients()

    def test_client_is_mock(self):
        m = build_mock_model()
        assert isinstance(m._client, MockModelClient)

    def test_invoke_returns_assistant_message(self):
        from openjiuwen.core.foundation.llm.schema.message import AssistantMessage

        m = build_mock_model()
        msg = asyncio.run(m.invoke("hello"))
        assert isinstance(msg, AssistantMessage)
        assert "mock" in msg.content


class TestMockModelClientValidateNoop:
    def test_validate_config_accepts_empty(self):
        # Mirrors how the mock is built: a client configured with placeholder
        # api_key/api_base must validate fine (real clients would raise here).
        from jiuwensymbiosis.agent.abstractions import ModelClientConfig, ModelRequestConfig

        client = MockModelClient(
            model_config=ModelRequestConfig(model_name="mock"),
            model_client_config=ModelClientConfig(
                client_provider="mock", api_key="mock", api_base="mock"
            ),
        )
        assert client._validate_config() is None
