# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.agent.abstractions — re-export smoke check."""

from __future__ import annotations


class TestAbstractionsImportable:
    def test_all_symbols_importable(self):
        from jiuwensymbiosis.agent.abstractions import (
            AgentRail,
            Tool,
            ToolCard,
            LocalFunction,
            ToolOutput,
            Model,
            ModelClientConfig,
            ModelRequestConfig,
            create_deep_agent,
            SubAgentConfig,
            AgentCard,
            SkillUseRail,
        )

        assert AgentRail is not None
        assert Tool is not None
        assert ToolCard is not None
        assert LocalFunction is not None
        assert ToolOutput is not None
        assert Model is not None
        assert create_deep_agent is not None

    def test_symbol_types(self):
        from jiuwensymbiosis.agent.abstractions import (
            AgentRail,
            Tool,
            ToolCard,
            LocalFunction,
            ToolOutput,
            Model,
            create_deep_agent,
            SubAgentConfig,
            AgentCard,
            SkillUseRail,
        )
        import inspect

        assert inspect.isclass(AgentRail)
        assert inspect.isclass(Tool)
        assert inspect.isclass(ToolCard)
        assert inspect.isclass(LocalFunction)
        assert inspect.isclass(ToolOutput)
        assert inspect.isclass(Model)
        assert callable(create_deep_agent)
