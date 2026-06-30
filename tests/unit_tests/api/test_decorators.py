# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.api.decorators."""

from __future__ import annotations

from typing import Literal, Optional

from jiuwensymbiosis.api.decorators import (
    ToolMeta,
    _annotation_to_schema,
    _schema_from_signature,
    robot_tool,
)


class TestAnnotationToSchema:
    def test_int(self):
        assert _annotation_to_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _annotation_to_schema(float) == {"type": "number"}

    def test_str(self):
        assert _annotation_to_schema(str) == {"type": "string"}

    def test_bool(self):
        assert _annotation_to_schema(bool) == {"type": "boolean"}

    def test_list_int(self):
        result = _annotation_to_schema(list[int])
        assert result == {"type": "array", "items": {"type": "integer"}}

    def test_dict(self):
        assert _annotation_to_schema(dict) == {"type": "object"}

    def test_optional_float(self):
        result = _annotation_to_schema(Optional[float])
        assert result == {"type": "number"}

    def test_literal(self):
        result = _annotation_to_schema(Literal["a", "b"])
        assert "enum" in result or result == {}
        if "enum" in result:
            assert set(result["enum"]) == {"a", "b"}

    def test_empty_annotation(self):
        from inspect import Parameter

        assert _annotation_to_schema(Parameter.empty) == {}

    def test_any(self):
        from typing import Any

        assert _annotation_to_schema(Any) == {}


class TestSchemaFromSignature:
    def test_simple_function(self):
        def f(x: int, y: float, z: str):
            pass

        schema = _schema_from_signature(f)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert schema["properties"]["x"] == {"type": "integer"}
        assert schema["properties"]["y"] == {"type": "number"}
        assert "x" in schema["required"]
        assert "x" in schema["required"]

    def test_default_values(self):
        def f(x: int, y: float = 1.0, z: str | None = None):
            pass

        schema = _schema_from_signature(f)
        assert schema["properties"]["y"]["default"] == 1.0
        assert "x" in schema["required"]
        assert "y" not in schema["required"]

    def test_self_excluded(self):
        def f(self, x: int):
            pass

        schema = _schema_from_signature(f)
        assert "self" not in schema["properties"]

    def test_kwargs_excluded(self):
        def f(x: int, **kwargs):
            pass

        schema = _schema_from_signature(f)
        assert "kwargs" not in schema["properties"]


class TestRobotToolDecorator:
    def test_attaches_meta(self):
        @robot_tool(desc="test tool", tags=["motion"])
        def my_tool(self, x: float) -> None:
            pass

        assert hasattr(my_tool, "__robot_tool__")
        meta = my_tool.__robot_tool__
        assert isinstance(meta, ToolMeta)
        assert meta.name == "my_tool"
        assert meta.description == "test tool"
        assert meta.tags == ["motion"]

    def test_custom_name(self):
        @robot_tool(name="custom_name", desc="custom desc")
        def my_tool(self) -> None:
            pass

        assert my_tool.__robot_tool__.name == "custom_name"

    def test_capability_gating(self):
        @robot_tool(capability="vision.detection")
        def detect(self) -> dict:
            pass

        assert detect.__robot_tool__.capability == "vision.detection"

    def test_docstring_as_description(self):
        @robot_tool()
        def my_tool(self) -> None:
            """First line of docstring."""

        assert my_tool.__robot_tool__.description == "First line of docstring."

    def test_input_params_auto_generated(self):
        @robot_tool()
        def my_tool(self, x: float, y: float = 0.0) -> None:
            pass

        assert "x" in my_tool.__robot_tool__.input_params["properties"]
        assert my_tool.__robot_tool__.input_params["type"] == "object"

    def test_explicit_input_params(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

        @robot_tool(input_params=schema)
        def my_tool(self) -> None:
            pass

        assert my_tool.__robot_tool__.input_params == schema
