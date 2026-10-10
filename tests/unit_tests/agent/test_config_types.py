# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The settings parser accepts an explicit subset of Python annotations."""

from typing import Any, SupportsIndex

import pytest
import yaml

from jiuwensymbiosis.agent.config import RobotAgentConfig, _typed


class CustomSetting:
    pass


class CustomStr(str):
    """Overrides conversion and slicing to exercise string normalization."""

    def __str__(self) -> str:
        return "CustomStr.MEMBER"

    def __getitem__(self, key: SupportsIndex | slice) -> str:
        return CustomStr(super().__getitem__(key))


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (tuple[str, int], ["first", "second"]),
        (tuple[str, int], ["first", 2]),
        (tuple[str, int], []),
        (dict[str, int], {"count": 1}),
        (dict[str, int], "invalid"),
        (frozenset[str], frozenset({"item"})),
        (CustomSetting, CustomSetting()),
        (list, []),
        (tuple, ()),
    ],
)
def test_unsupported_annotations_fail_even_when_values_look_valid(annotation, value):
    with pytest.raises(TypeError) as error:
        _typed(value, annotation, "agent.extension.setting")
    assert "agent.extension.setting: unsupported configuration annotation" in str(error.value)
    assert repr(annotation) in str(error.value)


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        (bool, False, False),
        (int, 2, 2),
        (str, "item", "item"),
        (float, 2, 2.0),
        (list[str], ["first", "second"], ["first", "second"]),
        (tuple[str, ...], ["reject", "recover"], ("reject", "recover")),
        (tuple[str, ...], [], ()),
        (str | None, None, None),
    ],
)
def test_supported_annotations_keep_their_value_conversion(annotation, value, expected):
    parsed = _typed(value, annotation, "agent.setting")
    assert parsed == expected
    assert type(parsed) is type(expected)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [(True, int), (False, int), (0, bool), (1, bool)],
)
def test_bool_and_int_are_not_interchangeable(value, annotation):
    with pytest.raises(TypeError, match=f"agent.setting must be {annotation.__name__}"):
        _typed(value, annotation, "agent.setting")


def test_string_subclass_is_normalized_to_builtin_type():
    parsed = _typed(CustomStr("raw"), str, "agent.setting")
    assert type(parsed) is str
    assert parsed == "raw"


def test_string_subclass_setting_survives_yaml_roundtrip():
    cfg = RobotAgentConfig(workspace=CustomStr("workspace"))
    text = yaml.safe_dump(cfg.to_dict())
    restored = RobotAgentConfig.from_dict(yaml.safe_load(text))
    assert restored.workspace == "workspace"
    assert restored.to_dict() == cfg.to_dict()


def test_any_is_the_explicit_escape_hatch():
    value = CustomSetting()
    assert _typed(value, Any, "agent.setting") is value


@pytest.mark.parametrize("annotation", [list[str], tuple[str, ...]])
def test_supported_collections_still_validate_each_element(annotation):
    with pytest.raises(TypeError, match="agent.setting must be str"):
        _typed(["valid", 2], annotation, "agent.setting")
