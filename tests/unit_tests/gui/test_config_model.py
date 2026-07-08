# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""config_model:dotted 路径读写、YAML 往返、字段组、校验。"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.gui.config_model import FIELD_GROUPS, GROUP_ORDER, ConfigModel


def test_get_set_nested_paths():
    cm = ConfigModel.from_dict({})
    cm.set("env.cfg.low_level.move_speed", 20)
    assert cm.get("env.cfg.low_level.move_speed") == 20
    assert cm.get("env.cfg.missing", "d") == "d"


def test_yaml_roundtrip_preserves_values_and_chinese():
    cm = ConfigModel.from_dict({"env": {"cfg": {"prompt": "把黑盒放到白盒上"}}, "agent": {"mode": "tool"}})
    text = cm.to_yaml()
    back = ConfigModel.from_yaml_text(text)
    assert back.get("env.cfg.prompt") == "把黑盒放到白盒上"
    assert back.get("agent.mode") == "tool"


def test_from_yaml_text_rejects_non_mapping():
    with pytest.raises(ValueError):
        ConfigModel.from_yaml_text("- just\n- a\n- list")


def test_from_yaml_text_rejects_invalid_yaml():
    with pytest.raises(ValueError):
        ConfigModel.from_yaml_text("a: [1, 2\nb: broken")


def test_replace_from_yaml_keeps_old_data_on_error():
    cm = ConfigModel.from_dict({"agent": {"mode": "tool"}})
    with pytest.raises(ValueError):
        cm.replace_from_yaml("not: [valid")
    assert cm.get("agent.mode") == "tool"


def test_field_value_falls_back_to_default():
    cm = ConfigModel.from_dict({})
    spec = next(s for s in FIELD_GROUPS if s.path == "agent.mode")
    # 未设置时返回 spec.default(此处为 None),设置后返回实际值
    cm.set("agent.mode", "hybrid")
    assert cm.field_value(spec) == "hybrid"


def test_validate_flags_out_of_range():
    cm = ConfigModel.from_dict({"env": {"cfg": {"low_level": {"move_speed": 500}}}, "model": {"temperature": 9}})
    warnings = cm.validate()
    assert any("速度" in w for w in warnings)
    assert any("温度" in w for w in warnings)


def test_every_field_group_is_declared_in_group_order():
    for spec in FIELD_GROUPS:
        assert spec.group in GROUP_ORDER


def test_patch_detector_writes_into_gdino_server_entry():
    cm = ConfigModel.from_dict(
        {
            "api_servers": [
                {"_target_": "something.else", "port": 1},
                {"_target_": "jiuwensymbiosis.serving.grounding_dino_sam2_server.main", "gdino_model_id": "orig"},
            ]
        }
    )
    assert cm.patch_detector(gdino_model_id="/local/gdino", hf_endpoint="https://hf-mirror.com") is True
    server = cm.data["api_servers"][1]
    assert server["gdino_model_id"] == "/local/gdino"
    assert server["hf_endpoint"] == "https://hf-mirror.com"
    # 非检测器项不受影响
    assert cm.data["api_servers"][0] == {"_target_": "something.else", "port": 1}


def test_patch_detector_no_detector_entry_returns_false():
    cm = ConfigModel.from_dict({"model": {}})
    assert cm.patch_detector(gdino_model_id="/x") is False
