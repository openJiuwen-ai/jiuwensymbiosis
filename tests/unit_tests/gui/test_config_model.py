# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""config_model:dotted 路径读写、YAML 往返、字段组、校验。"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.gui.config_model import (
    FIELD_GROUPS,
    GROUP_ORDER,
    ROBOT_PARAM_FIELDS,
    ConfigModel,
    field_groups_for_body,
)


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


def test_shared_field_groups_have_no_body_specific_robot_params():
    # 「机器人参数」组按本体切换,不进共享 FIELD_GROUPS(否则会给 so101 显示 piper 字段)。
    shared_paths = {spec.path for spec in FIELD_GROUPS}
    assert "env.cfg.low_level.move_speed" not in shared_paths


def test_field_groups_for_body_so101_exposes_so101_params():
    paths = {spec.path for spec in field_groups_for_body("so101")}
    assert "env.cfg.low_level.port" in paths
    assert "env.cfg.low_level.safety_validated" in paths
    # motion_profile 已被 upstream So101Config 移除(from_dict 会拒绝),表单不得再暴露它;
    # 速度/到位等进阶旋钮走「原始 YAML」兜底。
    assert "env.cfg.low_level.motion_profile" not in paths
    # 共享组仍在。
    assert "env.cfg.prompt" in paths and "model.model_name" in paths


def test_field_groups_for_body_piper_keeps_move_speed():
    paths = {spec.path for spec in field_groups_for_body("piper")}
    assert "env.cfg.low_level.move_speed" in paths


def test_field_groups_for_unknown_body_is_shared_only():
    assert field_groups_for_body("nope") == FIELD_GROUPS


def test_all_robot_param_specs_declared_in_group_order():
    for specs in ROBOT_PARAM_FIELDS.values():
        for spec in specs:
            assert spec.group in GROUP_ORDER


def test_disable_vision_toggle_present_for_all_bodies():
    # 「禁用视觉服务」是本体无关的共享开关,任何本体的配置页都应有。
    for body in ("piper", "so101"):
        paths = {s.path for s in field_groups_for_body(body)}
        assert "gui.disable_vision" in paths


def test_so101_form_exposes_camera_and_calib_for_full_vision_setup():
    paths = {s.path for s in field_groups_for_body("so101")}
    assert "env.cfg.low_level.camera_serial" in paths
    assert "env.cfg.low_level.calib_path" in paths  # 视觉三要素全在表单,不用碰 YAML


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
