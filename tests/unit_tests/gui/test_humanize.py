# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""humanize:工具名→中文短语、robot_control 解包、叙述。"""

from __future__ import annotations

from jiuwensymbiosis.gui import humanize


def test_friendly_label_known_tool():
    assert humanize.friendly_label("goto_xyzr", {"x": 1}) == "移动机械臂"
    assert humanize.friendly_label("home") == "回到初始位置"
    assert humanize.friendly_label("close_gripper") == "闭合夹爪抓取"


def test_friendly_label_unknown_falls_back_to_name():
    assert humanize.friendly_label("some_new_tool") == "some_new_tool"


def test_object_hint_appended_for_detection():
    label = humanize.friendly_label("get_grasp_info_simple", {"object_name": "black box"})
    assert label == "识别并定位物体「black box」"


def test_unwrap_robot_control():
    name, params = humanize.unwrap_robot_control("robot_control", {"action": "goto_xyzr", "params": {"x": 5}})
    assert name == "goto_xyzr"
    assert params == {"x": 5}


def test_unwrap_passthrough_for_plain_tool():
    name, params = humanize.unwrap_robot_control("home", {})
    assert name == "home"
    assert params == {}


def test_label_unwraps_robot_control():
    assert humanize.friendly_label("robot_control", {"action": "close_gripper", "params": {}}) == "闭合夹爪抓取"


def test_narration_is_a_sentence():
    assert humanize.narration("home").startswith("正在")
    assert "回到初始位置" in humanize.narration("home")


def test_frame_after_tools_contains_motion_and_grasp():
    assert "goto_xyzr" in humanize.FRAME_AFTER_TOOLS
    assert "close_gripper" in humanize.FRAME_AFTER_TOOLS
    assert "get_pose" not in humanize.FRAME_AFTER_TOOLS


def test_unwrap_parses_json_string_tool_args():
    """openjiuwen 常以 JSON 字符串传 tool_args;应解析出真实参数而非丢成 {}。"""
    name, params = humanize.unwrap_robot_control("goto_xyzr", '{"x": 385.8, "y": -6.0, "z": 148.0}')
    assert name == "goto_xyzr"
    assert params == {"x": 385.8, "y": -6.0, "z": 148.0}


def test_unwrap_parses_json_string_robot_control():
    """robot_control 以 JSON 字符串传入时也应解包出 action + params。"""
    name, params = humanize.unwrap_robot_control(
        "robot_control", '{"action": "goto_xyzr", "params": {"x": 1, "y": 2, "z": 80}}'
    )
    assert name == "goto_xyzr"
    assert params == {"x": 1, "y": 2, "z": 80}


def test_unwrap_bad_json_string_falls_back_to_empty():
    """非法 JSON 字符串不应抛异常,回落到空参数。"""
    name, params = humanize.unwrap_robot_control("goto_xyzr", "not json{")
    assert name == "goto_xyzr"
    assert params == {}


def test_friendly_label_from_json_string_args():
    """标签也应能从 JSON 字符串参数里提取物体名。"""
    label = humanize.friendly_label("get_grasp_info_simple", '{"object_name": "black box"}')
    assert label == "识别并定位物体「black box」"
