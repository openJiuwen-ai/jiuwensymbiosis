# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""mock_sessions:轮次统计、脚本化模型、以及脚本驱动 run_robot_task 的端到端。"""

from __future__ import annotations

from jiuwensymbiosis.gui.mock_sessions import (
    ScriptedMockModelClient,
    build_scripted_mock_model,
    count_tool_messages,
)

_SCRIPT = [
    {"tool": "home", "args": {}, "say": "回位"},
    {"tool": "get_grasp_info_simple", "args": {"object_name": "black box"}, "say": "识别"},
    {"tool": "goto_xyzr", "args": {"x": 230, "y": 0, "z": 80}, "say": "移动"},
    {"tool": "close_gripper", "args": {}, "say": "抓取"},
]


def test_count_tool_messages_counts_role_tool():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "ok"},
        {"role": "tool", "content": "ok2"},
    ]
    assert count_tool_messages(messages) == 2
    assert count_tool_messages("not a list") == 0


async def test_scripted_client_returns_tool_then_final():
    model = build_scripted_mock_model(_SCRIPT, final_text="完了")
    client = model._client  # 测试内白盒访问
    assert isinstance(client, ScriptedMockModelClient)
    tools = [{"type": "function", "function": {"name": s["tool"]}} for s in _SCRIPT]

    # 第 0 轮(无 tool 消息)返回第一个工具调用
    msg0 = await client.invoke([{"role": "user", "content": "go"}], tools=tools)
    assert msg0.tool_calls and msg0.tool_calls[0].name == "home"

    # 走完脚本后返回收尾文本、无工具调用
    done = [{"role": "tool", "content": "x"} for _ in _SCRIPT]
    msg_end = await client.invoke([{"role": "user"}, *done], tools=tools)
    assert not msg_end.tool_calls
    assert msg_end.content == "完了"


async def test_scripted_client_skips_unavailable_tools():
    model = build_scripted_mock_model([{"tool": "nonexistent", "args": {}}], final_text="空")
    client = model._client  # 测试内白盒访问
    tools = [{"type": "function", "function": {"name": "home"}}]
    msg = await client.invoke([{"role": "user"}], tools=tools)
    # 脚本里的工具都不可用 → 直接收尾,不会崩
    assert not msg.tool_calls
    assert msg.content == "空"
