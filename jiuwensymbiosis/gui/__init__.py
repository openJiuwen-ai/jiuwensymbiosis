# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwensymbiosis 图形界面 (NiceGUI,浏览器模式)。

把命令行的机器人任务运行器(``run_robot_task`` + ``RobotSession`` +
``RobotAgentConfig``)包装成用户友好的软件:导航 + 五个页面,用户在软件内点选
任务/本体、可视化改配置、一键运行,并实时看到"普通人也看得懂"的执行过程。

设计要点见 ``docs`` 与仓库根的实现计划。GUI 以进程内方式复用库函数,不 shell
调用 CLI;实时反馈通过注入 ``UIBridgeRail`` (``extra_rails``) 与自挂日志 handler,
经线程安全事件队列 + ``ui.timer`` 回传界面。

事件/逻辑模块(``run_engine`` / ``run_status`` / ``humanize`` / ``config_model``)
不依赖 NiceGUI,可独立测试。启动入口(``__main__``)先做预检(``preflight``):缺
NiceGUI 时给出清晰中文弹窗(含 ``pip install -e ".[gui]"``);其余启动异常由
``__main__.main`` 的兜底弹窗接住,不抛裸 traceback。
"""

from __future__ import annotations

__all__ = ["APP_NAME", "ABOUT_TEXT"]

# 面向用户的产品名与简介。刻意只讲 Jiuwen Symbiosis 本身的能力,不提"图形界面/
# 桌面软件/命令行"这类实现细节——用户关心的是产品能做什么。
APP_NAME = "Jiuwen Symbiosis"

ABOUT_TEXT = (
    "Jiuwen Symbiosis 是基于 openjiuwen 的具身智能体框架，一个专为具身智能打造的Symbiosis(共生)架构，"
    "面向具身智能场景提供构型无关的工具、安全策略与多智能体协同能力。"
    "通过能力织入（Capability Mixin）架构，一套代码适配 SCARA / 6-DoF / 吸盘 / 夹爪等不同构型的本体；"
    "内置安全防线与视觉反馈闭环，让大模型安全地操控物理世界。"
)
