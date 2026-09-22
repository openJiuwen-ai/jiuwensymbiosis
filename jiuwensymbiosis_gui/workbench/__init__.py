# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""现有工作台 GUI 的包级信息与轻量导出。

界面实现位于仓库根目录的 ``jiuwensymbiosis_gui.workbench``。统一启动与插件
选择由 ``jiuwensymbiosis_gui`` 提供；本包初始化不导入 NiceGUI 或执行层。
"""

from __future__ import annotations

__all__ = ["APP_NAME", "ABOUT_TEXT"]

# 面向用户的产品名与简介。刻意只讲 Jiuwen Symbiosis 本身的能力,不提"图形界面/
# 桌面软件/命令行"这类实现细节——用户关心的是产品能做什么。
APP_NAME = "Jiuwen Symbiosis"

ABOUT_TEXT = (
    "Jiuwen Symbiosis 是基于 openjiuwen 的具身智能体框架，一个专为具身智能打造的Symbiosis(共生)架构，"
    "面向具身智能场景提供构型无关的工具、安全策略与多智能体协同能力。\n\n"
    "通过共享动作契约（ActionSpec）与能力门控，一套代码适配 SCARA / 6-DoF / 吸盘 / 夹爪等不同构型的本体；"
    "内置安全防线与视觉反馈闭环，让大模型安全地操控物理世界。"
)
