# JiuwenSymbiosis 文档中心

[English](en/README.md) | 中文

本目录是项目用户文档的权威来源。仓根 `README.md` 为英文入口，`README.zh.md` 为中文入口；中文文档位于 `docs/zh/`，是内容权威源，英文位于 `docs/en/`，按相同路径跟随。

本仓资料体系遵循 openJiuwen 社区的[各仓资料体系治理规范](https://gitcode.com/openJiuwen/community/blob/main/contribute/%E5%90%84%E4%BB%93%E8%B5%84%E6%96%99%E4%BD%93%E7%B3%BB%E6%B2%BB%E7%90%86%E8%A7%84%E8%8C%83.md)。文档按完整读者任务组织，Diátaxis 用于确定主要意图，不用于机械切割章节。

## 阅读建议

- 第一次使用：从 Tutorial 开始，先完成无硬件 Quick Start。
- 移植硬件、配置日志或排查执行：进入 How-to。
- 查询类、方法、配置或数据格式：进入 Reference。
- 理解框架整体设计：进入 Explanation。
- 贡献流程见仓根 [CONTRIBUTING.md](../CONTRIBUTING.md)，内部设计记录见仓根 [design/](../design/)。

适配器文档按读者任务分工，不需要三篇顺序通读：

| 目标 | 中文 | English |
|---|---|---|
| 无硬件完成第一个可运行适配器 | [构建第一个机器人适配器](zh/tutorial/02-build-first-adapter.md) | [Build Your First Robot Adapter](en/tutorial/02-build-first-adapter.md) |
| 将厂商 SDK、相机和末端执行器接入真机 | [移植机器人硬件适配器](zh/how-to/port-hardware-adapter.md) | [Port a Robot Hardware Adapter](en/how-to/port-hardware-adapter.md) |
| 查询接口、字段、参数和 Piper 实现位置 | [机器人适配器参考](zh/reference/adapter-reference.md) | [Robot Adapter Reference](en/reference/adapter-reference.md) |

## 中文文档

### Tutorial（教程）

- [安装与快速开始](zh/tutorial/01-quick-start.md)
- [构建第一个机器人适配器](zh/tutorial/02-build-first-adapter.md)

### How-to（操作指南）

- [手眼标定指南](zh/how-to/calibrate-hand-eye.md)
- [图形界面使用指南](zh/how-to/configure-gui.md)
- [配置和使用日志](zh/how-to/configure-logging.md)
- [移植机器人硬件适配器](zh/how-to/port-hardware-adapter.md)
- [使用 Trace Feedback Loop](zh/how-to/use-trace-feedback.md)
- [记录和回放执行轨迹](zh/how-to/use-tracing.md)

### Reference（参考）

- [机器人适配器参考](zh/reference/adapter-reference.md)
- [特性矩阵](zh/reference/feature-matrix.md)
- [命令行参考](zh/reference/cli.md)
- [Agent 与框架 API 参考](zh/reference/framework-api.md)
- [机器人环境、能力与工具 API 参考](zh/reference/robot-api.md)
- [执行轨迹参考](zh/reference/tracing.md)

### Explanation（解释）

- [JiuwenSymbiosis 架构指南](zh/explanation/architecture.md)

## English Documentation

### Tutorial

- [Installation and Quick Start](en/tutorial/01-quick-start.md)
- [Build Your First Robot Adapter](en/tutorial/02-build-first-adapter.md)

### How-to

- [Calibrate Hand-Eye Geometry](en/how-to/calibrate-hand-eye.md)
- [Configure the GUI](en/how-to/configure-gui.md)
- [Configure and Use Logging](en/how-to/configure-logging.md)
- [Port a Robot Hardware Adapter](en/how-to/port-hardware-adapter.md)
- [Use the Trace Feedback Loop](en/how-to/use-trace-feedback.md)
- [Record and Replay Execution Traces](en/how-to/use-tracing.md)

### Reference

- [Robot Adapter Reference](en/reference/adapter-reference.md)
- [Feature Matrix](en/reference/feature-matrix.md)
- [Command-Line Reference](en/reference/cli.md)
- [Agent and Framework API Reference](en/reference/framework-api.md)
- [Robot Environment, Capability, and Tool API Reference](en/reference/robot-api.md)
- [Execution Tracing Reference](en/reference/tracing.md)

### Explanation

- [JiuwenSymbiosis Architecture](en/explanation/architecture.md)

## 文档维护约定

- 一篇功能文档应让读者完成一个完整目标；不要按二级标题机械拆分。
- Tutorial 使用两位数字前缀，其他类别使用 kebab-case 语义名且不编号。
- 功能、配置、CLI 或公共 API 变化时，同步更新中文权威源、英文跟随页和导航。
- 开发方案、实现机制和设计取舍放在 `design/`，Explanation 只保留稳定的用户架构认知和导航。
