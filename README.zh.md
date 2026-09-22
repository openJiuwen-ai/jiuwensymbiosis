# JiuwenSymbiosis

[English](README.md) | 中文

JiuwenSymbiosis 是基于 openjiuwen 的具身智能体框架，让一套安全、可审计的 Agent 工作流适配不同机器人本体。

## 核心特性

- **构型无关**：共享动作契约 `ActionSpec` + Capability 门控，经 Adapter 适配器桥接，将机器人几何、厂商 SDK 与 Agent 工作流解耦。
- **任务组合**：一条任务不是一个写死的调用序列，而是由动作契约（ActionSpec）和技能（Skill）动态**组合**出来的。规划器负责把"目标"组织成一条动作序列。
- **环境 + 本体感知的动态编排**：规划器以当前本体状态与环境感知结果为输入，动态编排与之匹配的动作契约与技能；执行过程中，一旦现实状态与下一步骤的前提条件相冲突，即自动触发重规划，而非照本宣科地执行既定流程。
- **执行记忆**：规划器实时掌握已知信息（目标位置、本体状态），依托动作契约自动维护 —— 感知即入账、移动即作废，无需手写缓存；已执行动作产生的状态会沉淀下来，被后续步骤继承引用。
- **实时追踪伺服**：边感知边执行——持续跟踪目标，以高频向本体发伺服指令；用于抓取/放置等关键动作，可支持实时跟随物体移动抓取。
- **主动搜索**：目标不在视野时原地扫视、报方位并逐步逼近；够不着时先规划可达位姿。
- **可达性推理**：当任务将目标描述为 "在抽屉里"" 在箱子下 " 等空间关系时，规划器基于空间关系判定目标是否真正可达；若目标被遮挡或包裹，则先行规划使目标变得可达。
- **动作契约**：每个动作声明前置条件、效果与结果形状；契约不编码顺序，只拒绝前置条件不满足的排列。
- **安全闭环**：运动边界检查、异常恢复、视觉反馈、执行诊断共同保护物理执行。
- **通用视觉感知**：开箱即用的可复用感知管线——取帧、深度、开放词汇检测与分割、质心与深度估计、像素到基座反投影、标定、坐标变换与校正等；eye-in-hand 与 eye-to-hand 只差一个投影函数。
- **技能工作流**：内置 `visual_pick`、`visual_place`、`transport` 技能，规范常见操作流程。
- **可审计执行**：结构化轨迹、帧保存、回放与反馈分析，便于复现与排障。

## 架构设计

![JiuwenSymbiosis 架构](docs/images/architecture-layers.zh.svg)

运行时形成“**感知 → 规划 → 执行 → 观测 → 反馈**”闭环：命令依次经过 Agent、Rails、Tools、API、Env 和 Hardware，观测、失败与轨迹证据反向反馈给 Agent。完整依赖关系和任务时序见[架构解释](docs/zh/explanation/architecture.md)。

## 相关文档

- [文档中心](docs/README.md) — 教程、操作指南、API 参考与设计解释
- [示例工程](examples/README.md) — 无硬件 Mock 示例和真机示例
- [特性矩阵](docs/zh/reference/feature-matrix.md) — 内置适配器与能力支持状态
- [贡献指南](CONTRIBUTING.md) — 开发、测试和提交要求

## 环境要求

| 依赖 | 版本或要求 |
| --- | --- |
| 操作系统 | Ubuntu 22.04（当前已验证平台） |
| Python | `>=3.11,<3.14`；SO-101 适配器需要 Python 3.12 |
| 核心依赖 | `openjiuwen>=0.1.13`；其他版本以 `pyproject.toml` 的 `[project.dependencies]` 为准 |
| 视觉/GPU | `[full]` 使用 CUDA 12.8 对应的 PyTorch 2.8.0 构建 |
| 真机 | 准备适配器所需的 CAN/串口、相机、标定、厂商 SDK，并验收安全边界 |

## 安装指南

```bash
git clone https://gitcode.com/openJiuwen/jiuwensymbiosis.git
cd jiuwensymbiosis
conda create -n jiuwensymbiosis python=3.12
conda activate jiuwensymbiosis
python -m pip install -e .
```

按需安装可选能力：

```bash
python -m pip install -e ".[dev]"       # 测试与开发工具
python -m pip install -e ".[piper]"     # Piper SDK
python -m pip install -e ".[so101]"     # SO-101 / LeRobot；Python 3.12
python -m pip install -e ".[cruzr]"     # Cruzr 双臂（pinocchio 臂 IK；rclpy 由 ROS 工作区提供，非 pip 依赖）
python -m pip install -e ".[voice]"     # ASR 与录音
python -m pip install -e ".[gui]"       # 现有工作台 GUI（NiceGUI 浏览器界面）
python -m pip install -e ".[calib]"     # 手眼标定
python -m pip install -e ".[full]" \
  --extra-index-url https://download.pytorch.org/whl/cu128  # 视觉/GPU 栈
```

组合安装和固定版本运行依赖见[安装与快速开始](docs/zh/tutorial/01-quick-start.md)。

统一启动器通过 GUI 插件 key 选择界面；本仓库当前提供 `workbench`：

```bash
jiuwensymbiosis-gui --list-guis
jiuwensymbiosis-gui --gui workbench
# 等价模块入口
python -m jiuwensymbiosis_gui --gui workbench
```

`.[gui]` 安装现有工作台的依赖。单独发布的 GUI 插件按各自说明安装到同一 Python
环境，再通过 `--gui <key>` 选择。

## 内置适配器

| 适配器 | 状态 | 主要能力 | 可选依赖 |
| --- | --- | --- | --- |
| Piper | 内置真机适配器 | 6-DoF 运动、平行夹爪、眼在手上 RealSense 视觉 | `[piper]`；视觉另加 `[full]` |
| SO-101 | 内置真机适配器 | 5-DoF 运动、平行夹爪、眼在手外 RealSense 视觉 | Python 3.12 + `[so101]`；视觉另加 `[full]` |
| Cruzr | 内置真机适配器 | 移动双臂（底盘 + 升降 + 腰部）、夹板夹取、腰/头双路相机 | `[cruzr]`；视觉另加 `[full]`；运行时需 source ROS 工作区 |

`MockArmEnv` 仍作为内置内存模拟 Env 提供，并用于 `--mock`（仅 Piper），但它不是硬件适配器。

SCARA 和吸盘属于框架已支持的扩展契约，但仓库目前没有经过真机验收的对应内置适配器。准确的启用条件见[特性矩阵](docs/zh/reference/feature-matrix.md)。

## Quick Start

两个真机验证过的运行示例。先在配置里填入自己的串口、序列号和端点，并完成安全边界验收。

SO-101 —— 桌面抓放：

```bash
python examples/run_task.py \
  --config configs/so101/so101.yaml \
  --query "把香蕉放到盘子里" \
  --api-key "$OPENJIUWEN_API_KEY"
```

Cruzr —— 优必选轮式人形机器人移动搬运（先 source ROS 2 + Cruzr 工作区）：

```bash
python examples/run_task.py \
  --config configs/cruzr/cruzr.yaml \
  --query "把棕色箱子上的白色箱子搬到有香蕉的白桌子上" \
  --api-key "$OPENJIUWEN_API_KEY"
```

`examples/run_task.py` 是通用任务入口：用 `--config` 选机器人（YAML 顶层 `adapter:` 字段从注册表选中），用 `--query`（或 `--voice`）给任务，任务不在 config 里。`--mock` 仅对 Piper 提供无硬件干跑；其他机器人走各自的真机会话。各自完整的前置条件见[示例说明](examples/README.md)。

自行编写 Python 入口时，必须在导入 `openjiuwen` 或间接导入它的模块前调用 `clear_proxy_env()`；仓库内置 CLI 和示例已处理。

## License

本项目基于 [Apache License 2.0](LICENSE) 开源。

本产品仅作为流程编排工具，不包含 AI 模型能力；用户在连接 AI 模型用于特定业务场景时，需自行承担欧盟 AI 法案等相关合规义务。
