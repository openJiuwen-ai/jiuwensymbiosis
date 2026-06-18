# JiuwenSymbiosis

## 简介

**JiuwenSymbiosis** 是基于 openjiuwen 的具身智能体框架，一个专为具身智能打造的Symbiosis(共生)架构，面向具身智能场景提供构型无关的工具、安全策略与多智能体协同能力。通过能力织入（Capability Mixin）架构，一套代码适配 SCARA / 6-DoF / 吸盘 / 夹爪等不同构型的本体；内置安全防线与视觉反馈闭环，让大模型安全地操控物理世界。

## 为什么选择 JiuwenSymbiosis？

- **本体解耦**：一套框架适配 SCARA / 6-DoF / 吸盘 / 夹爪等各类构型，新本体只需编写 YAML 配置 + 本体适配层即可接入，无需改动 Agent 核心逻辑。
- **安全防线内置**：SafetyRail（Z 下限 / 工作空间边界拦截）、RecoveryRail（运动异常自动回零 + 释放末端）、VisualFeedbackRail（每次动作后注入相机帧验证结果），确保 LLM 安全操作物理世界。
- **视觉闭环**：视觉感知服务以旁路进程运行，配合腕部相机反投影到基座坐标。LLM 只需用自然语言描述目标物体即可获得 3D 抓取位置，无需理解像素与相机内参。
- **可审计技能工作流**：预置 visual_pick / visual_place / slot_pick 等 SKILL.md 技能文档，LLM 按规范步骤执行而非自由编排，可重现、可审计。

## 快速开始

### 安装

- 操作系统：当前版本支持Ubuntu （已在22.04验证）。
- Python 版本：>= 3.11。

新建conda环境：
```bash
conda create -n 你的环境名 python=3.11
conda activate 你的环境名
```

安装本包（推荐可编辑模式）：

```bash
# 开发安装（核心 + 测试依赖）
pip install -e ".[dev]"

# 完整安装（额外包含视觉/GPU 依赖：torch、transformers、FastAPI 等）
# torch需要CUDA 12.8 构建（2.8.0+cu128），该版本仅存在于 PyTorch 官方源，
# 因此必须带上 --extra-index-url，否则会直接报错（而非静默装成 CPU 版）。
pip install -e ".[full]" --extra-index-url https://download.pytorch.org/whl/cu128

# Piper 真机环境（安装 piper_sdk）
pip install -e ".[piper]"
```

或使用锁定版本号的依赖文件以保障可复现性：

```bash
pip install -r requirements.txt
```

### 样例

以下示例通过 Piper 6-DoF 机械臂完成一次视觉引导抓取。需要真机环境（CAN 总线已激活、视觉感知服务已部署）。

```python
import asyncio

from jiuwensymbiosis.utils.proxy import clear_proxy_env

clear_proxy_env()  # 必须在 import openjiuwen 之前调用

from jiuwensymbiosis import build_robot_agent
from jiuwensymbiosis.agent import RobotAgentConfig, ModelSpec
from jiuwensymbiosis.adapters.piper import build_piper_session


def main():
    # 1. 从 YAML 构建本体会话
    session = build_piper_session.from_yaml("configs/piper/pick_box.yaml")

    # 2. 配置大模型
    model_spec = ModelSpec(
        provider="OpenAI",
        api_base="https://api.siliconflow.cn/v1",
        api_key="your-api-key-here",
        model_name="deepseek-ai/DeepSeek-V3.2",
    )

    # 3. 配置 Agent
    config = RobotAgentConfig(
        mode="hybrid",
        model_spec=model_spec,
        enable_skill=True,             # 启用 visual_pick / visual_place 技能文档
        enable_visual_feedback=False,  # 使用纯文本模型时关闭视觉反馈
        max_iterations=30,
    )

    # 4. 运行
    with session:
        agent = build_robot_agent(session, config=config)
        result = asyncio.run(
            agent.invoke({
                "query": "把黑色盒子放到白色盒子上面。",
                "conversation_id": "pick-box-001",
            })
        )
        print(result)


if __name__ == "__main__":
    main()
```

预期输出：Agent 按 visual_pick / visual_place 技能文档顺序执行回零 → 检测目标 → 接近 → 抓取 → 搬运 → 放置 → 释放，返回任务完成状态。

### 运行 Demo

```bash
# 通过命令行入口运行（pip install 后可用）
piper-pick-demo --config configs/piper/pick_box.yaml --mock

# 或直接运行脚本
python examples/piper_pick_demo.py --config configs/piper/pick_box.yaml --mock

# 真机模式（需激活 CAN 总线）
python examples/piper_pick_demo.py \
    --config configs/piper/pick_box.yaml \
    --max-iter 30 \
    --api-key ""
```

## 架构设计

```
env/         硬件抽象层 (BaseRobotEnv, RobotObservation, MockArmEnv)
api/         能力织入 + @robot_tool 装饰器 (MotionMixin, SuctionMixin, VisionMixin...)
tools/       工具构建器 / InProcessCodeTool / RobotControlTool / slot_pick
agent/       RobotSession + build_robot_agent / build_robot_agent_config + 配置
rails/       安全策略 (SafetyRail, RecoveryRail, VisualFeedbackRail)
skills/      预置技能 (visual_pick, visual_place, slot_pick)
adapters/    本体适配层 (piper/ + _common/ 通用构建器)
serving/     视觉感知服务子进程（当前版本：GroundingDINO + SAM2）
```

* **硬件抽象层**：`BaseRobotEnv` 定义最小硬件契约（连接 / 断开 / 观测），`MockArmEnv` 提供无硬件测试环境。
* **硬件能力层**：`MotionMixin`、`SuctionMixin`、`ParallelGripperMixin`、`VisionMixin` 等声明了各类本体的原子能力，具体本体 API 类通过多继承组合所需能力。
* **工具层**：`build_robot_tools` 将 `@robot_tool` 方法自动包装为 openjiuwen 工具；`RobotControlTool` 提供单一入口派发模式；`InProcessCodeTool` 支持进程内 Python 脚本执行。
* **Agent 层**：`RobotSession` 管理硬件生命周期与旁路进程；`build_robot_agent` 一键构建可调用的 DeepAgent。
* **安全策略层**：openjiuwen AgentRail 子类，在工具调用前后插入安全检查、异常恢复与视觉反馈。
* **本体适配层**：`piper/` 示范了如何接入一台具体本体；`_common/builder.py` 提供通用多态会话工厂。

## 硬件适配

JiuwenSymbiosis 通过 **能力织入 (Capability Mixin) + 适配器 (Adapter)** 模式支持任意机械臂、末端执行器及传感器形态。只需实现 6 个文件（配置、驱动、环境、接口、会话、YAML）即可完成新硬件接入，无需修改核心智能体逻辑。

详细步骤请参阅 **[硬件移植指南](docs/hardware-porting-guide.md)**（含分步教程、API 参考与验证工具说明）。

### 快速开始

```bash
# 1. 复制适配器模板
cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/my_robot/

# 2. 按指南实现 6 个文件（config, driver, env, api, session, yaml）

# 3. 验证适配器兼容性
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot
```

适配器模板（`templates/xxx_adapter/`）提供了完整骨架：

- `config.py` — 硬件配置数据类，标注必填/选填字段
- `lowlevel.py` — 驱动骨架，含 Mock 实现供离线验证
- `env.py` — 硬件抽象层，包装驱动（委托模式）
- `api.py` — 能力织入组合，含三种常见场景（SCARA+吸盘、6-DoF+夹爪、视觉使能）
- `session.py` — 一行式会话工厂，通过 `make_builder()` 创建
- `config_template.yaml` — 带中文注释的 YAML 配置模板

## 功能特性

### 态势感知LOOP

“感知-规划-执行-观测-反馈”闭环，持续提升系统稳定性。

**感知理解：**
使能具身Agent主动感知物理世界。
**安全规划：**
基于任务指令与世界状态进行任务规划。动态赋值技能参数，校验物理可行性、安全性和约束，过滤不可执行方案。
**物理执行：**
依照skill.md建议，调用原子动作工具完成执行。
**观测反馈：**
通过传感器采集执行后的真实世界状态，识别物体位姿、环境变化，将偏差、异常和结果信号回传至规划模块，实现动作参数实时调整、规划动态优化和异常自主恢复。


### 能力织入与自动工具生成

每个 Mixin 声明一个 capability 字符串和一组 `@robot_tool` 方法。框架自动从类型标注生成 JSON Schema，并按照 api.capabilities 与 env.capabilities 的交集进行能力门控——本体不支持的能力，对应工具不会被暴露给 LLM。

```python
class MotionMixin:
    capability = "motion.cartesian"

    @robot_tool(desc="移动到绝对位姿 (x,y,z[,r])，mm/deg", tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None: ...
```

### 双重工具策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `build_robot_tools(api)` | 每个 `@robot_tool` 方法独立暴露为 LLM 工具 | 工具少、prompt 容量充裕 |
| `RobotControlTool(api)` | 单一 `robot_control` 入口 + `action` 字段派发 | 配合 SKILL.md 工作流、缩短 tool list |

两者可并存。安全策略层统一解开 `robot_control` 的 `action/params` 包装，对实际动作进行安全检查。

### 三层安全防线

- **SafetyRail（运动前边界校验）**：在 `goto_xyzr` 等运动工具执行前，校验 Z 限位与 XY 工作空间边界，越界直接拒绝并告知 LLM 原因。
- **RecoveryRail（异常自动恢复）**：运动/抓取工具异常时，自动尝试回零 + 释放末端，将本体带回已知安全状态。
- **VisualFeedbackRail（动作后视觉验证）**：每次运动或抓取后捕获相机帧，注入 Agent 上下文，让 VLM 验证动作结果。

### 视觉感知服务

视觉感知服务以子进程方式运行，与 Agent 进程共享 GPU 但状态隔离。`RobotSession` 在 connect 时自动启动检测子进程，disconnect 时关闭；多个 Agent 可共用同一个检测实例。

### 多智能体协同

`build_robot_agent_config` 返回 `SubAgentConfig`，可将多台本体作为子智能体编排到同一个顶层 Agent 中：

```python
left_arm_cfg = build_robot_agent_config(left_session, name="left_arm")
right_arm_cfg = build_robot_agent_config(right_session, name="right_arm")
top_agent = create_deep_agent(..., subagents=[left_arm_cfg, right_arm_cfg])
```

## 参与贡献

我们欢迎所有形式的贡献，包括但不限于：
- 提交问题和功能建议
- 改进文档
- 提交代码
- 分享使用经验

## 开源许可证

本项目依据 Apache-2.0 许可证授权。
