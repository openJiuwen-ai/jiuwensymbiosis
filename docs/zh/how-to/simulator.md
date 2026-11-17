# 仿真器

Genesis 提供机器人场景预览、物理执行和观测，可用于机械臂运动、夹持与目标抓取。cuRobo 负责机械臂路径规划，GraspGenX 提供目标抓取位姿。

## 主要功能

- **运动规划**：通过 `move_joint` 和 `move_tool_poses` 执行关节或工具位姿目标，执行前检查机器人模型、关节限位与碰撞。
- **夹持与放置**：通过 `open_gripper` 和 `close_gripper` 控制夹爪，结合接触、物体运动和支撑状态确认抓取或放置结果。
- **目标抓取**：通过 `pick_target` 完成预抓、接近、闭合和抬升。感知模式使用 RGB-D 与目标分割，真值模式使用仿真物体网格；两种输入均可调用 GraspGenX。

## 使用入口

在独立环境安装所需的 `genesis`、`genesis-planning`，以及可选的 `genesis-graspgenx` 依赖。cuRobo、机器人资产与 GraspGenX 推理服务需另行准备；感知抓取还需要 GroundingDINO/SAM2 服务。

规划示例位于 `configs/simulators/planning/`。通过 Python 的 `PreviewSettings` 或 `GenesisPreview.configure` 设置 `planning_profile`，后台就绪后创建 Agent 会话并调用动作。动作是否可用取决于输入模式、机器人能力及配置检查结果。

## 适用范围

机械臂规划要求底盘先停稳。当前抓取面向平行或联动夹爪与单刚体物体；闭合指令本身不代表抓取成功。停止或持物状态无法确认时，后续动作会被阻止。

现有 MOZ1 配置用于仿真；Panda 配置为模型与规划参考。其他机器人及真实硬件需要独立适配和验证。

配置字段与后端说明见 [Genesis 物理场景后端](../reference/genesis-physical-scene.md)。
