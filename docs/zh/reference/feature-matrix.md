# JiuwenSymbiosis 特性矩阵

> 类别：Reference。本页记录当前代码已经提供的框架能力、内置适配器支持和启用条件，不表示未来路线图。

矩阵以 `KNOWN_CAPABILITIES`、各 Env 的 `capabilities`、Api Mixin 组合、`RobotAgentConfig` 和 `pyproject.toml` 为准。架构与调用关系见[架构指南](../explanation/architecture.md)，适配器接口定义见[机器人适配器参考](adapter-reference.md)。

## 1. 状态说明

| 标记 | 含义 |
|---|---|
| ✅ | 当前仓库有直接实现，可按对应配置使用 |
| ◐ | 条件支持，需要可选依赖、硬件、标定或显式开关 |
| ◇ | 框架已有词汇或扩展契约，但当前内置适配器未实现 |
| — | 当前对象不声明或不提供该能力 |

“支持”表示代码路径和接口存在，不等同于所有机器人、固件和工作区均已完成安全认证。真机仍需按[硬件移植指南](../how-to/port-hardware-adapter.md)验收。

## 2. 内置硬件适配器矩阵

| 特性 | MockArm | Piper | SO-101 |
|---|---|---|---|
| 定位 | 内存 4-DoF 仿真臂 | AgileX Piper 6-DoF CAN 机械臂 | LeRobot SO-101 5-DoF 欠驱动机械臂 |
| 会话入口 | `MockArmEnv` + Mock Api/Model | `build_piper_session` | `build_so101_session` |
| 笛卡尔运动 | ✅ 内存位姿 | ✅ XYZ/R + 完整 `goto_pose` | ✅ XYZ + 最佳努力姿态 IK |
| 关节运动 | — | ✅ 六关节 | ✅ 五个机械臂关节 |
| 实时伺服 | ✅ 仿真 sink | ✅ `servo_to_tip`/`servo_to_flange` | ✅ `servo_to_tip`/`servo_to_flange` |
| 平行夹爪 | ✅ 仿真状态 | ✅ 两态开合，宽度/力由配置控制 | ✅ 百分比两态，带保守接触判断 |
| 吸盘 | — | — | — |
| RGB | ✅ 合成图像 | ◐ 腕部 RealSense | ◐ 桌面 RealSense D405 |
| 深度 | ◐ 测试场景可生成，但不声明 `vision.depth` | ◐ 相机启用时 | ◐ 相机启用时 |
| 开放词汇检测 | ✅ 测试/仿真路径 | ◐ 相机 + 检测服务 | ◐ 相机 + 检测服务 |
| 相机安装模型 | 合成场景 | eye-in-hand | eye-to-hand |
| 手眼变换 | 测试几何 | `T_base_flange(live) @ T_flange_cam` | 固定 `T_base_cam` |
| 检测 sidecar | — | ◐ `detector.spawn=true` | ◐ `detector.spawn=true` |
| Fast path 跟踪 | ◐ 测试路径 | ◐ `track_detect` | ◐ `track_grasp` |
| 主要可选依赖 | `dev`（测试） | `piper`；视觉再加 `full` | Python 3.12 + `so101`；视觉再加 `full` |
| 配置目录 | 测试或示例内构造 | `configs/piper/` | `configs/so101/` |

视觉条件有一处实现差异：

- Piper Env 的类级能力包含视觉，`camera_serial` 未配置时不会创建相机；运行视觉任务仍必须提供相机、标定和检测服务。
- SO-101 根据 `camera_serial` 生成实例能力；未配置相机时不会声明视觉能力，已连接 Driver 若报告相机不可用还会进一步收窄能力。配置了相机但启动失败时，连接按 fail-closed 处理。

## 3. 框架 Capability 矩阵

| Capability | 框架接口 | LLM 工具 | MockArm | Piper | SO-101 |
|---|---|---|---|---|---|
| `motion.cartesian` | `MotionMixin` + `RobotDriver` | `home`、`get_pose`、`goto_xyzr` 等 | ✅ | ✅ | ✅ |
| `motion.joint` | `JointMotionMixin` + `JointDriver` | `move_joint` | — | ✅ | ✅ |
| `motion.servo` | `ServoDriver` + fast controller hook | 无独立公共 Mixin 工具 | ✅ | ✅ | ✅ |
| `grasp.parallel` | `ParallelGripperMixin` + `GripperDriver` | `open_gripper`、`close_gripper` | ✅ | ✅ | ✅ |
| `grasp.suction` | `SuctionMixin` + `SuctionDriver` | `activate_suction`、`deactivate_suction` | — | — | — |
| `vision.camera` | Env/Driver 标记 | 通过 `VisionMixin.get_image` 使用 | ✅ | ◐ | ◐ |
| `vision.depth` | Env/Driver 标记 | 不单独生成工具 | — | ◐ | ◐ |
| `vision.detection` | `VisionMixin` + RAW 投影接缝 | 抓取信息、像素投影、图像 | ✅ | ◐ | ◐ |
| `vision.eye_to_hand` | 相机安装标记 | 不单独生成工具 | — | — | ◐ |
| `sorting.command` | 词表和适配器扩展点 | 无内置通用工具 | — | — | — |
| `speech.tts` | 词表标记；语音前端独立提供 TTS | 无内置机器人工具 | — | — | — |

`grasp.suction` 已有完整框架契约，但当前仓库没有内置吸盘真机适配器；Tutorial 中的 SCARA + 吸盘是教学实现，不应视为经过真机验收的内置硬件支持。

## 4. 执行与工具策略矩阵

| 特性 | 状态 | 默认 | 启用或约束 |
|---|---|---|---|
| DeepAgent 逐步执行 | ✅ | `exec_mode="agent"` | 每步由模型规划并调用工具 |
| Fast path | ◐ | 关闭 | `exec_mode="fast"`；一次模型规划后顺序执行，跟踪动作还需 servo/vision/grasp 能力与适配器 hook |
| 独立工具模式 | ✅ | 可选 | `mode="tool"`，每个 `@robot_tool` 成为一个工具 |
| 代码模式 | ✅ | 可选 | `mode="code"`，提供 `InProcessCodeTool` |
| Hybrid 模式 | ✅ | `mode="hybrid"` | 同时提供独立工具和代码工具 |
| Skill 工作流 | ◐ | `enable_skill=False` | 启用 `SkillUseRail` 和 `RobotControlTool`；内置 `visual_pick`/`visual_place` |
| 自定义工具/Rail | ✅ | 无 | 通过 `extra_tools`、`extra_rails` 注入 |
| 并行工具调用 | ◐ | `parallel_tool_calls=False` | 仅适合审计后的非运动工具；运动/抓取会拒绝，且不能与 Trace 同开 |
| 无硬件/无模型干跑 | ✅ | `--mock` 时 | `MockArmEnv` + `MockModel`，不访问 CAN、相机或模型端点 |

## 5. Rails、Trace 与反馈矩阵

| 能力 | 状态 | 默认 | 自动启用条件或依赖 |
|---|---|---|---|
| SafetyRail | ✅ | 开 | Env 具有笛卡尔或关节运动；检查 Z、XY 和关节软限位 |
| RecoveryRail | ✅ | 开 | Env 具有运动、吸盘或夹爪；失败后尝试 Home 与释放 |
| VisualFeedbackRail | ◐ | 开 | 还需 `vision.camera`；动作后帧进入下一轮上下文 |
| SkillUseRail | ◐ | 关 | `enable_skill=True` |
| TraceRail | ◐ | 关 | `enable_tracing=True`；记录工具、观测、Rail 事件、日志和可选帧 |
| DiagnosisRail | ◐ | 关 | `enable_diagnosis=True` 且必须同时开启 Trace |
| WARNING+ 日志入 Trace | ◐ | 随 Trace | `trace_capture_loggers` 默认 `jiuwensymbiosis` |
| Trace HTML/文本回放 | ✅ | 按需 | `jiuwensymbiosis-replay <trace.json>`，默认生成自包含 HTML |
| 离线 Trace Feedback | ✅ | 按需 | `scripts/analyze_traces.py` 聚类失败并生成需人工审核的建议 |
| 中央日志 | ✅ | INFO + `./logs` | 控制台和轮转文件；`log_dir=None` 时仅控制台 |

## 6. 视觉与感知矩阵

| 特性 | 状态 | 实现或条件 |
|---|---|---|
| RGB + 对齐深度 | ◐ | RealSense/适配器 Driver；深度边界统一为米 |
| GroundingDINO 文本检测 | ◐ | 安装 `full` 并启动或连接检测服务 |
| SAM2 mask | ◐ | 检测配置 `use_sam2=true` |
| 检测 sidecar 生命周期 | ✅ | `make_detector_sidecar()` 随 Session 启停 |
| mask 质心与中值深度 | ✅ | `detect_and_centroid()` |
| eye-in-hand 投影 | ✅ | Piper 提供实现；需要 `T_flange_cam` 与实时法兰位姿 |
| eye-to-hand 投影 | ✅ | SO-101 提供实现；需要固定 `T_base_cam` |
| XY 多点/平移校正 | ✅ | `apply_xy_correction()`，多点变换优先 |
| 抓取与放置高度 | ✅ | `grasp_z_offset_mm`、`place_z_offset_mm` 由共享 `VisionMixin` 统一应用 |
| 手眼标定脚本 | ◐ | 安装 `calib`；Piper 操作流程见[手眼标定指南](../how-to/calibrate-hand-eye.md) |

## 7. 用户入口与可选依赖

| 入口或功能 | 状态 | 安装/命令 |
|---|---|---|
| Python API | ✅ | 核心安装后 import `jiuwensymbiosis` |
| Piper Demo | ◐ | `pip install -e ".[piper]"`；`piper-pick-demo` |
| SO-101 Adapter | ◐ | Python 3.12；`pip install -e ".[so101]"` |
| 视觉/GPU | ◐ | `pip install -e ".[full]"` 并使用 CUDA 12.8 PyTorch 源 |
| 浏览器 GUI | ◐ | `pip install -e ".[gui]"`；`jiuwensymbiosis-gui`，默认 `127.0.0.1:8770` |
| 语音前端 | ◐ | `pip install -e ".[voice]"`；FunASR/录音可选，默认 `NullTTS` |
| 手眼标定 | ◐ | `pip install -e ".[calib,piper]"` |
| Trace 回放 | ✅ | `jiuwensymbiosis-replay` |
| 单元测试 | ✅ | `pip install -e ".[dev]"`；`pytest tests/unit_tests/` |

矩阵维护时应同时核对以下权威源：

- [`env/base.py`](../../../jiuwensymbiosis/env/base.py)：Capability 词表；
- [`api/mixins.py`](../../../jiuwensymbiosis/api/mixins.py)：公共工具能力；
- [Piper Env](../../../jiuwensymbiosis/adapters/piper/env.py) 与 [SO-101 Env](../../../jiuwensymbiosis/adapters/so101/env.py)：适配器声明；
- [`agent/config.py`](../../../jiuwensymbiosis/agent/config.py) 与 [`agent/builder.py`](../../../jiuwensymbiosis/agent/builder.py)：执行和 Rail 开关；
- [`pyproject.toml`](../../../pyproject.toml)：Python、可选依赖和 CLI 入口。
