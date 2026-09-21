# JiuwenSymbiosis 特性矩阵

> 类别：Reference。本页记录当前代码已经提供的框架能力、内置适配器支持和启用条件，不表示未来路线图。

矩阵以 `KNOWN_CAPABILITIES`、各 Env 的 `capabilities`、动作词表（`api/actions.py`）、`RobotAgentConfig` 和 `pyproject.toml` 为准。架构与调用关系见[架构指南](../explanation/architecture.md)，适配器接口定义见[机器人适配器参考](adapter-reference.md)。

## 1. 状态说明

| 标记 | 含义 |
|---|---|
| ✅ | 当前仓库有直接实现，可按对应配置使用 |
| ◐ | 条件支持，需要可选依赖、硬件、标定或显式开关 |
| ◇ | 框架已有词汇或扩展契约，但当前内置适配器未实现 |
| — | 当前对象不声明或不提供该能力 |

“支持”表示代码路径和接口存在，不等同于所有机器人、固件和工作区均已完成安全认证。真机仍需按[硬件移植指南](../how-to/port-hardware-adapter.md)验收。

## 2. 内置硬件适配器矩阵

| 特性 | MockArm | Piper | SO-101 | Cruzr |
|---|---|---|---|---|
| 定位 | 内存 4-DoF 仿真臂 | AgileX Piper 6-DoF CAN 机械臂 | LeRobot SO-101 5-DoF 欠驱动机械臂 | Cruzr 移动双臂（底盘 + 升降 + 腰部） |
| 会话入口 | `MockArmEnv` + Mock Api/Model | `build_piper_session` | `build_so101_session` | `build_cruzr_session` |
| 笛卡尔运动 | ✅ 内存位姿 | ✅ XYZ/R + 完整 `goto_pose` | ✅ XYZ + 最佳努力姿态 IK | — |
| 关节运动 | — | ✅ 六关节 | ✅ 五个机械臂关节 | ✅ 双臂（named joint，弧度） |
| 实时伺服 | ✅ 仿真 sink | ✅ `servo_to_tip`/`servo_to_flange`（fast path 伺服 op 驱动） | ✅ `servo_to_tip`/`servo_to_flange`（fast path 伺服 op 驱动） | — |
| 移动底盘 | — | — | — | ✅ `navigate_relative`/`rotate_base`/`drive_arc`；连续 `base_servo` |
| 升降 / 腰部 | — | — | — | ✅ `set_lift_pose`/`lift_to_clearance`/`turn_waist` |
| 目标接近 | — | — | — | ✅ `approach_for_grasp`/`approach_for_place`（`motion.goal`） |
| 双臂协同 | — | — | — | ✅ `dual_arm_grasp`/`dual_arm_place`（`motion.dual_arm`） |
| 平行夹爪 | ✅ 仿真状态 | ✅ 两态开合，宽度/力由配置控制 | ✅ 百分比两态，带保守接触判断 | — |
| 夹板夹取 | — | — | — | ✅ `grasp.paddle`（双夹板各夹一面，接触力确认） |
| 吸盘 | — | — | — | — |
| RGB | ✅ 合成图像 | ◐ 腕部 RealSense | ◐ 桌面 RealSense D405 | ✅ 腰部 + 头部双路 |
| 深度 | ◐ 测试场景可生成，但不声明 `vision.depth` | ◐ 相机启用时 | ◐ 相机启用时 | ✅ 腰部 RGBD |
| 开放词汇检测 | ✅ 测试/仿真路径 | ◐ 相机 + 检测服务 | ◐ 相机 + 检测服务 | ◐ 相机 + 检测服务 |
| 主动搜索 | — | — | — | ✅ `vision.search` + `search_target`（当前朝向检查头/腰相机并报方位） |
| 相机安装模型 | 合成场景 | eye-in-hand | eye-to-hand | eye-to-hand（腰部 + 头部随动） |
| 手眼变换 | 测试几何 | `T_base_flange(live) @ T_flange_cam` | 固定 `T_base_cam` | 实时 `T_base_cam`（TF 查找，静态标定回退；头部按关节实时算） |
| 可达性（URDF） | — | — | — | ✅ `planning.reachability`（双臂 + 自适应升降判据） |
| 检测 sidecar | — | ◐ `detector.spawn=true` | ◐ `detector.spawn=true` | ◐（`api_servers`） |
| 主要可选依赖 | `dev`（测试） | `piper`；视觉再加 `full` | Python 3.12 + `so101`；视觉再加 `full` | `cruzr`；视觉再加 `full`；运行时需 source ROS 工作区 |
| 配置目录 | 测试或示例内构造 | `configs/piper/` | `configs/so101/` | `configs/cruzr/` |

视觉条件有一处实现差异：

- Piper Env 的类级能力包含视觉，`camera_serial` 未配置时不会创建相机；运行视觉任务仍必须提供相机、标定和检测服务。
- SO-101 根据 `camera_serial` 生成实例能力；未配置相机时不会声明视觉能力，已连接 Driver 若报告相机不可用还会进一步收窄能力。配置了相机但启动失败时，连接按 fail-closed 处理。
- Cruzr 声明腰部 + 头部双路相机与深度；视觉服务经 `api_servers` 提供，检测不可达时共享客户端返回空结果并转换为 `{"ok": false, "reason": "no_detection"}`。

## 3. 框架 Capability 矩阵

能力轴**正交、可自由组合**：运动（cartesian/joint/servo/base/base_servo/lift/waist/goal/dual_arm）、传感（camera/depth/detection/search/eye_to_hand）、末端（parallel/suction/paddle）、规划（reachability）各是各的。

| Capability | 框架接口 | 动作（`ActionSpec`） | MockArm | Piper | SO-101 | Cruzr |
|---|---|---|---|---|---|---|
| `motion.cartesian` | `CartesianDriver` | `goto_xyzr`、`goto_pose`、`move_direction`、`get_pose`、`get_home_pose` | ✅ | ✅ | ✅ | — |
| `motion.joint` | `JointDriver`/`NamedJointDriver` | `move_joint`、`move_named_joint`、`get_joint_positions` | — | ✅ | ✅ | ✅ |
| `motion.servo` | `ServoDriver` + fast controller hook | —（无独立公共动作，由 fast path 实时伺服 op 驱动） | ✅ | ✅ | ✅ | — |
| `motion.base` | Env 动词 `navigate_relative`/`navigate_arc` | `navigate_relative`、`rotate_base`、`drive_arc` | — | — | — | ✅ |
| `motion.base_servo` | Env 动词 `start_base_drive` 等 | —（连续底盘驱动原语） | — | — | — | ✅ |
| `motion.lift` | Env 动词 `set_lifter` | `set_lift_pose`、`lift_to_clearance` | — | — | — | ✅ |
| `motion.waist` | Env 动词 `turn_waist` | `turn_waist` | — | — | — | ✅ |
| `motion.goal` | `motion/approach` | `approach_for_grasp`、`approach_for_place` | — | — | — | ✅ |
| `motion.dual_arm` | `motion/dual_arm` | `dual_arm_grasp`、`dual_arm_place` | — | — | — | ✅ |
| `grasp.parallel` | `GripperDriver` | `open_gripper`、`close_gripper` | ✅ | ✅ | ✅ | — |
| `grasp.paddle` | 本体 `dual_arm_grasp` | —（经 `dual_arm_grasp` 实现） | — | — | — | ✅ |
| `grasp.suction` | `SuctionDriver` | `activate_suction`、`deactivate_suction` | — | — | — | — |
| `vision.camera` | `grab_rgb`/`grab_calibrated_frame` | `get_image`、`pixel_to_base_xyz` | ✅ | ◐ | ◐ | ✅ |
| `vision.depth` | `grab_calibrated_frame` | —（不单独生成动作） | — | ◐ | ◐ | ✅ |
| `vision.detection` | 标定 `CameraFrame` + 检测器钩子 + `perception/scene3d` | `get_grasp_info_simple`、`locate_for_grasp`、`locate_for_place`、`analyze_scene` | ✅ | ◐ | ◐ | ◐ |
| `vision.eye_to_hand` | 相机安装标记 | —（不生成独立动作） | — | — | ◐ | — |
| `vision.search` | `motion/approach.search_target` | `search_target` | — | — | — | ✅ |
| `planning.reachability` | `Reachability`（派生） | —（规划期判据） | — | — | — | ✅ |
| `sorting.command` | 词表和适配器扩展点 | —（无内置通用工具） | — | — | — | — |
| `speech.tts` | 词表标记；语音前端独立提供 TTS | —（无内置机器人动作） | — | — | — | — |

`grasp.suction` 已有完整框架契约，但当前仓库没有内置吸盘真机适配器；Tutorial 中的 SCARA + 吸盘是教学实现，不应视为经过真机验收的内置硬件支持。

## 4. 执行与工具策略矩阵

| 特性 | 状态 | 默认 | 启用或约束 |
|---|---|---|---|
| fastagent 编译一次执行 | ✅ | `exec_mode="fastagent"` | 一次模型规划（技能组合 + 动作组合两级）后顺序执行，无逐 step LLM；`--stepagent` 强制逐步 |
| 运行时重规划 | ✅ | 每步前 | 世界反驳下一步前置条件时重新规划，上限 `max_replans` |
| 独立工具模式 | ✅ | 可选 | `mode="tool"`，每个 `@implements` 方法成为一个工具 |
| 代码模式 | ✅ | 可选 | `mode="code"`，提供 `InProcessCodeTool` |
| Hybrid 模式 | ✅ | `mode="hybrid"` | 同时提供独立工具和代码工具 |
| Skill 工作流 | ◐ | `enable_skill=False` | 启用 `SkillUseRail` 和 `RobotControlTool`；内置 `visual_pick`/`visual_place`/`transport` |
| 自定义工具/Rail | ✅ | 无 | 通过 `extra_tools`、`extra_rails` 注入 |
| 并行工具调用 | ◐ | `parallel_tool_calls=False` | 仅适合审计后的非运动工具；运动/抓取会拒绝，且不能与 Trace 同开 |
| 无硬件/无模型干跑 | ✅ | `--mock` 时 | `MockArmEnv` + `MockModelClient`（工厂 `build_mock_model`，仅 Piper），不访问 CAN、相机或模型端点 |

## 5. Rails、Trace 与反馈矩阵

| 能力 | 状态 | 默认 | 自动启用条件或依赖 |
|---|---|---|---|
| SafetyRail | ✅ | 开 | Env 具有任一运动能力（cartesian/joint/base/lift/waist）；按声明能力派生检查 |
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
| mask 质心与中值深度 | ✅ | `scene3d.locate_for_grasp`/`analyze_scene` |
| eye-in-hand 投影 | ✅ | Piper 提供实现；需要 `T_flange_cam` 与实时法兰位姿 |
| eye-to-hand 投影 | ✅ | SO-101 使用固定 `T_base_cam`；Cruzr 优先使用实时 TF，静态标定回退 |
| XY 多点/平移校正 | ✅ | `apply_xy_correction()`，多点变换优先 |
| 抓取与放置高度 | ✅ | `grasp_z_offset_mm`、`place_z_offset_mm` 统一应用 |
| 主动搜索 | ✅ | `motion/approach.search_target`：当前朝向各看一眼、报方位、不移动本体；扫视寻靶由 `approach_*` 内部完成 |
| 可达性先验 | ◐ | 本体带 URDF + 臂链时派生 `planning.reachability`；Cruzr 提供双臂 + 升降判据 |
| 手眼标定脚本 | ◐ | 安装 `calib`；Piper 见[手眼标定指南](../how-to/calibrate-hand-eye.md)，SO-101 见[SO-101 固定相机标定指南](../how-to/calibrate-so101-eye-to-hand.md) |

## 7. 用户入口与可选依赖

| 入口或功能 | 状态 | 安装/命令 |
|---|---|---|
| Python API | ✅ | 核心安装后 import `jiuwensymbiosis` |
| 通用任务运行器 | ✅ | `jiuwensymbiosis-run --config <robot>.yaml --query "<任务>"` |
| Piper Adapter | ◐ | `pip install -e ".[piper]"` |
| SO-101 Adapter | ◐ | Python 3.12；`pip install -e ".[so101]"` |
| Cruzr Adapter | ◐ | `pip install -e ".[cruzr]"`；运行时需 source ROS 工作区 |
| 视觉/GPU | ◐ | `pip install -e ".[full]"` 并使用 CUDA 12.8 PyTorch 源 |
| 浏览器 GUI | ◐ | `pip install -e ".[gui]"`；`jiuwensymbiosis-gui --gui workbench`，默认 `127.0.0.1:8770`；`--list-guis` 列出已安装插件 |
| 语音前端 | ◐ | `pip install -e ".[voice]"`；FunASR/录音可选，默认 `NullTTS` |
| 手眼标定 | ◐ | `pip install -e ".[calib,piper]"` 或 `.[calib,so101]` |
| 动作/技能/状态自省 | ✅ | `jiuwensymbiosis-actions` / `-skills` / `-state` |
| Trace 回放 | ✅ | `jiuwensymbiosis-replay` |
| 核心测试 | ✅ | `pip install -e ".[dev]"`；`make test-core` |
| GUI 测试 | ◐ | `pip install -e ".[dev,gui]"`；`make test-gui` |

`make test` 运行这两套无硬件测试；`make test-all` 还会包含集成测试。
`.[gui]` 提供工作台的界面依赖；其它已安装的 GUI 插件各自提供所需的界面依赖。

矩阵维护时应同时核对以下权威源：

- [`env/base.py`](../../../jiuwensymbiosis/env/base.py)：Capability 词表；
- [`api/actions.py`](../../../jiuwensymbiosis/api/actions.py)：动作词表（`ActionSpec`）；
- [`adapters/_common/capability_spec.py`](../../../jiuwensymbiosis/adapters/_common/capability_spec.py)：能力→动作映射；
- [Piper Env](../../../jiuwensymbiosis/adapters/piper/env.py)、[SO-101 Env](../../../jiuwensymbiosis/adapters/so101/env.py)、[Cruzr Env](../../../jiuwensymbiosis/adapters/cruzr/env.py)：适配器声明；
- [`agent/config.py`](../../../jiuwensymbiosis/agent/config.py) 与 [`agent/builder.py`](../../../jiuwensymbiosis/agent/builder.py)：执行和 Rail 开关；
- [`pyproject.toml`](../../../pyproject.toml)：Python、可选依赖和 CLI 入口。
