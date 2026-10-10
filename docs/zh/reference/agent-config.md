# Agent 全量配置参考

[English](../../en/reference/agent-config.md) | [文档中心](../../README.md)

本文只描述 agent 设置。硬件、检测服务、语音后端和模型端点继续使用各自的配置块。
`RobotAgentConfig` 的规范结构是 `execution`、`modules`、`logging`；官方 CLI、Runtime 与工作台使用同一套解析和默认值。

## 最小配置与默认行为

```yaml
agent:
  execution:
    mode: fastagent
```

fastagent 默认使用技能规划，必要时退回动作组合；技能关闭后直接进行动作组合，动作执行通道仍然存在。
安全检查与恢复默认开启，跟踪默认允许并按可用能力选取动作，tracing 默认关闭。
逐步视觉反馈和失败诊断目前只支持 stepagent，不能在 fastagent 中显式启用。

切换到 stepagent 且未显式设置模块开关时，skills 默认 false、visual_feedback 默认 true。其他默认值不变。
仅修改 Python 对象的执行模式不会重置已经解析的模块值；切换模式时可重新解析配置或显式修改模块。

## 完整配置示例

下面列出所有可序列化字段及 fastagent 默认值。`execution.stepagent` 可以保存逐步模式参数，但不会影响 fast 规划。
`workspace: null` 展示 Python/CLI 的完整字段；通过 Runtime/工作台提交时省略该键，工作区由绑定提供。
这些是软件默认值，不能替代本体的控制约束。

```yaml
agent:
  execution:
    mode: fastagent
    stepagent:
      mode: hybrid
      max_iterations: 15
      system_prompt: null
      parallel_tool_calls: false
    fastagent:
      max_replans: 2
      tracking:
        enabled: true
        detect_hz: 5.0
        first_target_timeout_s: 8.0
        settle_grip_s: 0.5
        max_re_align_iters: 1
        max_grasp_retries: 1
        servo:
          control_hz: 30.0
          max_lin_step_mm: 6.0
          max_ang_step_deg: 5.0
          pos_tol_mm: 4.0
          ang_tol_deg: 3.0
          settle_ticks: 3
          timeout_s: 20.0
          absolute_timeout_s: 60.0
          progress_pos_epsilon_mm: 0.5
          progress_ang_epsilon_deg: 0.5
          lost_target_grace_s: 3.0
        mask_tracking:
          enabled: true
          min_score: 0.35
          dilation_px: 4
          static_containment: 0.85
          min_visible_ratio: 0.25
          occlusion_area_ratio: 0.98
          max_static_centroid_shift_px: 3.0
          max_static_depth_delta_mm: 10.0
          max_depth_span_mm: 40.0
          min_valid_depth_ratio: 0.5
          motion_min_area_ratio: 0.65
          motion_confirm_frames: 2
          max_motion_step_mm: 35.0
  modules:
    skills:
      enabled: true
    safety:
      enabled: true
    recovery:
      enabled: true
    visual_feedback:
      enabled: false
    diagnosis:
      enabled: false
      max_chars: 1500
      history_steps: 3
      history_kinds:
      - reject
      - recover
    tracing:
      enabled: false
      max_entries: 200
      max_frames: 50
      save_frames: false
      console: false
      dir: null
      capture_loggers:
      - jiuwensymbiosis
  logging:
    level: INFO
    dir: ./logs
    motion_dir: null
  workspace: null
  strict_capabilities: false
```

## 字段说明

所有路径均相对 `agent`。布尔值必须使用 YAML `true/false`，不能写成字符串；数值必须有限。
未知字段、错误类型和越界值会报错。关闭模块时允许保留其已校验参数，便于下次启用。

### 执行核心 `execution`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `mode` | `"fastagent"` | 执行机制：fastagent 或 stepagent |
| `stepagent.mode` | `"hybrid"` | 工具方式：tool、code、hybrid；仅 stepagent 决策使用 |
| `stepagent.max_iterations` | `15` | 逐步 agent 最大循环轮数，整数 ≥1；不限制 fast 序列 |
| `stepagent.system_prompt` | `null` | 覆盖逐步 agent 系统提示；null 使用内置提示；不传给 fast 编译器 |
| `stepagent.parallel_tool_calls` | `false` | 逐步工具并行开关；当前与 tracing 不兼容，运动能力还受 builder 校验 |
| `fastagent.max_replans` | `2` | 运行期间重规划次数上限，整数 ≥0；0 禁止重规划 |

### 跟踪执行 `execution.fastagent.tracking`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `detect_hz` | `5.0` | 后台检测频率上限，Hz，>0 |
| `first_target_timeout_s` | `8.0` | 等待首次目标的超时，秒，>0 |
| `settle_grip_s` | `0.5` | fast runner 普通夹爪动作后的等待，秒，≥0；tracking 关闭也适用 |
| `max_re_align_iters` | `1` | 跟踪下降后的重新对齐上限，整数 ≥0 |
| `max_grasp_retries` | `1` | 跟踪抓取确认失败后的重试上限，整数 ≥0 |
| `enabled` | `true` | 授权 track_detect/track_grasp；实际可用性仍由能力和接口判定 |

### 伺服执行参数 `execution.fastagent.tracking.servo`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `control_hz` | `30.0` | 伺服循环频率，Hz，范围 1–200 |
| `max_lin_step_mm` | `6.0` | 每个控制周期的线性步长上限，mm，>0 |
| `max_ang_step_deg` | `5.0` | 每周期角度步长上限，度，>0 |
| `pos_tol_mm` | `4.0` | 位置到位容差，mm，>0 |
| `ang_tol_deg` | `3.0` | 角度到位容差，度，>0 |
| `settle_ticks` | `3` | 连续满足容差的周期数，整数 ≥1 |
| `timeout_s` | `20.0` | 持续无进展超时，秒，>0；有效进展会刷新计时 |
| `absolute_timeout_s` | `60.0` | 单阶段绝对超时，秒，>0；null 关闭此上限 |
| `progress_pos_epsilon_mm` | `0.5` | 刷新进展计时的位置改善阈值，mm，>0 |
| `progress_ang_epsilon_deg` | `0.5` | 刷新进展计时的角度改善阈值，度，>0 |
| `lost_target_grace_s` | `3.0` | 目标丢失宽限时间，秒，>0 |

### 掩码跟踪 `execution.fastagent.tracking.mask_tracking`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `true` | 固定相机掩码跟踪过滤；仅提供对应取样接口时使用 |
| `min_score` | `0.35` | 最低检测置信度，0–1 |
| `dilation_px` | `4` | 可信掩码膨胀半径，像素，整数 ≥0 |
| `static_containment` | `0.85` | 静止目标掩码包含度阈值，0–1 |
| `min_visible_ratio` | `0.25` | 最小可见面积比例，0–1 |
| `occlusion_area_ratio` | `0.98` | 判断遮挡的面积比例阈值，介于 min_visible_ratio 与 1 |
| `max_static_centroid_shift_px` | `3.0` | 静止目标质心漂移上限，像素，>0 |
| `max_static_depth_delta_mm` | `10.0` | 静止目标深度变化上限，mm，>0 |
| `max_depth_span_mm` | `40.0` | 有效深度跨度上限，mm，>0 |
| `min_valid_depth_ratio` | `0.5` | 最小有效深度比例，0–1 |
| `motion_min_area_ratio` | `0.65` | 运动确认的最小面积比例，>0 且介于 min_visible_ratio 与 1 |
| `motion_confirm_frames` | `2` | 确认目标移动所需帧数，整数 ≥1 |
| `max_motion_step_mm` | `35.0` | 接受新目标的最大位移，mm，>0 |

### 技能 `modules.skills`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `true` | 是否使用技能知识；fast 默认 true，step 默认 false；fast 下不影响动作分派工具注册 |

### 安全检查 `modules.safety`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `true` | 是否挂载 Agent 动作预检；不关闭其他层的约束 |

### 失败恢复 `modules.recovery`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `true` | 是否启用自动恢复；同时控制 RecoveryRail 与 fast runner 的失败退避；不控制资源清理 |

### 视觉反馈 `modules.visual_feedback`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `false` | 执行后图像反馈；仅 stepagent；fast 默认 false，step 默认 true；需要图像模型 |

### 失败诊断 `modules.diagnosis`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `false` | 失败诊断反馈；仅 stepagent；依赖 tracing.enabled=true |
| `max_chars` | `1500` | 诊断文本软上限，整数 ≥1 |
| `history_steps` | `3` | 相关历史条数，整数 ≥0 |
| `history_kinds` | `["reject", "recover"]` | 纳入相关历史的事件种类列表 |

### 执行追踪 `modules.tracing`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `enabled` | `false` | 是否记录执行轨迹；fast 记录离散动作，不记录每个伺服 tick |
| `max_entries` | `200` | 轨迹条目上限，整数 ≥1 |
| `max_frames` | `50` | 图像帧上限，整数 ≥0 |
| `save_frames` | `false` | 保存动作观察 JPEG；依赖 tracing 开启 |
| `console` | `false` | 打印实时轨迹摘要；依赖 tracing 开启 |
| `dir` | `null` | 轨迹目录；null 使用 workspace/traces |
| `capture_loggers` | `["jiuwensymbiosis"]` | 捕获 WARNING 及以上记录的 logger 名称列表 |

### 日志 `logging`

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `level` | `"INFO"` | 日志级别，DEBUG/INFO/WARNING/ERROR/CRITICAL/NOTSET；兼容 WARN/FATAL |
| `dir` | `"./logs"` | 框架日志目录；null 只输出控制台 |
| `motion_dir` | `null` | 每轮运动产物的根目录；null 使用 ./jiuwen_motion_log；在连接前应用 |

### 运行元数据

| 字段 | 默认值 | 含义与约束 |
|---|---|---|
| `workspace` | `null` | 工作区；Python/CLI 可设置；Runtime 使用绑定的 workspace，不接受在 agent 块覆盖 |
| `strict_capabilities` | `false` | 能力一致性严格检查，必须在连接前应用；官方入口已接线 |

## 依赖、执行范围与错误

- `diagnosis.enabled=true` 必须同时开启 tracing；不再以日志警告后静默禁用。
- fastagent 不支持启用 `visual_feedback` 或 `diagnosis`。官方入口在连接前校验；`run_robot_task` / `run_fast_task` 再次校验。
- `execution.stepagent.parallel_tool_calls=true` 与 tracing 不兼容，builder 还会检查运动能力冲突。
- `recovery.enabled=false` 关闭 RecoveryRail 和 fast runner 的自动失败退避，不影响取消、连接清理，也不关闭动作内部的控制约束。跟踪动作的内部重试由 `max_grasp_retries` 独立控制。
- `tracking.enabled=false` 禁用跟踪特殊动作，普通动作仍能执行。`settle_grip_s` 仍作用于普通夹爪动作。
- `max_replans` 当前只限制重规划次数；预算用完后 runner 仍会派发待执行步骤，由现有执行检查处理。它不是最大步骤数或总时长上限。
- tracing 记录离散动作及其事件，不保证覆盖每个跟踪控制周期，也不覆盖 fast 规划之前尚未建立轨迹的阶段。
- 空间记忆和上下文增强尚未实现；`modules.spatial_memory`、`modules.context` 当前均会作为未知字段被拒绝。基础执行记忆不受这些未来开关控制。

## 覆盖优先级和路径

有效值 = 执行模式默认值 + YAML 显式值 + 调用者显式覆盖。覆盖以叶字段递归合并，不替换整个模块。
CLI 的 `--mode`、`--max-iter`、`--control-hz`、`--servo-step-mm` 未传入时不会覆盖 YAML。
`--stepagent` / `--mock` 选择 stepagent；`--no-skill` 仅禁用技能知识；`--no-visual-feedback` 禁用逐步图像反馈。
GUI 直接编辑分组字段，不隐式启用 tracing。任务定义中的 `agent_defaults` 逐字段补齐缺失值，不覆盖用户显式值（包括 null）。
任务默认值与运行 YAML 的 agent 设置先合并，再检查模块依赖；例如任务开启 diagnosis 时，可以使用 YAML 中已开启的 tracing。合并校验失败不会修改原配置。

`logging.dir`、`logging.motion_dir` 以及显式的 `modules.tracing.dir` 相对进程当前工作目录解析；不按 YAML 文件位置重定位。
`workspace` 优先级仍为显式值、`JIUWENSYMBIOSIS_WORKSPACE`、用户 settings、默认目录；Runtime 使用其绑定工作区。

### 随仓配置的本体伺服参数

框架缺省仍为 30 Hz / 6 mm。以下 YAML 在 `execution.fastagent.tracking.servo` 中显式覆盖：

| 本体 | `control_hz` | `max_lin_step_mm` | 配置来源 |
|---|---|---|---|
| SO-101 | 20.0 | 3.0 | `configs/so101/so101.yaml` 及 workbench 随包同名配置 |
| Piper | 10.0 | 5.0 | `configs/piper/piper.yaml` 及 workbench 随包同名配置 |

SO-101 的 20 Hz / 3 mm 是与驱动默认 20 Hz、60 mm/s 约束对齐的显式配置选择。
Piper 随包配置采用源码配置已经显式保留的 10 Hz / 5 mm，使源码与安装包使用相同节奏。

这些数值不代表已完成真机验收。Agent 循环频率与硬件发送频率也不是同一概念：SO-101 驱动有独立发送间隔和速度限制。
CLI、Runtime、工作台读取同一 YAML 时使用同一组值；只覆盖 `--control-hz` 时仍保留 YAML 的步长。
自定义 YAML 需要自行保留所需的显式参数；加载器不会按本体名称补值或从硬件 profile 派生。

## 配置格式约束

仅支持本文的分组结构。旧扁平字段、旧 Python 属性和 `exec_config` 已删除；发现旧字段会立即报错，不做自动转换。
CLI、Runtime 和工作台都通过 `RobotAgentConfig.from_dict` 合并与解析配置，不再维护各自的默认值。
配置文件读取或校验失败时，工作台显示源路径和错误并阻止运行，不用默认配置替代所选文件。
工作台切换执行模式后会刷新表单及缺省值，保留显式设置；显式启用的、不适用于 fastagent 的模块需先关闭。

## 后续模块扩展

接入空间记忆或上下文模块时，在 `ModulesConfig` 中增加对应类型，配置组织为 `enabled` 加模块自身参数；可选增强模块默认关闭。
构建入口根据开关决定是否创建和挂载模块，关闭时跳过其读写与资源初始化，依赖关系纳入 `validate()`。
基础执行状态、动作前置条件和位置新鲜度校验继续由执行核心负责，不随可选记忆模块关闭。
新增模块需同时补充开关行为测试及本页说明；当前尚未提供这些模块的实现或占位开关。

## Python 使用

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig.from_dict(
    {"execution": {"mode": "fastagent"}},
    overrides={"modules": {"tracing": {"enabled": True}}},
)
config.validate(for_execution=True)
# session = ...  # 已构建但未连接的 RobotSession
config.prepare_session(session)  # strict_capabilities / motion_dir 在连接前生效
# with session:
#     result = run_robot_task(session, query, config)
```

`execution`、`modules`、`logging` 也接受对应的 dataclass。`to_dict()` 导出全量有效配置，不包含模型对象或扩展实例。
`model`、`model_spec`、`extra_tools`、`extra_rails` 只用于 Python 构建；YAML 的模型参数仍放在顶层 `model` 块。
这四个 Python 专用参数出现在 `from_dict` 输入中均报错。
`build_robot_agent` 构建可 `invoke()` 的逐步 agent，`build_robot_agent_config` 构建多机器人子 Agent 配置。
两者省略 `config` 时均采用 stepagent 默认值（skills 关闭、visual feedback 开启）；显式传入配置时，要求 `execution.mode="stepagent"`。
直接传入 `RobotAgentConfig()` 会报 `ValueError`，因为该构造器默认选择 fastagent。模式不匹配会在模型、工具等构建前报错；builder 不改写模式，也不重算模块默认值。
显式设置的模块开关保持原值；两个 builder 的功能范围差异见 [框架 API](framework-api.md#多机器人子-agent-的配置范围)。

```python
from jiuwensymbiosis.agent import build_robot_agent, build_robot_agent_config

agent = build_robot_agent(session)  # stepagent 默认值
step_config = RobotAgentConfig(
    execution={"mode": "stepagent"},
    modules={"skills": {"enabled": True}, "visual_feedback": {"enabled": False}},
)
agent = build_robot_agent(session, step_config)
subagent = build_robot_agent_config(session, config=step_config)
```

需要按 `execution.mode` 选择执行方式时使用 `run_robot_task`。上述入口均不负责连接 Session。

相关页面：[框架 API](framework-api.md)、[执行追踪](tracing.md)、[CLI](cli.md)。
