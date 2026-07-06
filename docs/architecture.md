# JiuwenSymbiosis 架构指南：一份代码适配所有机器人形态

> JiuwenSymbiosis 是基于 `openjiuwen` 构建的具身智能体（embodied agent）框架，设计目标是让**同一份代码库适配不同机器人形态**——SCARA、6-DoF、吸盘、夹爪等。其核心是**能力混入（Capability Mixin）组合**机制：新增一种硬件只需 1 个 YAML 配置和 6 个适配器文件，框架核心层无需修改。

---

## 一、七层架构总览

数据流自上而下（命令）与自下而上（观测），7 层职责清晰：

```
┌─────────────────────────────────────────────────────────────┐
│  Agent Layer       RobotSession + build_robot_agent()       │  入口
├─────────────────────────────────────────────────────────────┤
│  Safety Rails      SafetyRail / RecoveryRail / VisualFB Rail │  before_tool_call 拦截
│  (平行观测)        TraceRail —— 可选，默认关，零开销        │  记录/落盘/回放（见 §七）
├─────────────────────────────────────────────────────────────┤
│  Tool Layer        build_robot_tools | RobotControlTool    │  LLM 可调用的工具
│                    | InProcessCodeTool                       │
├─────────────────────────────────────────────────────────────┤
│  Skill Layer       SKILL.md (visual_pick / visual_place     │  可复用技能文档
│                    / slot_pick)                              │
├─────────────────────────────────────────────────────────────┤
│  API Layer         MotionMixin / VisionMixin / SuctionMixin │  @robot_tool 方法
├─────────────────────────────────────────────────────────────┤
│  Env Layer         BaseRobotEnv —— 唯一的硬件契约           │  connect/disconnect/observe
├─────────────────────────────────────────────────────────────┤
│  Hardware Layer    XxxDriver —— 适配作者的主要工作          │  serial/CAN/socket
└─────────────────────────────────────────────────────────────┘
```

下面自底向上逐层拆解，重点看**能力如何被声明、被门控、最终成为 LLM 工具**。

---

## 二、Env 层：唯一的硬件契约

`jiuwensymbiosis/env/base.py` 定义了所有机器人 env 必须实现的接口。注意它**不直接驱动硬件**——它持有一个 `low_level` 驱动并委托：

```python
def set_end_effector(self, engaged: bool) -> None:
    driver = self._require_driver()
    if "grasp.parallel" in self.capabilities:
        driver.set_gripper(engaged)      # 夹爪机器人走这条
    elif "grasp.suction" in self.capabilities:
        driver.set_suction(engaged)       # 吸盘机器人走这条
    else:
        raise NotImplementedError(...)
```

这里有一个值得注意的设计：**同一个 `set_end_effector(True/False)` 动词，根据 env 声明的能力自动分派到 `set_gripper` 或 `set_suction`**。上层 API 因此无需关心末端执行器类型——一个 `set_end_effector` 同时覆盖夹爪与吸盘。

每个 env 必须以 `frozenset` 形式声明自己硬件支持的能力：

```python
# env/mock.py —— 一个 4-DoF + 夹爪 + 相机的仿真臂
capabilities = frozenset({
    "motion.cartesian", "grasp.parallel",
    "vision.camera", "vision.detection",
})
```

env 还需暴露 4 个只读属性供安全 rails 读取（**适配作者只需填值，无需写检查逻辑**）：

- `z_min_safe` —— Z 轴安全下限（mm）
- `workspace_bounds` —— XY 工作区边界 `(xmin, ymin, xmax, ymax)`
- `home_pose` —— 归零位姿
- `tool_offset_mm` —— flange 到 tip 的偏移

框架还内置了 `MockArmEnv`（`jiuwensymbiosis/env/mock.py`），**无需任何硬件即可跑通整条链路**——这是降低接入成本的另一根支柱：开发期全程用 mock，部署期才切真机。配套的 `MockModel`（`jiuwensymbiosis/agent/mock_model.py`）是它在 LLM 侧的对应物：`--mock` 时 demo 往 `RobotAgentConfig.model` 注入一个不联网的 `Model`（`invoke` 返回固定文本，`_validate_config` 跳过 `api_key` 校验），让占位符 `api_key`/`api_base` 也能跑完整 agent loop，无需真实 LLM endpoint。两者一起，"无硬件 + 无 LLM"的纯逻辑干跑才真正闭环。

### 已知能力（`KNOWN_CAPABILITIES`）

定义在 `env/base.py`，是全框架共享的能力词汇表：

| 能力字符串 | 含义 |
|---|---|
| `motion.cartesian` | base 坐标系下的 XYZ(R) 末端命令 |
| `motion.joint` | 关节空间命令 |
| `grasp.suction` | 吸盘开/关 |
| `grasp.parallel` | 平行夹爪开/合 |
| `vision.camera` | 原始图像流可用 |
| `vision.depth` | 深度流可用 |
| `vision.detection` | 高层目标检测 |
| `sorting.command` | 不透明分拣协议（无笛卡尔运动） |
| `speech.tts` | 文本转语音可用 |

---

## 三、API 层：能力 Mixin 与 `@robot_tool` 装饰器

这一层是整个框架设计的核心。`jiuwensymbiosis/api/mixins.py` 把机器人能力拆成若干 **Mixin**，每个 Mixin 声明一个 `capability` 字符串并提供默认实现：

```python
class MotionMixin:
    capability = "motion.cartesian"

    @robot_tool(desc="Move the end-effector tip to (x, y, z[, r])", tags=["motion"])
    def goto_xyzr(self, x, y, z, r=None):
        # 默认实现：假设 tip == flange，直接委托给 env
        self.env.move_to_flange(SimpleNamespace(x=x, y=y, z=z, rx=180, ry=0, rz=r))
```

具体机器人 API 通过**多继承**组合出自己需要的能力集（`api/base.py` 示例）：

```python
class RobotApi(MotionMixin, SuctionMixin, VisionMixin, BaseRobotApi):
    # 只需重写有几何差异的方法；其余沿用默认实现
    ...
```

`BaseRobotApi.capabilities` 属性会**自动沿 MRO 收集所有 Mixin 的 `capability`**——适配作者**无需手动维护能力列表**，加哪个 Mixin 就自动具备哪个能力。

### `@robot_tool` 装饰器

`api/decorators.py` 的 `robot_tool` 给方法挂上 `ToolMeta`（名称、描述、从类型注解自动生成的 JSON Schema、能力、标签）。重写方法**自动继承**父类的装饰器元数据——想定制描述再重装饰即可。

### 仅有 3 个方法没有默认实现

框架已提供大部分默认实现，剩下必须由适配作者实现的是依赖具体硬件的视觉方法：

- `get_grasp_info_simple` —— 一次检测 + 像素到 base XYZ 投影
- `pixel_to_base_xyz` —— 像素重投影（依赖手眼标定）
- `analyze_scene` —— 场景分析（依赖检测器客户端）

这 3 个在 Mixin 层 `raise NotImplementedError`，因为它们依赖具体机器人的检测器与手眼标定矩阵，无法给出**mixin 级**通用默认。但 eye-in-hand 相机机器人的检测→质心→投影→矫正→抓放几何流程是通用的，`adapters/_common/vision.py` 提供 `default_get_grasp_info_simple` / `default_pixel_to_base_xyz` 帮助函数把它抽出——适配作者只需提供检测器 `seg_fn` 和一个 `pose_to_tf(flange_pose) -> 4x4` 回调（把厂商 flange 位姿转成 base←flange 变换，这是唯一真正 per-vendor 的几何步骤），标定数据从 `env.low_level`（`tf_flange_cam` / `intrinsics` / `calibration` / `grab_frames`）读取。`analyze_scene` 仍需 per-adapter 实现。

### 已有的能力 Mixin

| Mixin | capability | 提供的工具方法 |
|---|---|---|
| `MotionMixin` | `motion.cartesian` | `home` / `get_pose` / `get_home_pose` / `goto_xyzr` |
| `JointMotionMixin` | `motion.joint` | `move_joint` |
| `SuctionMixin` | `grasp.suction` | `activate_suction` / `deactivate_suction` |
| `ParallelGripperMixin` | `grasp.parallel` | `open_gripper` / `close_gripper` |
| `VisionMixin` | `vision.detection` | `get_grasp_info_simple` / `pixel_to_base_xyz` / `get_image` / `analyze_scene` |

---

## 四、能力门控（Capability Gating）：工具与硬件自动对齐

这是"一份代码适配所有形态"的核心机制，三步走：

1. **Env 声明**硬件能做什么（手动 `frozenset`）
2. **Api 推导**自己的能力（沿 MRO 自动收集 Mixin 的 `capability`）
3. **`build_robot_tools(api, env=env)` 取交集**——只有 `api.capabilities ∩ env.capabilities` 里的方法才变成 LLM 工具（`tools/builder.py` 的 `_effective_capabilities`）

```python
def _effective_capabilities(api, env) -> frozenset[str]:
    api_caps = getattr(api, "capabilities", None) or frozenset()
    if env is None:
        return frozenset(api_caps)
    env_caps = getattr(env, "capabilities", None) or frozenset()
    return frozenset(api_caps) & frozenset(env_caps)   # 交集
```

**效果**：给一个只有吸盘的机器人接上 `ParallelGripperMixin`，夹爪工具**根本不会出现在 LLM 面前**。硬件不支持的能力对 agent 完全不可见，从源头杜绝"LLM 让吸盘机器人去开夹爪"这类问题。

构建器遍历 MRO 时还有两个细节值得指出：

- 子类重写优先于父类（`seen` 集合保证先遇到的重写胜出）
- `_owning_capability` 沿 MRO 找到声明该方法的那个 Mixin 的 `capability`，作为门控依据

---

## 五、Tool 层：三种工具策略可共存

`agent/builder.py` 的 `_build_tools` 根据 `mode` 组装工具列表，**三种策略可以并存**：

| 策略 | 适用场景 | 特点 |
|---|---|---|
| `build_robot_tools(api)` | 工具少 | 每个 `@robot_tool` 方法 → 一个独立 LLM 工具 |
| `RobotControlTool(api)` | SKILL.md 工作流 | 单一 `robot_control` 入口，`action`/`params` 分派 |
| `InProcessCodeTool` | `mode="code"/"hybrid"` | **进程内** Python 执行，能访问到内存中的 live `env` |

`mode` 取值：

- `"tool"` —— 仅 `build_robot_tools` 暴露的工具
- `"code"` —— 仅 `InProcessCodeTool`
- `"hybrid"`（默认）—— 两者并存

`InProcessCodeTool`（`tools/inproc_code.py`）的设计动机值得一提：openjiuwen 内置的 `CodeTool` 在**沙盒子进程**里跑代码，看不到 agent 进程里的 live 对象——而机器人控制恰恰需要拿到"已连接的 `env`、已预热 RealSense、检测客户端"这些热对象。因此框架提供了一个**进程内 executor**，每次 `exec()` 注入 `{env, api, np, ...}` 全局变量，让 LLM 写出的多步控制流能直接操作真实硬件。

### 安全 Rails 的透明解包

当用 `RobotControlTool` 时，所有动作都走 `robot_control` 一个入口，`action`/`params` 藏在参数里。SafetyRail 会**透明解包**后再做安全检查：

```python
if tool_name == "robot_control":
    action = args.get("action", "")
    params = args.get("params", {})
    tool_name = str(action); args = params
```

因此无论用哪种工具策略，安全检查都生效。

---

## 六、安全 Rails：三道防线

`jiuwensymbiosis/rails/` 三道 `before_tool_call` 钩子，在工具执行前拦截/兜底，由 `RobotAgentConfig` 开关启用、session 能力门控：

### 1. SafetyRail —— 动作前的"软件预检"

拦截 `goto_xyzr`/`goto_pose`，校验 Z 下限（`z_floor_mm` 或 env 的 `z_min_safe`）与 XY 边界（`xy_bounds_mm` 或 env 的 `workspace_bounds`）。越界 `raise ValueError`，被 openjiuwen 转成 tool-exception 回灌给 LLM **自行纠错**。它是硬件急停的**补充而非替代**，专门防 `goto_xyzr(0,0,-50)` 这类 LLM 幻觉。

### 2. RecoveryRail —— 失败后自动归零

动作/抓取失败时，自动 `home()` + 释放末端执行器。

### 3. VisualFeedbackRail —— 动作后拍照回灌

每次运动/抓取后抓一帧图像注入上下文，供 VLM **核验**结果。需 `vision.camera` 能力。

**两阶段注入**（保证消息顺序合法）：`after_tool_call` 只抓帧 + 编码 + 暂存到 `ctx.extra["visual_feedback_pending"]`（`_PendingFrame` 结构体，带 `b64`/`tool_name`/`trace_step`/`frame_path`），不碰 `ModelContext`；`before_model_call`（此时 openjiuwen 已写完所有 `ToolMessage`）才 `await ctx.context.add_messages(UserMessage([text, image_url]))` flush 暂存帧。最终序列为 `assistant(tool_calls) → tool(result) → user(image) → 下一轮 model call`——若在 `after_tool_call` 直接注入，会变成 `… → user(image) → tool(result)`，OpenAI 风格 API 会拒绝（tool result 必须紧跟 tool call）。`trace_step` 在 `after_tool_call` 时从 `ctx.extra[_TRACE_RAIL_KEY]` 读 TraceRail 的 `trace.current_step` 暂存，flush 时显式传给 `record_rail_event_at_step(step=...)`，事件钉到正确 entry（多 tool calls 一轮迭代也能分别对位，不会全落到 `entries[-1]`）。`frame_path` 由 `frame_sink` 返回、同样暂存，flush 时进事件 `detail`（`{tool_name, frame_path}` 契约，见 `docs/trace.md`）。`after_invoke` 清理未被消费的暂存帧。注入失败永不逃逸（`_inject` 返回 bool，`except Exception` 吞掉；`CancelledError` 继承 `BaseException` 不被吞，正常传播），避免成功动作被框架误判为 tool failure。fast-path op-ctx 无 `ModelContext`，`_inject` 返回 False，帧仍经 `frame_sink` 落盘供回放。

> 另有 `SkillUseRail`（`agent/builder.py`），非安全 rail——仅 `enable_skill=True` 时附加，加载内置 `SKILL.md` 并附 `RobotControlTool`。详见第五章。
> 还有平行观测 rail `TraceRail`，不拦截动作，只记录与回放，见下一节。

> **并行工具调用默认关 + 运动硬校验**：`RobotAgentConfig.parallel_tool_calls` 默认 `False`，透传给 `create_deep_agent`（单机器人）与 `SubAgentConfig`（多机器人）。机器人运动本就顺序；且 openjiuwen 各 tool-call 的 `ctx.extra` 是共享 dict，并行会让所有按 `ctx.extra`/`trace.entries[-1]` 定位当前步的 rail 竞态。更进一步：`build_robot_agent` / `build_robot_agent_config` 在 `parallel_tool_calls=True` 且 env 含 `motion.*` / `grasp.*` 能力时直接 `raise ValueError`——运动/抓取不允许并行。非运动能力（如 `vision.*` / `speech.tts`）不受限，允许"视觉+语音"并行。**TraceRail 与并行互斥**：`parallel_tool_calls=True` 且 `enable_tracing=True` 也直接 `raise ValueError`——TraceRail 用共享 `ctx.extra` 的 `_TRACE_CURRENT_KEY` 定位当前步、用 `entries[-1]` 兼容旧 sink，两者在并行下都会钉到错步。

---

## 七、执行轨迹与回放（TraceRail）

`TraceRail`（`jiuwensymbiosis/agent/trace.py`）是**平行观测 rail**——不拦截/兜底动作，只采集与持久化。通过 `RobotAgentConfig.enable_tracing` 启用，**默认关**（零开销）。它挂在 openjiuwen 的生命周期钩子上（`before_invoke`/`before_tool_call`/`after_tool_call`/`on_tool_exception`/`after_invoke`），不改任何 `@robot_tool`、env 或其它 rail。

每步工具调用记一条 `TraceEntry`：动作名（解包 `robot_control` 后的实际名）、参数、成功/错误、耗时、pose 快照（**不含**原始 rgb/depth，控 JSON 体积）、可选 JPEG 帧。Rail 事件用两套互补机制采集：`TraceEventSink` 通知钩子让三个安全 rail 在真实触发点推结构化结果（如 RecoveryRail 的 `{home_ok, released_ok}`），`TraceLogHandler` 把 `trace_capture_loggers`（默认 `["jiuwensymbiosis"]`）的 `WARNING`+ 日志记进来——无需改业务代码。

invoke 结束写一次 JSON 到 `<workspace>/traces/{run_token}.json`；帧（可选）存 `traces/frames/{run_token}/step_NNN.jpg`，**每次 invoke 独立子目录**，历史引用永久有效。`jiuwensymbiosis-replay <trace.json>` 默认生成自包含 HTML 回放，`--text` 回退纯文本时间线。

任务 YAML 加一个 `agent:` 块即可开启：

```yaml
agent:
  enable_tracing: true
  trace_save_frames: true   # 可选：保存 JPEG 帧
  trace_console: true       # 可选：运行时逐轮 dashboard
```

字段语义、配置项全表、handler 生命周期、序列化规则、典型 JSON 结构等细节见 [trace.md](trace.md)。

### 样例 trace（`examples/sample_trace/`）

仓库内置一份真机运行产物，供不接硬件时直接翻阅 trace 长什么样：

- [piper-demo-…1743847.json](../examples/sample_trace/piper-demo-77816242_20260626_113438_033124_1743847.json) —— 一次完整 invoke 的 trace JSON
- [piper-demo-…1743847.html](../examples/sample_trace/piper-demo-77816242_20260626_113438_033124_1743847.html) —— `jiuwensymbiosis-replay` 生成的自包含 HTML 回放
- [frames/…/step_NNN.jpg](../examples/sample_trace/frames/piper-demo-77816242_20260626_113438_033124_1743847/) —— 每步动作后帧 + `step_000.jpg` 首帧

**这份 trace 演示了什么**（query「把黑盒子放到白盒子上面」，共 30 步）：

| 步 | 动作 | 看点 |
|---|---|---|
| 5/6 | `get_grasp_info_simple` | 检测成功，`output_summary` 含 `position`/`grasp_position` 等 base XYZ |
| 9 | `robot_control` | 解包后的实际动作 `open_gripper`（非裸 `robot_control`） |
| 13-20 | `goto_xyzr` / `robot_control` | 一连串运动+抓取成功步，每步带 pose 快照 + 帧 |
| 21/24/25/27/29/30 | `home`/`goto_xyzr`/`move_joint` | Piper 硬件超限失败（`OUT OF REACH`），`error` 字段记异常 |
| 同上 | — | 每个失败步都带一条 `RecoveryRail/recover` 的 `rail_event`（`{home_ok:false, released_ok:true}`）+ 一条 recovery 的 `WARNING` `log_event`——正好把 §七讲的 `TraceEntry` 字段、`TraceEventSink` 通知钩子、`TraceLogHandler` 日志捕获三样东西具象化 |

翻阅建议：先开 HTML 看整体流程（帧与参数同卡），再用 JSON 查具体步的 `observation`/`rail_events`/`log_events`。

---

## 八、RobotSession：生命周期聚合器

`jiuwensymbiosis/agent/session.py` 是上下文管理器，`with session:` 即完成连接/断开，**两者都幂等**。它把一个机器人单元的全部状态聚合到一处：

- `env`（硬件驱动实例）
- `api`（能力 Mixin 对象）
- `sidecar_starters`（如检测子进程，自动随 connect 启动、disconnect 停止）
- `globals_provider`（给 `InProcessCodeTool` 注入的全局变量）

`connect()` 里还有一道**能力一致性检查**：如果 env 声明了能力但 api 没有，或反之，会记 warning——这是排错时最早暴露问题的信号。其中「api 声明了但 env 不支持」（明显的配置错误：加了 Mixin 没改 env，或硬件能力变了）在 `RobotSession(strict_capabilities=True)` 或 `RobotAgentConfig(strict_capabilities=True)` 下会**硬失败**，抛出带修复指引的 `ValueError`；`env`-only 的能力（硬件有但 api 没暴露）始终只 warning，因为那是"少了个工具"而非配置错误。

`globals_provider` 返回的 `{env, api, np, **extra_globals}` 会在 `build_robot_agent` 渲染 system prompt 时自动反射成一段「可用全局变量」声明，让 LLM 在代码模式知道 `extra_globals` 里新增的 helper 可用——适配作者加 `extra_globals["my_helper"] = ...` 后无需手改 prompt。

`describe()` 返回的 JSON 摘要里，`effective_capabilities` 就是 `env ∩ api` 的交集，即真正会被门控成工具的能力集。

---

## 九、视觉感知：检测器作为子进程

检测（GroundingDINO + SAM2）跑在**独立子进程**里，通过 HTTP 通信（`adapters/_common/detector_client.py`）：

```
init_detector(service_url) → segment_fn(image, text_prompt) → [{"mask", "box", "score", "label"}]
```

`RobotSession` 用 `sidecar_starters` 管理这个子进程的生命周期，**适配作者无需关心启停**。

`_common` 包提供共享工具，让 piper 之外的适配器也能低成本复用：

- `detector_client.init_detector()` —— 检测器 HTTP 客户端
- `vision.detect_and_centroid()` —— 检测 + 取掩膜质心 + 中值深度窗口
- `vision.apply_xy_correction()` —— 2D 线性 XY 矫正

注意 `detect_and_centroid` 的一个细节：检测器返回的 mask 分辨率可能与 RGB 不同，所以内部做了 mask→image 坐标缩放，保证后续像素→base 反投影用的是一致的图像内参。

### 视觉感知管线的数据流

```
相机帧 (RGB + depth)
   │
   ▼
detect_and_centroid()  ──→  最佳掩膜 + 质心 (u,v) + 中值深度
   │
   ▼
pixel_to_base_xyz(u, v, depth)  ──→  base 坐标系 XYZ (依赖手眼标定)
   │
   ▼
apply_xy_correction()  ──→  矫正后的 XY（多采样仿射拟合或单点平移）
   │
   ▼
get_grasp_info_simple()  ──→  {position, grasp_position, place_position, ...}
```

---

## 十、`make_builder`：消除样板代码

每个适配器最后都要提供一个 `build_xxx_session`，支持三种调用方式（传 config / 传 YAML / 传 dict）。手写纯属样板，因此 `adapters/_common/builder.py` 提供 `make_builder`：

```python
build_xxx_session = make_builder(
    XxxConfig, XxxEnv, XxxApi,
    # 声明式字段映射：cfg 同名属性直接透传，"cfg_attr:api_kwarg" 重映射，
    # 支持 "detector.url:detector_service_url" 这样的点路径取嵌套子配置。
    api_kwargs_from_cfg=["z_correction_mm", "detector.url:detector_service_url"],
    sidecar_builders=[make_detector_sidecar()],  # 内置检测器 sidecar 工厂
    decorate=_set_extra_globals,
)
# 之后可这样用：
build_xxx_session(cfg)
build_xxx_session.from_yaml("path.yaml")
build_xxx_session.from_dict({...})
```

它负责构造 env → 构造 api → 收集 sidecar → 装配 `RobotSession` → 可选 `decorate` 回调。`api_kwargs_from_cfg` 既接受声明式列表（同名字段透传 / `cfg:api` 重映射 / 点路径），也接受传统的 `cfg -> dict` 回调（向后兼容）。`make_detector_sidecar()` 封装了从 `cfg.detector` 读取 GroundingDINO+SAM2 sidecar 参数并按 `spawn` 开关决定是否启动的逻辑，让带视觉的适配器 `session.py` 真正接近一行。**适配器作者一行代码拿到一个支持三种入口的工厂**。

---

## 十一、接入新硬件的成本有多低

了解了上面的架构分层，再回答最关键的问题：**接入一个新机器人要付出多少代价？**

答案是 **6 个文件 + 1 个 YAML**，其中大部分是从模板拷贝后填空：

| 你要写的文件 | 你实际做什么 | 是否可纯靠模板生成 |
|---|---|---|
| `config_template.yaml` | 填写硬件参数（CAN 口、夹爪行程、安全 Z 轴下限等） | ✅ 中文注释逐项引导 |
| `config.py` | `@dataclass` + `from_yaml()`/`from_dict()` | ✅ 模板已给 |
| `lowlevel.py` | 驱动：串口/CAN/Socket 翻译成 `move_to_pose_blocking(pose, ...)` 等动词 | ⚠️ 唯一需要写真实硬件逻辑的地方 |
| `env.py` | `BaseRobotEnv` 子类：声明 `capabilities` + 暴露 4 个属性 | ✅ 模板已给 |
| `api.py` | 多继承 Mixin，**仅重写有几何差异的方法**；eye-in-hand 视觉可委托 `_common/vision.default_*` 帮助函数 | ✅ 多数方法无需手写 |
| `session.py` | `make_builder(...)` 一行（声明式 `api_kwargs_from_cfg` + `make_detector_sidecar()`） | ✅ 一行代码 |

关键点在于：**`api.py` 里绝大多数方法无需自己实现**。框架的 Mixin 默认实现会把 `goto_xyzr` 这类高层动作委托给 `self.env.<动词>()`。只有当机器人本体的几何与标准假设不一致时才需要重写——例如 Piper 是倾斜工具（tip 不等于 flange），需要重写 `goto_xyzr` 做 tip→flange 的坐标换算（见 `jiuwensymbiosis/adapters/piper/api.py` 的 `goto_xyzr`）。

写完后两行命令验证：

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot   # 静态结构
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_robot  # 运行时冒烟（需可连接的 mock env）
```

这是这套架构对开发者的核心承诺：**把适配硬件的工作量压到最小，把唯一的硬件逻辑隔离在单个驱动文件里。**

---

## 十二、接入新硬件的完整流程

把上面的机制串起来，接入一个新机器人 `acme` 的标准流程：

1. **拷贝模板** `templates/xxx_adapter/` → `jiuwensymbiosis/adapters/acme/`
2. **填 YAML** `config_template.yaml`（CAN 口、夹爪行程、Z 安全下限、工作区边界……）
3. **写 `lowlevel.py`** —— 唯一的硬件逻辑：把厂商 SDK 翻译成 `move_to_pose_blocking(pose, ...)` / `set_gripper` / `grab_frames` 等动词
4. **写 `env.py`** —— 声明 `capabilities` frozenset，暴露 4 个属性（从模板填值即可）
5. **写 `api.py`** —— 多继承需要的 Mixin；**只有当本体几何与默认假设（tip==flange）不符时**才重写（如 Piper 的倾斜工具换算）；eye-in-hand 视觉可委托 `_common/vision.default_get_grasp_info_simple` / `default_pixel_to_base_xyz`
6. **写 `session.py`** —— `make_builder(...)` 一行
7. **静态校验** `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.acme`
8. **运行时冒烟** `python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.acme` —— 用可连接的 mock env 驱动每个 `@robot_tool`，断言不崩、返回可序列化
9. **跑 mock** `python examples/xxx_demo.py --config ... --mock` —— 无需真机先验证逻辑

**整个流程里，框架核心层（agent/api/env/tools/rails）无需改动。** 这是 Capability Mixin 架构的杠杆点：把"形态差异"完全收敛进适配器目录，把"共性能力"沉淀为可组合的 Mixin。

更详细的硬件移植步骤见 [hardware-porting-guide.md](hardware-porting-guide.md)。

---

## 十三、关键设计原则小结

| 设计 | 收益 |
|---|---|
| Mixin 默认委托 `self.env.<动词>()` | 适配作者只重写有几何差异的方法，多数方法无需手写 |
| 能力自动从 MRO 推导 | 无需手动维护能力清单 |
| `api ∩ env` 交集门控工具 | 硬件不支持的能力对 LLM 不可见，防幻觉 |
| env 是唯一硬件契约 | 换硬件只换 env + driver，上层零改动 |
| `make_builder` 工厂 | 一行代码拿到支持 cfg/YAML/dict 三入口的 session 构造器 |
| 检测器独立子进程 | 重模型隔离，生命周期自动随 session 管理 |
| Mock 环境全程可用 | 开发期无硬件也能跑通链路 |
| Rails 透明解包 `robot_control` | 安全检查对工具策略无关 |
| SafetyRail 抛 `ValueError` 而非硬终止 | LLM 可自行纠错，不中断整轮 |
| `restrict_to_work_dir=False` 仅在 `include_tools=False` 时 | 安全地加载打包的内置 SKILL.md |
| TraceRail 平行采集，默认关 | 一次 invoke 一个 JSON + 可选帧，可回放可复盘，关闭时零开销 |

---

**总结**：JiuwenSymbiosis 把"机器人形态的多样性"这个本质复杂度，用**能力 Mixin + 能力门控 + 单一硬件契约**三个机制收敛到了适配器目录里。对开发者而言，接入新硬件的成本被压缩到了 **1 个 YAML + 1 个驱动文件 + 4 个填空文件**，而 agent 层、安全层、工具层、感知层的能力是开箱即用的——只要 env 声明了对应能力，工具和安全策略就会自动就位。
