# Trace Feedback Loop 使用指南

> 本模块把 `TraceRail` 从「可回放记录」升级为**双层反馈系统**：
>
> - **在线**：`DiagnosisRail` 在失败步后把当前参数 + 相关历史 + 系统状态回灌下一轮 LLM，让模型当次就能自纠正。
> - **离线**：`analyze_traces` 批量聚类多次运行的失败 step，产出失败报告和人审用的 SKILL.md 补丁建议。
>
> 采集层（`TraceRail` / trace JSON 格式 / 回放）见 [trace.md](trace.md)；设计原理见 [trace-feedback-loop-design.md](trace-feedback-loop-design.md)。本文只讲怎么用。

---

## 一、两层反馈总览

| 层 | 模块 | 运行时机 | 产物 | 谁看 |
|----|------|----------|------|------|
| **在线** | `DiagnosisRail` | 单次失败后、下一轮 LLM 前 | 合成 user 诊断消息 | LLM（当次自纠正） |
| **离线** | `jiuwensymbiosis.trace_feedback` + `scripts/analyze_traces.py` | 批量 trace 分析 | `failure_clusters.json` / `failure_report.md` / `skill_patch_proposals.md` | 工程师（人审改 SKILL.md） |

两层共享同一个 trace substrate（`TraceRail` 落盘的 JSON），互不依赖：你可以只开在线，也可以只跑离线分析已落盘的 trace。

---

## 二、在线模式：DiagnosisRail

### 2.1 它做什么

工具调用失败时，`DiagnosisRail` 在**下一轮 LLM 调用前**追加一条合成 user 消息，内容三段：

1. **当前失败步**：工具名、参数、rail/log 事件、错误。
2. **相关历史（因果链）**：最近 N 条同工具名或同类 rail 事件的 step，提示「是否反复失败」。
3. **系统状态**：RecoveryRail 的 home/release 结果 + 当前 pose，提示「机械臂状态已变化」。

LLM 看到这些就能换参数 / 换策略，而不是盲目重试。诊断消息有 token 软上限，超限按「历史 → 系统状态」顺序丢弃，**始终保留当前步**。

### 2.2 开启（配置文件，推荐）

在任务 YAML 的 `agent:` 块加两个字段：

```yaml
agent:
  enable_tracing: true        # 前置：DiagnosisRail 依赖 trace
  enable_diagnosis: true      # 开启在线诊断
  diagnosis_max_chars: 1500   # 可选：诊断消息软上限
  diagnosis_history_steps: 3  # 可选：因果链回看步数
  diagnosis_history_kinds: ["reject", "recover"]  # 可选：视为相关的 rail kind
```

`enable_diagnosis` 依赖 `enable_tracing`；tracing 关闭时 DiagnosisRail 自动禁用并 warning。

### 2.3 开启（代码）

```python
from jiuwensymbiosis.agent import RobotAgentConfig

config = RobotAgentConfig(
    enable_tracing=True,
    enable_diagnosis=True,
    # 其余字段同上
)
```

### 2.4 失败通道（两种都覆盖）

| 类型 | 触发 | 典型场景 | 诊断来源 |
|------|------|----------|----------|
| **Type A（catch-path）** | 工具把异常转成 `ToolOutput(success=False, error=...)` | SKILL 模式 `RobotControlTool` 派发 | `tool_result.error` + 当前 entry |
| **Type B（传播）** | 异常逃逸出工具 / before-hook | 非 SKILL 直接暴露 `@robot_tool`；`SafetyRail` 抛 `ValueError` | `ctx.exception` |

同一 step 不会重复注入（per-step 幂等标记）。

### 2.5 fast path 行为

fast path（`run_fast_task`）没有 per-step LLM，所以诊断消息**不改变当次 fast 执行**。但 fast path 仍会触发 TraceRail 落盘 trace JSON——这些 trace 可以被离线分析纳入同一个 corpus。

### 2.6 诊断消息长什么样

```
### 诊断：上一步失败
[diagnosis] step failed: goto_xyzr
  error: SafetyRail: z=-50 below z_floor=10
  params: {'x': 150, 'y': 0, 'z': -50, 'r': 0}
  rail: SafetyRail/reject {'tool_name': 'goto_xyzr', 'reason': 'z=-50 below z_floor=10'}

### 相关历史（可能反复失败）
  - #2 goto_xyzr({'x': 120, 'y': 0, 'z': -40, 'r': 0}) → FAIL: SafetyRail: z below floor

### 系统状态
  pose: {'x': 120.0, 'y': 0.0, 'z': 80.0}

请据此修正参数或换策略，不要用相同参数重试。
```

---

## 三、离线模式：analyze_traces

### 3.1 它做什么

把一批 trace JSON 加载、抽取每个失败 step、按归一化签名聚类，产出三份报告：

| 文件 | 内容 | 给谁 |
|------|------|------|
| `failure_clusters.json` | 机器可读的聚类结果（signature / count / examples / affected conversations） | 后处理脚本 |
| `failure_report.md` | 人读报告：概览 + 每 cluster 的工具/rail/reason/参数桶/证据 step | 工程师复盘 |
| `skill_patch_proposals.md` | 人审用的 SKILL.md 补丁建议：模板 diff + 锚点 + 风险 + 验证建议 | 工程师改 skill |

**不自动改任何源文件**——所有建议都写 reports，人审后手动改 SKILL.md。

### 3.2 快速上手

```bash
# 分析整个 trace 目录
python scripts/analyze_traces.py \
  --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
  --out reports/trace_feedback/latest \
  --min-cluster-size 3

# 调试单条 trace
python scripts/analyze_traces.py --trace path/to/one_trace.json --out /tmp/out
```

输出在 `--out` 目录（默认 `reports/trace_feedback/latest`）。

### 3.3 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--trace-dir <DIR>` | — | trace JSON 目录（取顶层 `*.json`，不递归） |
| `--trace <FILE>` | — | 单条 trace JSON（调试用，与 `--trace-dir` 二选一） |
| `--out <DIR>` | `reports/trace_feedback/latest` | 输出目录 |
| `--min-cluster-size <N>` | `3` | 聚类最小样本数，低于此不报告 |
| `--context-steps <N>` | `2` | 失败 step 前后保留的上下文步数（evidence 的 before/after_context） |

### 3.4 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 正常完成。包括「有合法 trace 但没失败 step」——输出空报告，不是错误 |
| `1` | 输入错误：路径不存在 / 未给 trace 来源 / 加载后无合法 trace（含目录有 json 但全坏） |

### 3.5 聚类规则（FailureSignature 怎么算）

同一 signature 的失败聚成一类。signature 由这些字段组成：

| 字段 | 来源 | 归一化 |
|------|------|--------|
| `tool_name` | entry | 原值 |
| `rail_name` / `kind` | 首个 `success=False` 的 rail event（仅 SafetyRail/reject 当根因；RecoveryRail/recover 是 remedy，不当根因） | 原值 |
| `reason_norm` | SafetyRail 取 `detail["reason"]`，否则取 `entry.error` | `trim+lower` + 数值替换成 `<num>`（`z=-50 below z_floor=10` → `z=<num> below z_floor=<num>`） |
| `param_bucket` | `input_params` 的运动/视觉字段 | 数量级桶（见下） |

**param_bucket 归一化**（只做 trace 内可定的，不读 env/config）：

- `x`/`y`/`z`/`r`：正负零（`neg`/`pos`/`zero`）+ 数量级（`abs<1` / `1-10` / `10-100` / `>=100`）
- `q`：长度 + 是否非有限
- `object_name`/`target`/`prompt`：trim+lower；超 40 字符取 `sha256` 前 8 位（跨进程稳定，不用 builtin `hash()`）
- 缺失字段不进入 `param_bucket`；字段存在但值为 `None` 记为 `<none>`，非有限数值记为 `<nan>`/`<inf>`

效果：`z=-50`、`z=-99`、`z=-20` 的同 reason SafetyRail 拒绝聚成一类。

### 3.6 补丁建议模板（SkillPatchProposal）

每类 cluster 生成一条 proposal，按 failure pattern 选模板：

| pattern | 建议方向 |
|---------|----------|
| SafetyRail/reject，reason 含 z/floor/below | SKILL.md 补 `z ≥ env.z_min_safe` 预检 |
| 同上，含 x/y/out of bounds | 补 workspace XY 边界约束 |
| 同上，含 joint/limit/q | 补 `move_joint` 的 `q` 长度/范围校验 |
| 视觉工具（`analyze_scene`/`get_grasp_info_simple`/`pixel_to_base_xyz`）失败 | 补视觉确认步骤或 prompt/target 消歧 |
| 兜底 | 复核 reason_norm，加 guard/retry/参数约束 |

**全局 post-process**：任何 cluster 的 examples 里出现 `RecoveryRail/recover` 事件，都追加「建议在『## 失败处理』补『动作失败后重新 `get_observation` 确认末端空载与位姿再继续』」。

`target_skill` 第一版恒为 `<unresolved>`——trace 不记录激活的 skill 名，且 tool_name→skill 映射不可靠（`goto_xyzr` 多 skill 共用），错配比留空更有害。人审时定 skill。

### 3.7 报告长什么样

**`failure_report.md`**：

```markdown
# Trace Failure Report

- traces analyzed: 3
- failed steps clustered: 3
- clusters: 1

## Cluster 1 — goto_xyzr / SafetyRail / reject

- count: **3**
- affected conversations: ['c0', 'c1', 'c2']
- reason (normalised): `z=<num> below z_floor=<num>`
- param bucket: `x=pos/>=100, y=zero/abs<1, z=neg/10-100, r=zero/abs<1`

- **t0.json:step 1** — goto_xyzr
  - error: `SafetyRail: z=-50 below z_floor=10`
  - rail: SafetyRail/reject {'tool_name': 'goto_xyzr', 'reason': '...'}
```

**`skill_patch_proposals.md`**：

```markdown
## Proposal 1 — target: `<unresolved>`

- confidence: **medium**
- summary: 3 次 SafetyRail/reject 失败（z=<num> below z_floor=<num>），建议人审定 skill 后改 SKILL.md。

### Proposed diff (human review required)

```
在相关 SKILL.md 的『## 参数取值约定』或『## 标准 Workflow』章节，补充：
调用 `goto_xyzr` 时 `z` 必须 ≥ `env.z_min_safe`，否则被 SafetyRail 拒绝。
建议加 pre-check 或失败后上抬 z 重试。
```

- evidence signatures: `goto_xyzr/SafetyRail/reject`
- example: `t0.json:step 1` — goto_xyzr: SafetyRail: z=-50 below z_floor=10

### Risks
- 建议基于聚类证据，未在真实硬件验证。
- target_skill 未自动确定，需人审确认目标 SKILL.md。
```

`confidence`：count≥5 high，3-4 medium，2 low。

---

## 四、典型工作流

### 4.1 开发期：在线 + 离线一起用

```yaml
# 任务 YAML
agent:
  enable_tracing: true
  enable_diagnosis: true
```

跑几次任务（`--mock` 或真机），trace 落盘到 `<workspace>/traces/`。然后：

```bash
python scripts/analyze_traces.py \
  --trace-dir ~/.jiuwensymbiosis/piper_workspace/traces \
  --min-cluster-size 2
```

看 `failure_report.md` 找反复失败模式，看 `skill_patch_proposals.md` 拿建议，人审后改 SKILL.md。

### 4.2 线上：只开在线

```yaml
agent:
  enable_tracing: true
  enable_diagnosis: true
```

LLM 失败时自动收到诊断消息，当次自纠正。trace 仍落盘，供事后离线分析。

### 4.3 事后复盘：只跑离线

已有 trace JSON（不论在线模式是否开启，只要 `enable_tracing=true` 就有），直接：

```bash
python scripts/analyze_traces.py --trace-dir <trace 目录>
```

---

## 五、作为库使用（离线）

不想走 CLI，想在自己的脚本里调：

```python
from pathlib import Path
from jiuwensymbiosis.trace_feedback import (
    load_trace_corpus,
    extract_failure_evidence,
    cluster_failures,
    propose_skill_patches,
)
from jiuwensymbiosis.trace_feedback.report import (
    render_failure_report,
    render_clusters_json,
    render_patch_proposals,
)

paths = sorted(Path("~/.jiuwensymbiosis/piper_workspace/traces").expanduser().glob("*.json"))
corpus = load_trace_corpus(paths)                       # 加载（坏 JSON 跳过不抛）
evidence = extract_failure_evidence(corpus, context_steps=2)  # 抽失败 step
clusters = cluster_failures(evidence, min_size=2)       # 聚类
proposals = propose_skill_patches(clusters)             # 生成建议

print(render_failure_report(clusters, corpus=corpus))
print(render_patch_proposals(proposals))
```

---

## 六、不做什么（边界）

- **不**改 TraceRail / TraceEntry schema。
- **不**在 DiagnosisRail 内硬编码修复策略——它只增强 LLM 可见证据。
- **不**自动写回生产 SKILL.md——离线只写 reports。
- **不**读 env/config 做语义分桶（z floor / workspace bounds / joint limits）——离线第一版只做 trace 内可定的数量级桶。
- **不**解析 SKILL.md frontmatter——`target_skill` 恒 `<unresolved>`，skill 匹配留后续。
- **不**引 LLM/VLM——离线第一版纯确定性。
- **不**引新依赖——yaml 解析 / markdown 拼字符串都用 stdlib。

---

## 七、相关文件

| 文件 | 作用 |
|------|------|
| `jiuwensymbiosis/rails/diagnosis.py` | 在线 DiagnosisRail |
| `jiuwensymbiosis/trace_feedback/analysis.py` | 离线 load/extract/signature/cluster |
| `jiuwensymbiosis/trace_feedback/report.py` | 离线 json/markdown 渲染 |
| `jiuwensymbiosis/trace_feedback/patches.py` | 离线 SkillPatchProposal |
| `scripts/analyze_traces.py` | 离线 CLI |
| `docs/trace.md` | 采集层（TraceRail）手册 |
| `docs/trace-feedback-loop-design.md` | 设计原理 |
