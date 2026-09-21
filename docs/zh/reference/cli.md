# 命令行参考

> 类别：Reference。控制台入口由 `pyproject.toml` 的 `[project.scripts]` 定义。

## jiuwensymbiosis-run（通用任务运行器）

```bash
jiuwensymbiosis-run --config configs/cruzr/cruzr.yaml --query "把箱子搬到桌上"
jiuwensymbiosis-run --config configs/piper/piper.yaml   --query "把瓶子放到左边"
```

`--config` 必填；YAML 顶层 `adapter:` 字段从注册表选中机器人（`--robot` 覆盖）。任务默认由 `--query` 给出（config 不内置任务；YAML 保留了可选的 `env.cfg.prompt` 回退），两者皆空时启动报错。其他常用覆盖项：

| 选项 | 作用 |
|---|---|
| `--robot` | 覆盖 config 的 `adapter:` 字段（piper / so101 / cruzr） |
| `--mock` | 仅 Piper 内存干跑：MockArmEnv + 离线模型；隐含 `--stepagent` |
| `--stepagent` | 强制逐 step LLM（单步调试）；默认 `fastagent`（编译一次、无逐 step LLM） |
| `--voice` / `--voice-text` / `--voice-audio-file` / `--voice-once` / `--no-wake` / `--tts` / `--asr-device` | 语音模式 |
| `--no-skill` | 关闭 SkillUseRail + robot_control 分派器 |
| `--mode` | `tool` / `code` / `hybrid` |
| `--no-visual-feedback` | 关闭 VisualFeedbackRail |
| `--server-url` / `--model` / `--api-key` | 覆盖 LLM 端点/模型/key |
| `--max-iter` / `--workspace` / `--debug` | 迭代上限、工作区、日志级别 |

`--mock` 使用离线模型和 Mock 环境；`--control-hz`/`--servo-step-mm` 调 fastagent 实时伺服。

## piper-pick-demo

```bash
piper-pick-demo --config PATH [--query TEXT | --voice ...] [--mock]
```

向后兼容别名，指向同一个通用 `run_task.py`（`jiuwensymbiosis-run`）。

## jiuwensymbiosis-replay

```bash
jiuwensymbiosis-replay TRACE_JSON [--open] [--text]
```

默认生成自包含 HTML 回放并打印路径；`--open` 生成后自动用默认浏览器打开；`--text` 输出终端时间线。

## jiuwensymbiosis-gui

```bash
jiuwensymbiosis-gui --list-guis
jiuwensymbiosis-gui --gui workbench --config configs/piper/piper.yaml
python -m jiuwensymbiosis_gui --gui workbench
```

`--gui` 选择已安装的 GUI 插件，默认值为 `workbench`；`--list-guis` 列出可用插件。`--config`、
`--workspace`、`--host`、`--port` 和 `--no-browser` 用于配置所选界面；`--gui-config` 用于传入插件专属配置，
是否支持取决于所选插件。启动器只加载所选插件的应用入口。插件及其依赖需安装在同一 Python 环境；
`.[gui]` 提供当前工作台使用的 NiceGUI 依赖。

`python -m jiuwensymbiosis.gui` 暂时作为兼容转发入口保留，转发到统一启动器。

## 手眼标定

安装 `.[calib]` 后提供三个入口：

```bash
jiuwensymbiosis-calibrate-hand-eye --collect-poses OUTPUT --config RUNTIME_YAML [--import-poses INPUT]
jiuwensymbiosis-calibrate-hand-eye --auto WAYPOINT_ARCHIVE --config RUNTIME_YAML --confirm-estop
jiuwensymbiosis-calibrate-hand-eye --replay STATION_ARCHIVE [--config RUNTIME_YAML]
```

`jiuwensymbiosis-calibrate-hand-eye` 是 mount-neutral 统一入口；相机安装方式来自运行时配置或 archive，不能由命令行覆盖。`--collect-poses --import-poses INPUT` 不连硬件，只把自描述 waypoint archive 规范化到 OUTPUT。`jiuwensymbiosis-calibrate-eye-to-hand` 使用相同流程并强制要求 `eye_to_hand`；`jiuwensymbiosis-calibrate-eye-in-hand` 对 archive 模式（`--collect-poses`/`--replay`/带参 `--auto`）强制要求 `eye_in_hand`，其余调用走 legacy Piper 向导并打印兼容提示。

退出码：`0` 成功或 dry-run 通过，`1` 执行错误，`2` preflight 契约失败，`3` 只生成不可加载的 REVIEW/candidate 报告。

图形界面的「工具 → 手眼标定」向导驱动的是同一套工作流，并额外提供标定板 PDF 生成与示教期实时角点反馈；首次标定建议从向导开始。

## 自省工具（jiuwensymbiosis-actions / -skills / -state）

```bash
jiuwensymbiosis-actions --vocabulary [--json]                       # 共享动作词表（无机器人）
jiuwensymbiosis-actions --config configs/cruzr/cruzr.yaml [--json]  # 门控到一个本体的动作词表
jiuwensymbiosis-skills  [--json]                                    # 技能库 + 契约
jiuwensymbiosis-state   --config configs/cruzr/cruzr.yaml [--json]  # 实时世界状态（可连接）
```

这三者是规划器 / 编码智能体读取的机器可读视图（见[架构指南：两级自主规划](../explanation/architecture.md#two-tier-planning)）：一个动作是什么、一条技能有什么前置条件、当前世界都在哪个位置。
