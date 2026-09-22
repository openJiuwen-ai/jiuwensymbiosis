# 接入一个 GUI 插件

[English](../../en/how-to/add-gui.md) | 中文

本文说明如何把一套独立界面作为已安装的 GUI 插件接入 JiuwenSymbiosis。本文只覆盖插件契约、共享运行时和测试边界；具体产品界面由插件自己设计。可运行样例见 [`examples/gui_plugin`](../../../examples/gui_plugin/README.md)。

## 插件边界

GUI 插件是独立 Python distribution，通过 `jiuwensymbiosis.gui` entry point 注册。插件使用核心包提供的轻量契约 `jiuwensymbiosis_gui.plugin`，声明 `GuiSpec`，并提供接收 `LaunchOptions` 的 `main(options)`：

```python
from jiuwensymbiosis_gui.plugin import GUI_API_VERSION, GuiSpec, LaunchOptions


def describe() -> GuiSpec:
    return GuiSpec(
        key="example",
        display_name="Example GUI",
        api_version=GUI_API_VERSION,
        launch_target="my_gui.app:main",
    )


def main(options: LaunchOptions) -> int:
    ...
```

在插件的 `pyproject.toml` 声明入口，入口名称必须与 `GuiSpec.key` 相同：

```toml
[project.entry-points."jiuwensymbiosis.gui"]
example = "my_gui.plugin:describe"
```

`describe()` 会在启动器列出和验证插件时调用，因此只应导入契约和轻量模块。把 NiceGUI、Qt、WebGL、点云或其他专属 UI 依赖延迟到所选插件的启动路径中加载。插件框架依赖需写在各自 distribution 的 `dependencies` 或 extra 中，不要加入核心包依赖。插件版本不兼容时应抛出带安装说明的 `StartupError`。

启动器从当前 Python 环境发现已安装 entry point。各插件的依赖虽然可以分别写在包元数据中，但同一个进程只有一套 Python 包版本；若两个界面要求互不兼容的同名依赖版本，应将它们安装到不同虚拟环境，再从对应环境启动。启动器不会为每个插件自动建立独立解释器。

## 安装和选择

在仓库根目录安装核心开发包与示例插件：

```bash
python -m pip install -e ".[dev]"
python -m pip install -e examples/gui_plugin
python -m jiuwensymbiosis_gui --list-guis
python -m jiuwensymbiosis_gui --gui example \
  --gui-config examples/gui_plugin/jiuwensymbiosis_example_gui/configs/self-check.json \
  --workspace /tmp/jiuwensymbiosis-example --no-browser
```

工作台插件单独需要 NiceGUI，可安装 `pip install -e ".[gui]"`；仅使用无 NiceGUI 依赖的插件时不需要安装该 extra。统一启动参数、默认工作台以及临时兼容入口见 [GUI 配置和启动](configure-gui.md) 与 [CLI 参考](../reference/cli.md)。

实际任务运行时，用 `--config` 指向机器人运行 YAML：

```bash
python -m jiuwensymbiosis_gui --gui example \
  --config configs/piper/piper.yaml \
  --workspace /tmp/jiuwensymbiosis-example
```

`--gui-config` 仅传给所选 GUI，适合保存布局、主题或插件自己的参数；它不替代 `--config`，也不会自动把真实适配器变成 mock。样例中的 `self-check` 只验证插件选择和 Runtime 的空闲生命周期，不连接机器人。真实执行需要有效的 `--config`，以及对应适配器和模型的运行依赖。

启动器选择插件后只调用其 `main(options)`。插件启动时加载自己的界面依赖，建立 `Runtime`，校验配置并准备 binding，然后启动自身 UI。退出时必须调用 `Runtime.close()` 并等待正在执行的任务完成安全收尾。不要让浏览器连接断开自动释放硬件资源。

## 通过共享 Runtime 执行任务

界面应把用户命令交给统一 Runtime，而不是自己构建 Agent、Planner、Rails 或 `RobotSession`。不同 GUI 通过 Runtime 共享任务状态和本机资源准入；已接入的官方 CLI 目前只共享本机资源准入，仍保留各自执行和输出，不会写入 Runtime 的 job 记录。

下面用 `start_ui_event_loop` 表示插件自己的 UI 生命周期；实际页面回调调用 `submit`、`read_job`、`cancel`，而 Runtime 在整个 UI 存活期间保持打开：

```python
from uuid import uuid4

from jiuwensymbiosis.runtime import Runtime

runtime = Runtime(workspace=options.workspace)
try:
    binding = runtime.prepare_binding(options.config_path)
    def submit(query, request_id=None):
        return runtime.submit_task(
            binding.binding_id,
            request_id=request_id or uuid4().hex,
            query=query,
        )

    def read_job(job_id, after_seq):
        return {
            "snapshot": runtime.get_job(job_id),
            "events": runtime.read_events(job_id, after_seq=after_seq, limit=128),
        }

    def cancel(job_id):
        return runtime.cancel(job_id)

    start_ui_event_loop(submit, read_job, cancel)  # plugin-specific; handlers return promptly
finally:
    closing = runtime.close(timeout=5.0)
    if not closing["closed"]:
        raise RuntimeError("cleanup is unconfirmed; keep this host stopped and disable automatic restart")
```

`submit()`、`read_job()` 和 `cancel()` 是可供 UI handler 调用的短操作；不要在 UI handler 中调用阻塞的 `wait_for_job()`。它适合命令行脚本或测试，不适合事件循环线程。样例在关闭期间会继续等待仍在执行的任务；若 Runtime 最终返回 `closed=false`，会报错退出且要求宿主禁用自动重启。Runtime 不会因 `close()` 超时自动释放资源，也不能把未确认收尾的实例当作正常关闭。

`prepare_binding()` 解析并冻结配置，不连接硬件；每次任务提交时 Runtime 再做资源准入，获准后才创建 session 并执行。binding 固定有效配置、来源路径、工作区和设备资源身份，避免页面改配置或环境变量变化后任务被静默切换到另一台设备。不同 GUI 应使用同一个标准配置和 workspace 语义。

未传工作区时，Runtime 和绑定共用 `runtime.default_workspace()`；需要显示默认目录的插件应调用该接口，避免复制路径字面量。

重跑或编辑后执行时，应将配置快照和它当前的来源路径一起交给 `prepare_binding()`，不要沿用另一份配置的来源目录。设备相关环境变量在有效配置构建时捕获，驱动和子进程也必须使用捕获值；例如 Cruzr 的 `ros_domain_id` 未显式配置时取当时的 `ROS_DOMAIN_ID`（默认 0）。

`request_id` 用于网络重试去重，同一 ID 携带不同内容会冲突。处理准入错误时要区分 `ResourceBusyError`（资源目前正在使用）和 `ResourceBlockedError`（上次收尾未确认）；blocked 状态需要检查并确认真实清理，不能删除占用记录来强行继续。取消是协作式停止，收到取消请求后仍需等待清理结果。

任务终态和 `run_finished` 表示执行结果；资源可复用以任务快照的 `cleanup.released=true` 为准。Runtime 在释放前完成产物登记、结果保存和任务日志 handler 移除。启动取消且清理已确认可正常释放；无法确认子进程退出或力矩恢复时保持 blocked，并保留具体原因和人工处理指引。

事件读取使用 `read_events(job_id, after_seq, limit)`，通过 `next_seq` 续读。每个 job 最多保留最近 10,000 条事件；`gap=true` 表示游标之前的事件已过期，应重新读取当前 job snapshot。Runtime 还提供 `latest_frame()` 和 `read_artifact()` 读取不透明预览/trace 引用：每个 job 至多存 128 个 step frame，每个 artifact 读取上限为 32 MiB。前端应按需读取，并限制自己持有的预览缓存。

读取上限不保证 UI 延迟。展示侧自行负责帧解码与编码；现有工作台暂在 UI 轮询中同步处理，如实测卡顿再移到展示侧线程池，不阻塞核心动作线程。

资源准入覆盖 Runtime 任务以及已经接入统一准入的官方 CLI；CLI 的执行状态和输出仍由各自入口管理，不进入 Runtime job store。仅供可信 Python 工作流使用的维护操作通过 Runtime 的 maintenance 接口协调；不要把 lease、任意回调或维护接口暴露给浏览器客户端。浏览器端只提交受 Runtime 管理的单次任务并查看状态、事件和 artifact。

示例 HTTP 页面只绑定 `127.0.0.1`，并对写请求校验 `Host` 与精确同源 `Origin`，不开放 CORS。采用其他传输方式的插件也必须明确监听范围、写请求来源校验和认证边界；不能把本机可信假设扩展成网络服务假设。

## 依赖和故障隔离

插件入口的 `describe()` 不能依赖所选 UI 框架，因此 `--list-guis` 可在未安装 NiceGUI 或其他可选界面包时显示插件。只有选中插件后才导入它的 `main` 及专属组件；缺少所选插件依赖时给出明确安装命令。

如果新增 GUI 依赖需要与工作台完全不同的 Python 版本或不兼容依赖版本，使用单独虚拟环境安装该插件和核心包。entry point 发现范围是当前解释器，不会自动跨环境导入插件。

## 测试边界

- `tests/unit_tests/` 测核心 Agent、Runtime、适配器、准入和无 GUI 依赖的行为，运行 `make test-core`。
- `tests/gui/` 测启动器、插件安装/发现，以及需要 GUI 可选依赖的界面和接入行为，运行 `make test-gui`。
- `make test` 运行两个无硬件套件；工作台 GUI 测试需要先安装 `pip install -e ".[gui]"`。`make test-all` 还会收集 integration 测试。
- 插件专属的 view/layout 测试放在 `tests/gui/<plugin>/components/`；其他 GUI/plugin 测试放在 `tests/gui/<plugin>/unit/`。Runtime、资源准入与核心行为仍放在 `tests/unit_tests/runtime/`。
- 从 `jiuwensymbiosis.gui` 迁移的测试应改为导入 `jiuwensymbiosis_gui.workbench`；只有兼容期入口测试继续实际执行旧模块转发。不要在核心单元测试中导入 NiceGUI 或具体工作台页面。

旧 `python -m jiuwensymbiosis.gui` 入口目前只用于迁移兼容；新安装、文档和测试使用 `python -m jiuwensymbiosis_gui` 或 `jiuwensymbiosis-gui`。兼容 shim 移除后会以迁移说明公告。
