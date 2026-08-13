# 命令行参考

> 类别：Reference。控制台入口由 `pyproject.toml` 的 `[project.scripts]` 定义。

## piper-pick-demo

```bash
piper-pick-demo --config PATH [--query TEXT | --voice ...] [--mock]
```

`--config` 必填；非语音模式必须提供 `--query`。`--mock` 使用离线模型和 Mock 环境。常用覆盖项包括 `--model`、`--server-url`、`--api-key`、`--max-iter`、`--workspace` 和 `--debug`。

## jiuwensymbiosis-replay

```bash
jiuwensymbiosis-replay TRACE_JSON [--open] [--text]
```

默认生成自包含 HTML 回放并打印路径；`--open` 生成后自动用默认浏览器打开；`--text` 输出终端时间线。

## jiuwensymbiosis-gui

```bash
jiuwensymbiosis-gui
# 等价于
python -m jiuwensymbiosis.gui
```

启动监听 `127.0.0.1` 的 NiceGUI 浏览器界面。依赖缺失时，启动前检查会提示安装 `.[gui]`。
