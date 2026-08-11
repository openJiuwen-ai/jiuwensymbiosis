# 样例执行轨迹

本目录保存一次 Piper 演示的脱敏执行轨迹：

- `*.json`：结构化 `ExecutionTrace`。
- `*.html`：可直接在浏览器打开的自包含回放。
- `frames/`：轨迹步骤引用的图像帧。

安装本项目后，从仓库根目录进行文本回放：

```bash
jiuwensymbiosis-replay \
  examples/sample_trace/piper-demo-77816242_20260626_113438_033124_1743847.json \
  --text
```

去掉 `--text` 会生成自包含 HTML。预期结果：终端显示步骤时间线，或生成并打印 HTML 回放路径。
