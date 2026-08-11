# 示例工程

[English](README.en.md) | 中文

本目录提供可从仓库根目录直接运行的示例。先按照中文 [README](../README.zh.md) 安装依赖。

## Piper Mock Agent

以下命令使用内存机械臂和离线模型验证 Session、Agent、Skills、Tools 与 Rails 的接线，不需要真机、GPU、外部服务或 API Key：

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --mock \
  --max-iter 1 \
  --no-visual-feedback \
  --workspace /tmp/jiuwensymbiosis-agent-demo \
  --query "把黑色盒子放到白色盒子上面"
```

预期结果包含 `"mock: no real model, task skipped"`，退出码为 `0`。固定离线模型不会调用机器人工具，因此该命令只验证 Agent 接线，不模拟物理抓放成功。

## Piper 真机

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --query "把黑色盒子放到白色盒子上面" \
  --api-key "$OPENJIUWEN_API_KEY"
```

运行前必须完成 CAN、Piper SDK、相机、检测服务、工作空间和安全边界验收。不要在未验证的工作空间无人值守运行。

## SO-101 真机

SO-101 需要 Python 3.12、LeRobot 0.6.x、电机标定和有效的眼在手外标定。先复制一份不会提交的本机配置：

```bash
cp configs/so101/so101.yaml configs/so101/so101.local.yaml
```

随包配置包含已验收设备的示例值，复制后应先把 `safety_validated` 改为 `false`，再填写串口、相机序列号、标定路径和本机安全边界；只有完成限位、工作空间和急停验收后，才能重新改为 `true`。然后运行：

```bash
python examples/so101_pick_demo.py \
  --config configs/so101/so101.local.yaml \
  --query "抓起桌面上的香蕉" \
  --fast \
  --no-visual-feedback
```

部署字段和默认值见 [SO-101 配置模板](../jiuwensymbiosis/adapters/so101/config_template.yaml)。

## 样例轨迹

[`sample_trace/`](sample_trace/README.md) 保存一份脱敏的轨迹 JSON、HTML 回放和逐步图像，用于了解 trace 产物，不应作为机器人正确性基准。
