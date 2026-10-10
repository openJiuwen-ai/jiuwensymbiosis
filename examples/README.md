# 示例工程

[English](README.en.md) | 中文

本目录提供可从仓库根目录直接运行的示例。先按照中文 [README](../README.zh.md) 安装依赖。

## 通用任务运行器

`examples/run_task.py` 是所有机器人和任务的统一入口：用 `--config` 选机器人（YAML 顶层 `adapter:` 字段从注册表选中），用 `--query`（或 `--voice`）给任务，任务不在 config 里。执行模式取 YAML `agent.execution.mode`，默认 `fastagent`（编译一次、无逐步 LLM）；单步调试加 `--stepagent`。

下面两个示例都在真机上验证过。运行前必须完成硬件、标定、检测服务和安全边界验收，不要在未验证的工作空间无人值守运行。

### SO-101 真机

SO-101 需要 Python 3.12、LeRobot 0.6.x、电机标定和有效的眼在手外标定。

`configs/so101/so101.yaml` 包含已验收设备的示例值。动手前先把 `safety_validated` 改为 `false`，再填写串口、相机序列号、标定路径和本机安全边界；只有完成限位、工作空间和急停验收后，才能重新改为 `true`。然后运行：

```bash
python examples/run_task.py \
  --config configs/so101/so101.yaml \
  --query "把香蕉放到盘子里" \
  --api-key "$OPENJIUWEN_API_KEY"
```

部署字段和默认值见 [SO-101 配置模板](../jiuwensymbiosis/adapters/so101/config_template.yaml)。

### Cruzr 真机

Cruzr 需要 ROS 2（Jazzy）工作区、腰部/头部相机、标定和检测服务。

在 `configs/cruzr/cruzr.yaml` 里保持 `adapter:` 字段为 `cruzr`；填写 ROS 工作区路径、相机话题、标定文件、URDF 路径、检测服务地址与编排 LLM 端点，并验收安全边界。运行前需 `source` ROS + Cruzr 工作区并启动检测服务。然后运行：

```bash
python examples/run_task.py \
  --config configs/cruzr/cruzr.yaml \
  --query "把棕色箱子上的白色箱子搬到有香蕉的白桌子上" \
  --api-key "$OPENJIUWEN_API_KEY"
```

> 不想让串口、序列号和密钥出现在 `git status` 里，可以复制成 `configs/<机器人>/<机器人>.local.yaml` 再改——`.gitignore` 里的 `*.local.yaml` 会忽略它，`--config` 指向该副本即可。

Cruzr 声明了 `motion.base`，`transport` 技能因此通过能力门；固定臂机型不具备移动能力，同一条任务编排出的序列并不相同。

## 样例轨迹

[`sample_trace/`](sample_trace/README.md) 保存一份脱敏的轨迹 JSON、HTML 回放和逐步图像，用于了解 trace 产物，不应作为机器人正确性基准。
