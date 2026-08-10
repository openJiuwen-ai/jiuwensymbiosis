# 安装与快速开始

> 类别：Tutorial。本教程带你在无机器人、无 GPU、无真实 LLM 的环境中初始化 JiuwenSymbiosis Agent。

## 1. 准备环境

当前验证平台是 Ubuntu 22.04，Python 必须满足 `>=3.11,<3.14`。SO-101 适配器使用 LeRobot 0.6.x，因此需要 Python 3.12。

```bash
git clone https://gitcode.com/openJiuwen/jiuwensymbiosis.git
cd jiuwensymbiosis
conda create -n jiuwensymbiosis python=3.12
conda activate jiuwensymbiosis
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. 运行 Piper Mock Agent

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --mock \
  --max-iter 1 \
  --no-visual-feedback \
  --workspace /tmp/jiuwensymbiosis-demo \
  --query "把黑色盒子放到白色盒子上面"
```

该命令使用 `MockArmEnv` 和离线 `MockModelClient`，会建立 `RobotSession`、构造 Agent、加载内置技能与工具并执行一次模型调用，不连接 CAN、相机、检测服务或模型端点。

## 3. 确认结果

最终结果应包含：

```text
"mock: no real model, task skipped"
```

程序以退出码 `0` 结束。这是 Agent 接线冒烟测试：固定离线模型会直接返回最终答复，不调用机器人工具，因此 Mock 模式不代表物理操作成功。

## 4. 保持代理导入安全

HTTP 代理环境变量可能破坏本地 vLLM 和检测调用。自行编写 Python 入口时，必须在导入 `openjiuwen` 或任何间接导入它的模块前清理代理：

```python
from jiuwensymbiosis.utils.proxy import clear_proxy_env

clear_proxy_env()

# 在下方导入 openjiuwen 或其余 JiuwenSymbiosis Agent 模块。
```

仓库内置 CLI 和示例已经执行这一步。

## 5. 安装可选能力

```bash
# 测试与开发工具
python -m pip install -e ".[dev]"

# 视觉与 GPU
python -m pip install -e ".[full]" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Piper SDK
python -m pip install -e ".[piper]"

# SO-101 / LeRobot（Python 3.12）
python -m pip install -e ".[so101]"

# ASR 与录音
python -m pip install -e ".[voice]"

# 浏览器 GUI
python -m pip install -e ".[gui]"

# 手眼标定
python -m pip install -e ".[calib]"
```

可选依赖可以组合安装，例如 `.[full,piper]` 或 `.[full,so101]`；凡是包含 `[full]` 的命令，仍需使用上方 PyTorch CUDA 12.8 附加源。

## 6. 安装固定版本的运行依赖

[`requirements.txt`](../../../requirements.txt) 固定了项目直接依赖的完整运行版本，包括视觉/GPU 栈；它不是完整的环境锁文件，因为传递依赖、开发工具和 Piper SDK 并未全部锁定。若要安装这组运行依赖，再安装项目本身且不重新解析依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

首次运行真机前必须验收适配器连接、限位、急停、标定和工作空间。下一步可阅读[构建第一个机器人适配器](02-build-first-adapter.md)或[图形界面配置](../how-to/configure-gui.md)。
