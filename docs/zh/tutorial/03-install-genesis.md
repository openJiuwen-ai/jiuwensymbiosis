# 安装 Genesis 仿真插件

Genesis 仿真作为可选插件安装，通过工作台的「仿真」页使用；核心安装不包含仿真依赖。
本教程统一说明环境准备、安装和首次启动。

## 环境要求

使用 Linux x86_64、Conda 和支持 CUDA、OptiX、NVENC 的 NVIDIA GPU 与驱动。
仓库配置采用 Python 3.12、PyTorch 2.11.0 + CUDA 13.0；当前运行环境为
RTX 5090 D v2、驱动 595.91.07，其他设备需自行验证兼容性。

仿真依赖及已验证的关键版本统一在 [`pyproject.toml`](../../../pyproject.toml) 的 `genesis` extra 声明，
并非完整锁文件。仿真使用独立环境；不要混装根目录 `requirements.txt` 或
`dev`、`full`、`remote`、`vision-server` extras：它们要求 Pillow ≥12，当前仿真需要 Pillow <12。

## 安装

以下命令均在仓库根目录执行。已有同名环境时跳过创建，直接激活：

```bash
conda create -n symbiosis_genesis python=3.12 pip
conda activate symbiosis_genesis
python -m pip install -e '.[gui,genesis]' --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip check
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"'
```

`.[gui,genesis]` 安装工作台和仿真，附加下载源提供 CUDA 版 PyTorch。
`pip check` 应无依赖冲突，CUDA 检查应成功；安装成功后仍需检查实际画面。

## 首次启动

```bash
python -m jiuwensymbiosis_gui --gui workbench
```

1. 打开 <http://127.0.0.1:8770/>，进入「仿真」页。
2. 先选内置「动态物理演示」，无需准备外部场景或机器人资产。
3. 点击「启动仿真」，等待就绪后点击「连接视频」；首次编译可能需要数分钟。
4. 确认画面持续更新、相机操作正常，再按需调整分辨率与画质。请求 FPS 是上限，实际帧率取决于负载。

「断开视频」不会关闭后台；结束使用时点击「关闭仿真」以释放 GPU。
启动失败时查看 `~/.jiuwensymbiosis/genesis-preview/worker.log`，先检查依赖、驱动和 CUDA 状态。

## 使用本地场景与机器人

按[资产配置说明](../../../configs/simulators/README.md)登记本机已有资源并构建索引；
源码运行的 `configs/simulators/assets.yaml` 保存本机路径，不提交，默认索引位于 `outputs/simulator-assets/index.sqlite`。
wheel 安装默认在 `~/.jiuwensymbiosis/simulators/` 下采用相同相对布局；启动前可用绝对路径
`JIUWENSYMBIOSIS_SIMULATOR_HOME` 覆盖数据根。先运行
`python -m jiuwensymbiosis.simulators.assets init-config` 生成示例，再编辑资产路径并 build；
已有配置不会被覆盖。内置材质、灯光 YAML 随 wheel 分发，结果写入数据根下 `outputs/`。
页面操作见 [GUI 仿真器使用指南](../how-to/gui-simulator.md)。

## 接入 RGB-D 检测服务

RGB-D 感知需要 GroundingDINO 和 SAM2 同时可用。推荐把 HTTP 推理服务安装在独立环境中，
仿真环境继续使用 `.[gui,genesis]`，避免混合 Pillow 版本约束。以下命令在仓库根目录的另一终端执行：

```bash
conda create -n symbiosis_vision python=3.12 pip
conda activate symbiosis_vision
python -m pip install -e '.[vision-server]'
python -m pip check
python -m jiuwensymbiosis.serving.grounding_dino_sam2_server \
  --host 127.0.0.1 --port 8114 --device cuda \
  --gdino-model-id IDEA-Research/grounding-dino-base \
  --sam2-model-id facebook/sam2.1-hiera-large
```

按推理设备选择兼容的 PyTorch/CUDA 安装；服务与仿真共用 GPU 时要预留两者的显存。
首次启动会下载模型，也可把两个模型参数改为已下载的本地目录。
在仿真页选择「感知 · RGB-D 目标定位与接近」，检测服务选择「连接已有服务」，
地址填 `http://127.0.0.1:8114`，再点击「应用感知配置」。服务端必须启用 SAM2，不能使用 `--no-sam2`。
服务器由这个终端管理；关闭 Agent 会话不会关闭它。

「本地启动模型」使用工作台当前 Python 环境启动子进程，仅适用于已单独验证推理依赖兼容的环境。
它会检查端口占用，并在结束会话时关闭自己启动的进程；不接管已有服务。
Nyx 投影修正仅覆盖当前已验证版本 `gs-nyx 0.1.7` / `gs-nyx-plugin 0.1.9`，升级后需要重新验证配准。
检测失败、深度缺失或转向空间不可见时，RGB-D 接近会拒绝动作，具体语义见使用指南。
