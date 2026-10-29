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
