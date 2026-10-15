# 独立 HTTP 推理服务

Agent 主机只安装客户端；服务器安装所需的模型运行时。视觉、ASR、TTS 可分别部署，
不需要 MCP。串口、USB、相机、音频播放和本体 HTTP 控制仍由 Agent 主机上的原适配器处理。

本文部署的是外部管理的 HTTP 推理服务：在另一台服务器或 Agent 同一主机启动，都用
`detector.mode: remote` 接入，Agent 退出不会停止服务。只有 `detector.mode: local` 下
由框架随 Session 启停的检测子进程称为“本地托管子进程（sidecar）”；两种模式均使用 HTTP。

## Agent 安装与配置

```bash
python -m pip install -e '.[remote]'
# 需要麦克风/扬声器时（Linux 系统另需 PortAudio / libsndfile）
python -m pip install -e '.[remote,voice-io]'
```

将 [remote-client.yaml](remote-client.yaml) 的 `detector` 和 `voice` 合入已有本体配置，
保留硬件、安全、标定字段，替换三个 URL。示例是配置片段，不是独立机器人配置。
服务器默认绑定回环地址；跨主机部署时配置服务器的监听地址和端口，
客户端填写可访问的 HTTP 服务地址，无需配置令牌。

`remote` 不安装 Torch、TorchVision、TorchAudio、FunASR 或 ChatTTS。
OpenJiuwen 的 Transformers 间接依赖仍存在；使用 HTTP 客户端不要求它加载 Torch。
硬件组独立安装，例如 SO-101 的 LeRobot 仍可能带入 Torch；不要给精简 Agent 安装 `full`。

## 准备服务环境与模型

在独立 Python 3.12 环境选择配套的 Torch wheel，以下是 CUDA 12.8 构建示例：

```bash
# 视觉服务器
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[vision-server]'
# ASR 服务器（可用另一个环境）
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[speech-server,voice-local]'
# TTS 服务器（必须使用独立环境）
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[speech-server,chattts-server]'
```

`chattts-server` 固定 ChatTTS 0.2.4 + Transformers 4.53.3。ChatTTS 使用 4.x 的生成缓存接口，与视觉服务器的
`transformers>=5` 不兼容；不要把 `vision-server` 和 `chattts-server` 安装到同一个
Python 环境。Compose 的三个服务分别构建镜像，避免混合这两个模型依赖。

CPU 部署把 wheel 索引换成 `https://download.pytorch.org/whl/cpu`，启动时选择 `--device cpu`。
这只解决安装与设备选择；大模型的 CPU 延迟需要单独测量。
不要把不同版本或 CPU/CUDA 构建的 Torch、TorchVision、TorchAudio 混在同一环境。
GPU 主机还需兼容 CUDA 12.8 的 NVIDIA 驱动；容器使用时另需 NVIDIA Container Toolkit。

模型权重自行按模型发布方的说明和许可准备，记录实际使用的 revision。服务端不替 Agent 下载权重。
示例目录结构如下（不是空目录，需包含完整配置、分词器和权重）：

```text
/srv/jws-models/
  grounding-dino/  # IDEA-Research/grounding-dino-base 的 Transformers 快照
  sam2/            # facebook/sam2.1-hiera-large 的 Transformers 快照
  paraformer/      # FunASR paraformer-zh 本地模型目录
  chattts/         # ChatTTS 0.2.4 source=custom 所需的完整目录（含 asset）
```

直接运行：

```bash
python -m jiuwensymbiosis.serving.grounding_dino_sam2_server \
  --host 127.0.0.1 --port 8114 --device cuda:0 \
  --gdino-model-id /srv/jws-models/grounding-dino --sam2-model-id /srv/jws-models/sam2

python -m jiuwensymbiosis.serving.speech_server \
  --host 127.0.0.1 --port 8115 --device cuda:0 \
  --asr funasr --asr-model-path /srv/jws-models/paraformer

python -m jiuwensymbiosis.serving.speech_server \
  --host 127.0.0.1 --port 8116 --device cuda:0 \
  --tts chattts --tts-model-path /srv/jws-models/chattts
```

一个 speech 进程也可同时指定 ASR/TTS。默认未启用任何模型，必须显式选择。
每个进程同时执行一次模型推理，最多排队 8 个请求，队列等待 30 秒；语音 CLI 可调整
`--max-queue`、`--queue-timeout-s`、请求/响应字节、音频时长和文本长度上限。
视觉请求上限 16 MiB、最多 16,777,216 像素、最多 32 个目标；客户端也独立校验结果。
几个独立进程共用 GPU 时需要自行测量总显存，队列限制不是跨进程的显存调度器。

## Compose 部署

从仓库根目录运行。只开启实际需要的 profile；权重只读挂载，镜像不包含权重。

```bash
export MODEL_ROOT=/srv/jws-models
export MODEL_DEVICE=cuda:0
docker compose -f deploy/inference/compose.yaml -f deploy/inference/compose.gpu.yaml \
  --profile vision --profile asr --profile tts up --build -d
```

CPU 验证省略 GPU overlay，设置 `MODEL_DEVICE=cpu` 和
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` 后重新构建。
端口默认只映射到 `127.0.0.1`；HTTPS 代理可以运行在同一主机。
明确选择专网直连时用 `INFERENCE_BIND` 指定该主机的专网地址，限制可访问的客户端。
健康检查验证服务类型和模型就绪状态。

## 不连接本体的验证

在 Agent 环境运行，可只配置需要测的某一种 remote 后端：

```bash
python deploy/inference/smoke.py --config deploy/inference/remote-client.yaml
python deploy/inference/smoke.py --config path/to/runtime.yaml \
  --image path/to/scene.jpg --prompt 'cup' --audio path/to/command.wav \
  --output-wav /tmp/jws-speech.wav
```

不提供文件时使用固定红色矩形图和 1 秒静音；无目标、无识别文本都是合法结果。
脚本检查请求/响应身份、mask、音频格式，不驱动本体、不打开麦克风、不播放声音；
真实模型质量请用固定标注样本另行测量。WAV 必须是单声道 PCM16，ASR 采样率与服务器配置一致。

成功只表示该组请求完成。正式使用前需在目标两台主机测量冷启动、稳定延迟、网络断连、
排队超时及峰值内存/显存，并按任务调整 `max_frame_age_s` 与 `max_command_age_s`。
连续视觉跟踪的网络预算比静态定位严格，不能由一次静态成功推断。

## 当前验证范围

开发回归覆盖协议、超时/取消、故障与无目标区分、资源清理和配置迁移；
另以干净 Python 3.12 环境安装 `.[remote]` 检查无 Torch 的 Agent 路径。
部署配置已通过 `docker compose config` 检查（含 GPU overlay），三个客户端和容器健康检查脚本
已对临时 HTTP stub 验证成功，包括 WAV 导出与错误服务类型的拒绝。
ChatTTS 0.2.4 + Transformers 4.53.3 已验证导入、构造和随机小型 Llama 的缓存生成；
这项检查不包含真实 ChatTTS 权重或完整音频合成。

依赖审计仍发现 OpenJiuwen 的间接依赖以及可选旧模型栈的已知漏洞。
本次未为消除审计项而强行升级不兼容的模型运行时；
可选服务只加载部署者准备的可信权重，生产上线前还需处理依赖风险。
此仓库交付容器构建和启动模板；本次没有下载模型权重、启动 GPU 容器、
测量跨机推理性能或执行真实本体运动，不将 stub 回归作为这些验证的替代品。
