# 使用远程 HTTP 视觉与语音服务

将 GroundingDINO + SAM2、ASR 和 TTS 部署到独立推理服务器，可以让运行
JiuwenSymbiosis 的 **Agent 主机不配 GPU**。Agent 与本体仍通过原来的串口、USB、CAN、
ROS 或 HTTP 适配器连接；Agent 主机不要求安装在机器人上。

本功能使用固定的 HTTP 推理接口，不需要 MCP。模型服务由部署者独立管理；Agent
仅管理自己的连接，退出时不会停止远端服务。

文档以“HTTP 推理服务”统称模型服务；HTTP 是通信方式。只有 `detector.mode: local` 下
由 Session 启停的检测子进程称为“本地托管子进程（sidecar）”。在本机手动启动服务后，
使用 `remote` 和 `http://127.0.0.1:8114` 即可模拟远程调用；进程仍由你管理。

## 安装与部署

在 Agent 主机安装客户端及实际使用的采集组件：

```bash
pip install -e ".[remote]"
# 使用 RealSense / 本机音频采集与播放时，按需增加：
pip install -e ".[remote,camera,voice-io]"
```

硬件 SDK 仍按适配器要求安装，例如 Piper 的 `.[piper]`。这些客户端安装组不引入视觉、
FunASR 或 ChatTTS 模型。**SO-101 的 LeRobot 安装组仍依赖 Torch**，所以“无需 GPU”
不等于每一种适配器都能完全移除 Torch。已有 `.[full]` / `.[voice]` 保留为本地依赖的兼容入口；
远程 Agent 不应为使用 HTTP 客户端而安装它们。

在模型服务器安装对应的 `vision-server`、`speech-server` 和实际选用的模型安装组，
选择相匹配的 Torch/CUDA 环境。启动命令与模型路径见
[推理服务部署说明](../../../deploy/inference/README.md)。ASR 和 TTS 可以与视觉分开部署。

## 配置远程视觉

保留原配置中的 `adapter`、硬件连接、标定、LLM 和护栏设置，将检测配置改为：

```yaml
detector:
  mode: remote
  endpoint:
    url: https://inference.example/vision
    connect_timeout_s: 3
    request_timeout_s: 30
  max_frame_age_s: null  # 可选：填写正数（秒）才启用额外的图像年龄限制
```

`endpoint` 只需填写 `url`，其值是服务基础地址，不包含 `/v1/segment`。
本地与远程统一使用 `/v1` 接口，无需选择接口版本；连接超时默认 3 秒，请求总超时默认 30 秒。
客户端直接通过 HTTP 调用，无需配置令牌。

`request_timeout_s` 是一次请求的总时限，默认 30 秒。`max_frame_age_s` 不填写或为 `null` 时，
不额外限制图像年龄；填写有限正数时，限制从采集开始到结果可用的总时间，包括同一帧的多次检测和几何计算。
实际请求预算取请求超时与剩余图像有效期中较小的一项；未启用图像年龄限制时只使用请求超时。
该规则适用于本地、远程和兼容入口，tracking 原有的 8 秒图像年龄上限仍独立生效。
Python 兼容入口可写为 `init_detector(url, timeout_s=30, max_frame_age_s=None)`，按需将 `None` 改成正数。
检测失败与“没有找到目标”分别返回；
远端超时不会自动加载本地模型。图像采集、标定变换、深度投影和三维几何仍由 Agent 侧代码完成。

检测有三种显式模式：

| 模式 | 行为 |
|---|---|
| `remote` | 调用 `endpoint`，不启动模型进程、不检查本地权重 |
| `local` | 启动并管理一个可选的本地模型进程，需要模型依赖 |
| `disabled` | 不请求推理；检测调用明确返回不可用，采集和单点投影仍可使用 |

没有检测配置时默认 `disabled`。远程模式也适用于由部署者独立启动的 `localhost` 服务。
`local` 模式端口已占用会报错，不会把未知进程当作本 Session 所有的模型服务。

## 配置远程语音

将下面的 `voice` 块加入同一份运行配置：

```yaml
voice:
  wake_enabled: true
  wake_command_timeout_s: 10
  sample_rate: 16000
  audio_backend: sounddevice
  asr:
    backend: remote
    endpoint:
      url: https://inference.example/speech
      request_timeout_s: 15
    max_audio_duration_s: 30
    max_command_age_s: 20
  tts:
    backend: remote
    endpoint:
      url: https://inference.example/speech
      request_timeout_s: 30
    voice: default
    playback_backend: sounddevice
```

```bash
jiuwensymbiosis-run --config configs/piper/piper.remote.yaml --voice
```

先将编辑后的完整运行配置保存为命令中的文件。录音和播放发生在 Agent 主机；远端
ASR 接收音频并返回文本，TTS 返回音频，不访问本体或 Agent 主机上的麦克风、扬声器。
语音循环采用半双工，播报期间不将自己的声音识别成新任务。过期识别结果不派发动作，
播报失败不重新执行已完成的任务。

ASR 默认 `disabled`，TTS 默认 `null`。使用真实 `--voice` 前必须显式选择 ASR；
仅填写模型名不会隐式启用 FunASR。

## 按需启用本地模型

本地视觉与远程视觉使用同一个客户端结果契约：

```yaml
detector:
  mode: local
  local:
    host: 127.0.0.1
    port: 8114
    device: cuda
    startup_timeout_s: 300
    gdino_model_id: ./models/grounding-dino
    sam2_model_id: ./models/sam2
    use_sam2: true
```

模型字段可以使用 Hub 模型 ID，也可以使用本地目录。`./`、`../`、`~` 或绝对路径按配置
来源解析，Hub ID 保持不变。本地 FunASR 使用 `voice.asr.backend: funasr`，并显式选择
`device: cpu` 或 CUDA 设备；本地 ChatTTS 使用 `voice.tts.backend: chattts` 与
`module_path` 指向已有兼容后端。安装相应依赖后再选择这些后端。

## GUI 与旧配置迁移

工作台的「配置 → 视觉服务」提供模式切换。远程模式显示 URL 和总超时，
可主动点击「检查远程服务」；不会弹出本地权重选择对话框。本地模式显示模型、设备和启动参数。
「禁用视觉服务」会覆盖新旧检测配置。标定与硬件维护会话不启动检测进程、不检查远端服务。

旧 `api_servers` 检测条目仍在兼容期内：显式 `spawn: false` 转为远程模式，
其他已有检测条目按旧本地启动意图解析并告警。不要同时提供新 `detector` 和旧检测条目。
旧配置与 `init_detector(url)` 入口统一使用当前 `/v1` 接口；独立运行的视觉服务需更新到提供该接口的配套版本。
旧语音平铺字段暂时兼容，但应迁移到 `voice.asr` / `voice.tts`；不能混用同一后端的新旧字段。

部署后先检查服务就绪，再分别验证静态定位、连续跟踪与语音延迟。远程化移除了 Agent 主机
上的模型计算负担，实际可用控制频率仍取决于链路和服务器性能。
