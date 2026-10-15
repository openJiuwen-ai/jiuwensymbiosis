# Use remote HTTP vision and speech services

Deploying GroundingDINO + SAM2, ASR and TTS on independent inference servers allows the
**JiuwenSymbiosis Agent host to run without a GPU**. The Agent still connects to physical
hardware through its existing serial, USB, CAN, ROS or HTTP adapter. The Agent host does
not have to be installed on the robot.

This integration uses fixed HTTP inference APIs and does not require MCP. Operators own
the remote servers; the Agent owns its client connections and never stops a remote server
when a session ends.

“HTTP inference service” is the general term; HTTP describes communication. A “managed local
subprocess (sidecar)” specifically means a detector started and stopped by the Session in
`detector.mode: local`. To simulate remote calls with a manually started localhost service,
use `remote` and `http://127.0.0.1:8114`; you still own the server process.

## Install the Agent and deploy servers

Install only the client and capture components needed on the Agent host:

```bash
pip install -e ".[remote]"
# Add these when using RealSense and host audio capture/playback:
pip install -e ".[remote,camera,voice-io]"
```

Install hardware SDKs separately, such as `.[piper]` for Piper. These client extras do not
install the vision, FunASR or ChatTTS models. **The SO-101 LeRobot extra still depends on
Torch**, so GPU independence does not imply that every hardware adapter can omit Torch.
The existing `.[full]` and `.[voice]` extras remain compatibility options for local model
dependencies; they are unnecessary for a remote HTTP client.

On inference servers, install `vision-server` or `speech-server` together with the model
extras actually selected, using compatible Torch/CUDA wheels. Follow the
[inference deployment instructions](../../../deploy/inference/README.md) for startup commands and
model paths. Vision, ASR and TTS may use different servers.

## Configure remote vision

Keep the adapter, hardware, calibration, LLM and rail settings in your runtime YAML.
Replace its detector configuration with:

```yaml
detector:
  mode: remote
  endpoint:
    url: https://inference.example/vision
    connect_timeout_s: 3
    request_timeout_s: 30
  max_frame_age_s: null  # Optional: set a positive number of seconds to enforce a frame-age limit
```

Only `endpoint.url` is required; it is the service base, without `/v1/segment`.
Local and remote clients use the same `/v1` interface, with no version selector.
Connection and total request timeouts default to 3 and 30 seconds. Clients call the HTTP service directly;
no token configuration is required.

`request_timeout_s` bounds the entire request and defaults to 30 seconds. Omitting
`max_frame_age_s` or setting it to `null` adds no frame-age limit. A finite positive
value bounds the time from capture until the result is usable, including multiple
prompts on the same frame and local geometry calculations. Each request uses the
smaller of its request timeout and the remaining frame-age budget, when enabled.
This applies to local, remote and compatibility clients; tracking's existing
8-second image-age limit still applies independently. The Python compatibility
entry is `init_detector(url, timeout_s=30, max_frame_age_s=None)`; set the optional
age limit to a positive value when required. Service failure is distinct from a
successful detection with no objects. Remote failure never loads a local model as a
fallback. Capture, calibration transforms, depth projection and 3D geometry remain in
the Agent framework.

| Mode | Behavior |
|---|---|
| `remote` | Calls `endpoint`; starts no model process and checks no local weights |
| `local` | Starts and owns an optional local model process; requires model dependencies |
| `disabled` | Sends no inference requests; detection reports unavailable while capture and pixel projection remain available |

Omitted detector settings default to `disabled`. Use `remote` even for an independently
managed service on `localhost`. Local mode rejects an occupied port instead of taking
ownership of an unknown process.

## Configure remote speech

Add this block to the same runtime YAML:

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

Save the complete configuration at the path used in this command:

```bash
jiuwensymbiosis-run --config configs/piper/piper.remote.yaml --voice
```

Capture and playback happen on the Agent host. Remote ASR returns text; remote TTS
returns audio. Neither accesses the robot or the Agent's microphone/speaker. The voice
loop is half duplex. Expired transcripts never dispatch actions, and a playback failure
never reruns a completed task.

ASR defaults to `disabled` and TTS to `null`. Select an ASR backend before using real
`--voice` capture; specifying a model name alone does not enable FunASR.

## Enable local models explicitly

Local vision uses the same client result contract:

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

Model IDs accept Hub IDs or local directories. Path-like values beginning with `./`,
`../`, `~` or an absolute path resolve against the configuration source; Hub IDs stay
unchanged. For local FunASR, use `voice.asr.backend: funasr` and explicitly select
`device: cpu` or a CUDA device. For an existing compatible local ChatTTS backend,
use `voice.tts.backend: chattts` and its `module_path`. Install those model dependencies
before selecting them.

## Workbench and migration

The workbench's vision settings switch between modes. Remote mode shows its URL and total timeout, with an explicit readiness check button. It does not prompt for local weights. Local mode shows model, device
and startup settings. The disable-vision switch overrides both new and legacy settings.
Hardware maintenance and calibration sessions start no detector process, contact no
inference server.

Legacy `api_servers` detector entries remain temporarily supported with a warning.
Explicit `spawn: false` selects remote mode; other existing detector
entries retain their local-start intent. Do not combine a new `detector` block with a
legacy detector entry. Migrated configurations and `init_detector(url)` use the current
`/v1` interface; independently running vision servers must be updated to a matching version.
Legacy flat voice fields remain supported during
migration; move them to `voice.asr` / `voice.tts` and do not mix both forms for one backend.

After deployment, check readiness and measure static localization, continuous tracking
and speech latency separately. Moving model computation off the Agent host does not
guarantee a particular control frequency across the network.
