# JiuwenSymbiosis Feature Matrix

> Category: Reference. This page records features present in the current code, built-in adapter support, and activation conditions; it is not a roadmap.

The matrix follows `KNOWN_CAPABILITIES`, Env capability declarations, Api Mixin composition, `RobotAgentConfig`, and
`pyproject.toml`. See [Architecture](../explanation/architecture.md) for call relationships and the
[Robot Adapter Reference](adapter-reference.md) for exact contracts.

## 1. Status legend

| Mark | Meaning |
|---|---|
| ✅ | Direct implementation exists in this repository and is usable with its configuration |
| ◐ | Conditional support requiring an optional dependency, hardware, calibration, or explicit switch |
| ◇ | Framework vocabulary or extension contract exists, but no built-in adapter implements it |
| — | The object does not declare or provide this capability |

Support means that a code path and interface exist; it does not certify every robot, firmware, or workspace. Accept real
hardware according to the [porting guide](../how-to/port-hardware-adapter.md).

## 2. Built-in hardware adapter matrix

| Feature | MockArm | Piper | SO-101 |
|---|---|---|---|
| Positioning | In-memory 4-DoF simulated arm | AgileX Piper 6-DoF CAN arm | LeRobot SO-101 underactuated 5-DoF arm |
| Session entry | `MockArmEnv` + Mock Api/Model | `build_piper_session` | `build_so101_session` |
| Cartesian motion | ✅ In-memory pose | ✅ XYZ/R plus full `goto_pose` | ✅ XYZ plus best-effort orientation IK |
| Joint motion | — | ✅ Six joints | ✅ Five arm joints |
| Real-time servo | ✅ Simulated sink | ✅ `servo_to_tip`/`servo_to_flange` | ✅ `servo_to_tip`/`servo_to_flange` |
| Parallel gripper | ✅ Simulated state | ✅ Two-state operation with configured width/effort | ✅ Two-state percentage control with conservative contact detection |
| Suction | — | — | — |
| RGB | ✅ Synthetic image | ◐ Wrist RealSense | ◐ Desktop RealSense D405 |
| Depth | ◐ Test scenes can provide it, but `vision.depth` is not advertised | ◐ With camera enabled | ◐ With camera enabled |
| Open-vocabulary detection | ✅ Test/simulation path | ◐ Camera plus detector service | ◐ Camera plus detector service |
| Camera mounting | Synthetic scene | Eye-in-hand | Eye-to-hand |
| Hand-eye transform | Test geometry | `T_base_flange(live) @ T_flange_cam` | Fixed `T_base_cam` |
| Detector sidecar | — | ◐ `detector.spawn=true` | ◐ `detector.spawn=true` |
| Fast-path tracking | ◐ Test path | ◐ `track_detect` | ◐ `track_grasp` |
| Main optional dependency | `dev` for tests | `piper`; add `full` for vision | Python 3.12 + `so101`; add `full` for vision |
| Configuration directory | Tests or examples | `configs/piper/` | `configs/so101/` |

Visual capability activation differs:

- Piper's class-level capability set includes vision. With no `camera_serial`, no camera is created; a visual task still
  requires camera hardware, calibration, and a detector.
- SO-101 derives instance capabilities from `camera_serial`. With no configured camera it does not advertise vision,
  and a connected Driver can narrow capabilities further when it reports no camera. A configured camera that fails to
  start makes connection fail closed.

## 3. Framework Capability matrix

| Capability | Framework interface | LLM tools | MockArm | Piper | SO-101 |
|---|---|---|---|---|---|
| `motion.cartesian` | `MotionMixin` + `RobotDriver` | `home`, `get_pose`, `goto_xyzr`, and related tools | ✅ | ✅ | ✅ |
| `motion.joint` | `JointMotionMixin` + `JointDriver` | `move_joint` | — | ✅ | ✅ |
| `motion.servo` | `ServoDriver` + fast-controller hook | No standalone public Mixin tool | ✅ | ✅ | ✅ |
| `grasp.parallel` | `ParallelGripperMixin` + `GripperDriver` | `open_gripper`, `close_gripper` | ✅ | ✅ | ✅ |
| `grasp.suction` | `SuctionMixin` + `SuctionDriver` | `activate_suction`, `deactivate_suction` | — | — | — |
| `vision.camera` | Env/Driver marker | Used through `VisionMixin.get_image` | ✅ | ◐ | ◐ |
| `vision.depth` | Env/Driver marker | No standalone tool | — | ◐ | ◐ |
| `vision.detection` | `VisionMixin` + raw projection seam | Grasp information, pixel projection, image | ✅ | ◐ | ◐ |
| `vision.eye_to_hand` | Camera-mount marker | No standalone tool | — | — | ◐ |
| `sorting.command` | Vocabulary and adapter extension point | No built-in generic tool | — | — | — |
| `speech.tts` | Vocabulary marker; voice front end provides TTS separately | No built-in robot tool | — | — | — |

The framework fully defines `grasp.suction`, but this repository has no built-in real suction adapter. The Tutorial's
SCARA-with-suction implementation is educational and is not hardware-accepted built-in support.

## 4. Execution and tool strategy matrix

| Feature | Status | Default | Activation or constraint |
|---|---|---|---|
| Per-step DeepAgent execution | ✅ | `exec_mode="agent"` | The model plans and calls tools at every step |
| Fast path | ◐ | Off | `exec_mode="fast"`; compile once then execute sequentially; tracking also needs servo/vision/grasp capabilities and adapter hooks |
| Individual tool mode | ✅ | Optional | `mode="tool"`; each `@robot_tool` becomes one tool |
| Code mode | ✅ | Optional | `mode="code"`; provides `InProcessCodeTool` |
| Hybrid mode | ✅ | `mode="hybrid"` | Provides individual and code tools together |
| Skill workflows | ◐ | `enable_skill=False` | Enables `SkillUseRail` and `RobotControlTool`; built-in `visual_pick`/`visual_place` |
| Custom tools/Rails | ✅ | None | Inject through `extra_tools` and `extra_rails` |
| Parallel tool calls | ◐ | `parallel_tool_calls=False` | Only for audited non-motion tools; motion/grasp rejects it, and it cannot run with Trace |
| No-hardware/no-model dry run | ✅ | With `--mock` | `MockArmEnv` + `MockModel`; no CAN, camera, or model endpoint |

## 5. Rails, Trace, and feedback matrix

| Capability | Status | Default | Automatic condition or dependency |
|---|---|---|---|
| SafetyRail | ✅ | On | Cartesian or joint motion; checks Z, XY, and joint soft limits |
| RecoveryRail | ✅ | On | Motion, suction, or gripper; attempts Home and release after failure |
| VisualFeedbackRail | ◐ | On | Also requires `vision.camera`; stages a post-action frame |
| SkillUseRail | ◐ | Off | `enable_skill=True` |
| TraceRail | ◐ | Off | `enable_tracing=True`; records tools, observations, Rail events, logs, and optional frames |
| DiagnosisRail | ◐ | Off | `enable_diagnosis=True` and tracing must also be enabled |
| WARNING+ logs in Trace | ◐ | With Trace | `trace_capture_loggers` defaults to `jiuwensymbiosis` |
| Trace HTML/text replay | ✅ | On demand | `jiuwensymbiosis-replay <trace.json>`; default output is self-contained HTML |
| Offline Trace Feedback | ✅ | On demand | `scripts/analyze_traces.py` clusters failures and produces human-review proposals |
| Central logging | ✅ | INFO + `./logs` | Console plus rotating file; `log_dir=None` is console-only |

## 6. Vision and perception matrix

| Feature | Status | Implementation or condition |
|---|---|---|
| RGB plus aligned depth | ◐ | RealSense/adapter Driver; depth boundary is meters |
| GroundingDINO text detection | ◐ | Install `full` and start or connect the detector service |
| SAM2 masks | ◐ | Detector configuration `use_sam2=true` |
| Detector sidecar lifecycle | ✅ | `make_detector_sidecar()` follows Session lifecycle |
| Mask centroid and median depth | ✅ | `detect_and_centroid()` |
| Eye-in-hand projection | ✅ | Piper implementation; needs `T_flange_cam` and live flange pose |
| Eye-to-hand projection | ✅ | SO-101 implementation; needs fixed `T_base_cam` |
| Multi-point/translation XY correction | ✅ | `apply_xy_correction()`; multi-point transform takes priority |
| Grasp and place heights | ✅ | Shared `VisionMixin` applies `grasp_z_offset_mm` and `place_z_offset_mm` |
| Hand-eye calibration script | ◐ | Install `calib`; see [Calibrate Hand-Eye Geometry](../how-to/calibrate-hand-eye.md) for Piper |

## 7. User entry points and optional dependencies

| Entry point or feature | Status | Install/command |
|---|---|---|
| Python API | ✅ | Import `jiuwensymbiosis` after core installation |
| Piper demo | ◐ | `pip install -e ".[piper]"`; `piper-pick-demo` |
| SO-101 adapter | ◐ | Python 3.12; `pip install -e ".[so101]"` |
| Vision/GPU | ◐ | `pip install -e ".[full]"` with the CUDA 12.8 PyTorch index |
| Browser GUI | ◐ | `pip install -e ".[gui]"`; `jiuwensymbiosis-gui`, default `127.0.0.1:8770` |
| Voice front end | ◐ | `pip install -e ".[voice]"`; optional FunASR/capture, default `NullTTS` |
| Hand-eye calibration | ◐ | `pip install -e ".[calib,piper]"` |
| Trace replay | ✅ | `jiuwensymbiosis-replay` |
| Unit tests | ✅ | `pip install -e ".[dev]"`; `pytest tests/unit_tests/` |

When maintaining this matrix, check these sources of truth together:

- [`env/base.py`](../../../jiuwensymbiosis/env/base.py): Capability vocabulary;
- [`api/mixins.py`](../../../jiuwensymbiosis/api/mixins.py): public tool capabilities;
- [Piper Env](../../../jiuwensymbiosis/adapters/piper/env.py) and [SO-101 Env](../../../jiuwensymbiosis/adapters/so101/env.py): adapter declarations;
- [`agent/config.py`](../../../jiuwensymbiosis/agent/config.py) and [`agent/builder.py`](../../../jiuwensymbiosis/agent/builder.py): execution and Rail switches;
- [`pyproject.toml`](../../../pyproject.toml): Python requirements, optional dependencies, and CLI entry points.
