# JiuwenSymbiosis Feature Matrix

> Category: Reference. This page records features present in the current code, built-in adapter support, and activation conditions; it is not a roadmap.

The matrix follows `KNOWN_CAPABILITIES`, Env capability declarations, the action vocabulary (`api/actions.py`), `RobotAgentConfig`, and `pyproject.toml`. See [Architecture](../explanation/architecture.md) for call relationships and the [Robot Adapter Reference](adapter-reference.md) for exact contracts.

## 1. Status legend

| Mark | Meaning |
|---|---|
| ✅ | Direct implementation exists in this repository and is usable with its configuration |
| ◐ | Conditional support requiring an optional dependency, hardware, calibration, or explicit switch |
| ◇ | Framework vocabulary or extension contract exists, but no built-in adapter implements it |
| — | The object does not declare or provide this capability |

Support means that a code path and interface exist; it does not certify every robot, firmware, or workspace. Accept real hardware according to the [porting guide](../how-to/port-hardware-adapter.md).

## 2. Built-in hardware adapter matrix

| Feature | MockArm | Piper | SO-101 | Cruzr |
|---|---|---|---|---|
| Positioning | In-memory 4-DoF simulated arm | AgileX Piper 6-DoF CAN arm | LeRobot SO-101 underactuated 5-DoF arm | Cruzr mobile dual-arm (base + lift + waist) |
| Session entry | `MockArmEnv` + Mock Api/Model | `build_piper_session` | `build_so101_session` | `build_cruzr_session` |
| Cartesian motion | ✅ In-memory pose | ✅ XYZ/R plus full `goto_pose` | ✅ XYZ plus best-effort orientation IK | — |
| Joint motion | — | ✅ Six joints | ✅ Five arm joints | ✅ Dual arms (named joint, radians) |
| Real-time servo | ✅ Simulated sink | ✅ `servo_to_tip`/`servo_to_flange` (driven by fast-path servo ops) | ✅ `servo_to_tip`/`servo_to_flange` (driven by fast-path servo ops) | — |
| Mobile base | — | — | — | ✅ `navigate_relative`/`rotate_base`/`drive_arc`; continuous `base_servo` |
| Lift / waist | — | — | — | ✅ `set_lift_pose`/`lift_to_clearance`/`turn_waist` |
| Target approach | — | — | — | ✅ `approach_for_grasp`/`approach_for_place` (`motion.goal`) |
| Dual-arm coordination | — | — | — | ✅ `dual_arm_grasp`/`dual_arm_place` (`motion.dual_arm`) |
| Parallel gripper | ✅ Simulated state | ✅ Two-state operation with configured width/effort | ✅ Two-state percentage control with conservative contact detection | — |
| Paddle grasp | — | — | — | ✅ `grasp.paddle` (two plates clamp a face each; force-confirmed) |
| Suction | — | — | — | — |
| RGB | ✅ Synthetic image | ◐ Wrist RealSense | ◐ Desktop RealSense D405 | ✅ Waist + head, dual cameras |
| Depth | ◐ Test scenes can provide it, but `vision.depth` is not advertised | ◐ With camera enabled | ◐ With camera enabled | ✅ Waist RGBD |
| Open-vocabulary detection | ✅ Test/simulation path | ◐ Camera plus detector service | ◐ Camera plus detector service | ◐ Camera plus detector service |
| Active search | — | — | — | ✅ `vision.search` + `search_target` (check head/waist cameras at the current heading and report bearing) |
| Camera mounting | Synthetic scene | Eye-in-hand | Eye-to-hand | Eye-to-hand (waist + moving head) |
| Hand-eye transform | Test geometry | `T_base_flange(live) @ T_flange_cam` | Fixed `T_base_cam` | Live `T_base_cam` (TF lookup with static-calibration fallback; head computed from live joints) |
| Reachability (URDF) | — | — | — | ✅ `planning.reachability` (dual-arm + adaptive-lift judge) |
| HTTP detector service | — | ◐ `detector.mode: remote/local` | ◐ `detector.mode: remote/local` | ◐ `detector.mode: remote/local` |
| Owned local detector subprocess (sidecar) | — | ◐ Only `mode: local` | ◐ Only `mode: local` | ◐ Only `mode: local` |
| Main optional dependency | `dev` for tests | `piper`; HTTP client `remote`; RealSense `camera` | Python 3.12 + `so101`; HTTP client `remote`; RealSense `camera` | `cruzr`; HTTP client `remote`; source the ROS workspace at runtime |
| Configuration directory | Tests or examples | `configs/piper/` | `configs/so101/` | `configs/cruzr/` |

Visual capability activation differs:

- Piper's class-level capability set includes vision. With no `camera_serial`, no camera is created; a visual task still requires camera hardware, calibration, and a detector.
- SO-101 derives instance capabilities from `camera_serial`. With no configured camera it does not advertise vision, and a connected Driver can narrow capabilities further when it reports no camera. A configured camera that fails to start makes connection fail closed.
- Cruzr declares waist + head cameras and depth and shares `detector.mode` and the HTTP client with the other official adapters.

Detector configuration defaults to `disabled` for all three adapters. `local` means
an owned local subprocess (sidecar) managed by Session. `remote` means an externally
managed HTTP inference service, including one running independently on `localhost`.
Session closes its remote client without stopping the external service. Install model
dependencies only in the environment that runs the model; hardware connections from
the Agent host remain unchanged. SO-101's LeRobot dependency may still bring in Torch.
See [remote inference](../how-to/remote-inference.md) for configuration and migration.

The client reports connection, timeout, protocol and expired-result
failures through `InferenceServiceError` and `inference_*` codes. Perception actions
retain `error_code` and return `reason="detector_unavailable"`. Service errors never
become an empty result or `no_detection`; that reason means successful inference
found no target.

## 3. Framework Capability matrix

Capability axes are **orthogonal and freely combinable**: motion (cartesian/joint/servo/base/base_servo/lift/waist/goal/dual_arm), sensing (camera/depth/detection/search/eye_to_hand), end effector (parallel/suction/paddle), and planning (reachability) each stand alone.

| Capability | Framework interface | Actions (`ActionSpec`) | MockArm | Piper | SO-101 | Cruzr |
|---|---|---|---|---|---|---|
| `motion.cartesian` | `CartesianDriver` | `goto_xyzr`, `goto_pose`, `move_direction`, `get_pose`, `get_home_pose` | ✅ | ✅ | ✅ | — |
| `motion.joint` | `JointDriver`/`NamedJointDriver` | `move_joint`, `move_named_joint`, `get_joint_positions` | — | ✅ | ✅ | ✅ |
| `motion.servo` | `ServoDriver` + fast-controller hook | — (no standalone public action; driven by fast-path real-time servo ops) | ✅ | ✅ | ✅ | — |
| `motion.base` | Env verbs `navigate_relative`/`navigate_arc` | `navigate_relative`, `rotate_base`, `drive_arc` | — | — | — | ✅ |
| `motion.base_servo` | Env verbs `start_base_drive` etc. | — (continuous base-drive primitive) | — | — | — | ✅ |
| `motion.lift` | Env verb `set_lifter` | `set_lift_pose`, `lift_to_clearance` | — | — | — | ✅ |
| `motion.waist` | Env verb `turn_waist` | `turn_waist` | — | — | — | ✅ |
| `motion.goal` | `motion/approach` | `approach_for_grasp`, `approach_for_place` | — | — | — | ✅ |
| `motion.dual_arm` | `motion/dual_arm` | `dual_arm_grasp`, `dual_arm_place` | — | — | — | ✅ |
| `grasp.parallel` | `GripperDriver` | `open_gripper`, `close_gripper` | ✅ | ✅ | ✅ | — |
| `grasp.paddle` | Body `dual_arm_grasp` | — (implemented via `dual_arm_grasp`) | — | — | — | ✅ |
| `grasp.suction` | `SuctionDriver` | `activate_suction`, `deactivate_suction` | — | — | — | — |
| `vision.camera` | `grab_rgb`/`grab_calibrated_frame` | `get_image`, `pixel_to_base_xyz` | ✅ | ◐ | ◐ | ✅ |
| `vision.depth` | `grab_calibrated_frame` | — (no standalone action) | — | ◐ | ◐ | ✅ |
| `vision.detection` | Calibrated `CameraFrame` + detector hook + `perception/scene3d` | `get_grasp_info_simple`, `locate_for_grasp`, `locate_for_place`, `analyze_scene` | ✅ | ◐ | ◐ | ◐ |
| `vision.eye_to_hand` | Camera-mount marker | — (no standalone action) | — | — | ◐ | — |
| `vision.search` | `motion/approach.search_target` | `search_target` | — | — | — | ✅ |
| `planning.reachability` | `Reachability` (derived) | — (planning-time judge) | — | — | — | ✅ |
| `sorting.command` | Vocabulary and adapter extension point | — (no built-in generic tool) | — | — | — | — |
| `speech.tts` | Vocabulary marker; voice front end provides TTS separately | — (no built-in robot action) | — | — | — | — |

The framework fully defines `grasp.suction`, but this repository has no built-in real suction adapter. The Tutorial's SCARA-with-suction implementation is educational and is not hardware-accepted built-in support.

## 4. Execution and tool strategy matrix

| Feature | Status | Default | Activation or constraint |
|---|---|---|---|
| fastagent compile-once | ✅ | `exec_mode="fastagent"` | One model plan (tiered skill + action composition) then execute sequentially, no per-step LLM; `--stepagent` forces per-step |
| Runtime re-planning | ✅ | Before each step | Re-plan when the world contradicts the next step's pre-conditions, capped by `max_replans` |
| Individual tool mode | ✅ | Optional | `mode="tool"`; each `@implements` method becomes one tool |
| Code mode | ✅ | Optional | `mode="code"`; provides `InProcessCodeTool` |
| Hybrid mode | ✅ | `mode="hybrid"` | Provides individual and code tools together |
| Skill workflows | ◐ | `enable_skill=False` | Enables `SkillUseRail` and `RobotControlTool`; built-in `visual_pick`/`visual_place`/`transport` |
| Custom tools/Rails | ✅ | None | Inject through `extra_tools` and `extra_rails` |
| Parallel tool calls | ◐ | `parallel_tool_calls=False` | Only for audited non-motion tools; motion/grasp rejects it, and it cannot run with Trace |
| No-hardware/no-model dry run | ✅ | With `--mock` | `MockArmEnv` + `MockModelClient` (factory `build_mock_model`, Piper only); no CAN, camera, or model endpoint |

## 5. Rails, Trace, and feedback matrix

| Capability | Status | Default | Automatic condition or dependency |
|---|---|---|---|
| SafetyRail | ✅ | On | Env has any motion capability (cartesian/joint/base/lift/waist); checks derived from declared capability |
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
| GroundingDINO text detection | ◐ | Agent installs `remote` to call HTTP inference; the model environment installs `vision-server` |
| SAM2 masks | ◐ | Enabled on the inference server; local configuration uses `detector.local.use_sam2=true` |
| HTTP client lifecycle | ✅ | Session injects and closes clients with `managed_detector=True`; remote servers are externally managed |
| Owned local subprocess lifecycle | ✅ | `make_detector_sidecar()` starts/stops with Session only in `mode: local` |
| Mask centroid and median depth | ✅ | `scene3d.locate_for_grasp`/`analyze_scene` |
| Eye-in-hand projection | ✅ | Piper implementation; needs `T_flange_cam` and live flange pose |
| Eye-to-hand projection | ✅ | SO-101 uses fixed `T_base_cam`; Cruzr prefers live TF with static-calibration fallback |
| Multi-point/translation XY correction | ✅ | `apply_xy_correction()`; multi-point transform takes priority |
| Grasp and place heights | ✅ | `grasp_z_offset_mm` and `place_z_offset_mm` applied uniformly |
| Active search | ✅ | `motion/approach.search_target`: check cameras at the current heading, report bearing, and do not move; `approach_*` performs the sweep |
| Reachability prior | ◐ | Derived `planning.reachability` when the body ships a URDF + arm chains; Cruzr provides a dual-arm + lift judge |
| Hand-eye calibration script | ◐ | Install `calib`; see [Calibrate Hand-Eye Geometry](../how-to/calibrate-hand-eye.md) for Piper and [SO-101 Fixed-Camera Calibration](../how-to/calibrate-so101-eye-to-hand.md) for SO-101 |

## 7. User entry points and optional dependencies

| Entry point or feature | Status | Install/command |
|---|---|---|
| Python API | ✅ | Import `jiuwensymbiosis` after core installation |
| Generic task runner | ✅ | `jiuwensymbiosis-run --config <robot>.yaml --query "<task>"` |
| Piper adapter | ◐ | `pip install -e ".[piper]"` |
| SO-101 adapter | ◐ | Python 3.12; `pip install -e ".[so101]"` |
| Cruzr adapter | ◐ | `pip install -e ".[cruzr]"`; source the ROS workspace at runtime |
| Remote HTTP inference clients | ◐ | `pip install -e ".[remote]"`; installs no vision/speech models |
| Optional vision model service | ◐ | Install `.[vision-server]` on the model host with compatible Torch/CUDA; Agent owns a subprocess only in `mode: local` |
| Browser GUI | ◐ | `pip install -e ".[gui]"`; `jiuwensymbiosis-gui --gui workbench`, default `127.0.0.1:8770`; `--list-guis` lists installed plugins |
| Voice front end | ◐ | Remote uses `.[remote,voice-io]`; ASR defaults to disabled, TTS to null; local FunASR/ChatTTS are explicit options |
| Hand-eye calibration | ◐ | `pip install -e ".[calib,piper]"` or `.[calib,so101]` |
| Actions/skills/state introspection | ✅ | `jiuwensymbiosis-actions` / `-skills` / `-state` |
| Trace replay | ✅ | `jiuwensymbiosis-replay` |
| Core tests | ✅ | `pip install -e ".[dev]"`; `make test-core` |
| GUI tests | ◐ | `pip install -e ".[dev,gui]"`; `make test-gui` |

`make test` defaults to core tests and requires only `.[dev]`. To run both no-hardware
suites, install `.[dev,gui]` and use `make test test-gui`. `make test-all` also includes
integration tests and requires the dependencies and environment for those tests.
The `.[gui]` extra supplies the workbench UI dependencies; other installed GUI plugins supply their own UI dependencies.

When maintaining this matrix, check these sources of truth together:

- [`env/base.py`](../../../jiuwensymbiosis/env/base.py): Capability vocabulary;
- [`api/actions.py`](../../../jiuwensymbiosis/api/actions.py): action vocabulary (`ActionSpec`);
- [`adapters/_common/capability_spec.py`](../../../jiuwensymbiosis/adapters/_common/capability_spec.py): capability→action mapping;
- [Piper Env](../../../jiuwensymbiosis/adapters/piper/env.py), [SO-101 Env](../../../jiuwensymbiosis/adapters/so101/env.py), [Cruzr Env](../../../jiuwensymbiosis/adapters/cruzr/env.py): adapter declarations;
- [`agent/config.py`](../../../jiuwensymbiosis/agent/config.py) and [`agent/builder.py`](../../../jiuwensymbiosis/agent/builder.py): execution and Rail switches;
- [`pyproject.toml`](../../../pyproject.toml): Python requirements, optional dependencies, and CLI entry points.
