"""Consume the official Jumper controller bundle, without its training stack."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import sys
from collections import deque
from pathlib import Path
from tomllib import loads

import numpy as np

from ..assets.policy_files import JUMPER_EXPORTS
from .policy_actions import RECORDINGS


class JumperPolicy:
    def __init__(self, manifest_path: Path, assets: dict[str, Path] | None = None):
        import onnxruntime as ort

        self.root = manifest_path.parent.resolve()
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema") != "kk-policy-bundle/1":
            raise ValueError("Unsupported Jumper bundle schema")
        self.contracts = {
            name: self.file(entry["contract"]).read_text() for name, entry in self.manifest["modes"].items()
        }
        self.config = self.file("controller.toml").read_text()
        self.models = {name: self.file(entry["models"]["onnx"]) for name, entry in self.manifest["modes"].items()}
        self.trajectories = {
            name: self.file(entry["trajectory"]).read_text()
            for name, entry in self.manifest["modes"].items()
            if entry.get("trajectory")
        }
        self._add_exports(assets or {})
        parsed = {name: json.loads(text) for name, text in self.contracts.items()}
        self.contract = parsed["locomotion"]
        self.wire = self.contract["wire_joint_order"]
        if len(set(self.wire)) != len(self.wire):
            raise ValueError("Duplicate Jumper wire joints")
        for contract in parsed.values():
            if contract["wire_joint_order"] != self.wire or contract["joint_limits"] != self.contract["joint_limits"]:
                raise ValueError("Jumper modes disagree on joint order or limits")
        self.hz = max(c["control"]["control_hz"] for c in parsed.values())
        self.dt = float(self.contract["control"]["sim_dt"])
        if not 0 < self.dt <= 0.005 or abs(1 / (self.dt * self.hz) - round(1 / (self.dt * self.hz))) > 1e-6:
            raise ValueError("Jumper physics and controller rates must be commensurate")
        key = "macosx-universal2" if sys.platform == "darwin" else f"{sys.platform}-{platform.machine().lower()}"
        runtime = self.manifest["runtimes"]["mjlab"]
        if key not in runtime["extensions"]:
            raise ValueError(f"Jumper bundle has no controller for {key}")
        path = self.file(runtime["extensions"][key]["file"])
        loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
        spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Jumper controller: {path}")
        self.extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.extension)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        # Models, native hooks, gains and trajectories remain the bundled versions.
        self.sessions: dict[Path, ort.InferenceSession] = {}
        for path in self.models.values():
            if path not in self.sessions:
                self.sessions[path] = ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
        self.parity = self.verify_reference()
        self.fsm = self.new_fsm()
        self.operators = False
        self.key_events: deque[tuple[str, bool]] = deque()
        self.pressed_keys: set[str] = set()
        self.posture: dict[str, float] = {}
        self.arm_key: str | None = None
        self.trigger_axes: dict[str, float] = {}
        self.send_trigger_frame = False
        self.durations = {name: parsed[name]["reference"]["duration"] for name in RECORDINGS if name in parsed}
        self.claw_controls = {
            side: parsed["claw_" + side]["controller"]["devices"]["keyboard"] for side in ("left", "right")
        }
        config = loads(self.config)["fsm"]
        self.modifiers = {e["name"]: "key_left_" + e["with"] for e in config["button"] if e.get("key") and "with" in e}
        self.mode_keys = {
            entry["name"]: entry["key"]
            for entry in config["button"]
            if entry.get("key") and entry.get("on") == "toggle"
        }
        self.claw_keys = {
            side: parsed["claw_" + side]["controller"]["devices"]["keyboard"]["task"]["claw_" + side][0]
            for side in ("left", "right")
        }
        self.claw_triggers = {
            side: parsed["claw_" + side]["controller"]["devices"]["gamepad"]["task"]["claw_" + side]
            for side in ("left", "right")
        }
        self.summary = {
            "controller_commit": runtime["commit"],
            "bundle_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "backend": "ONNX Runtime CPU / official native FSM",
            "exposed_modes": list(self.contracts),
            "notes": self.manifest.get("notes", []),
            "reference_check": self.parity,
        }

    def _add_exports(self, paths: dict[str, Path]) -> None:
        # Extend the host bindings for the two separately published examples.
        # Keep every bundled rule, including tilt and stale-feedback priorities.
        fallback = '[[fsm.rule]]\nwhen = "always"'
        for mode, key in (("dance_waist", "keypad_5"), ("ref_free_jump", "keypad_6")):
            if f"{mode}/layout.json" not in paths:
                continue
            for filename, digest in JUMPER_EXPORTS[mode].items():
                if hashlib.sha256(paths[f"{mode}/{filename}"].read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Jumper export digest mismatch: {mode}/{filename}")
            if self.config.count(fallback) != 1:
                raise ValueError("Unsupported Jumper FSM fallback")
            rule = f'[[fsm.rule]]\nwhen = "button:{mode}"\nenter = "{mode}"\n\n'
            self.config = self.config.replace(fallback, rule + fallback)
            self.config += (
                f'\n[[fsm.state]]\nname = "{mode}"\nmodel = "{mode}.onnx"\ntask = "jumper.{mode}"\n'
                f'\n[[fsm.button]]\nname = "{mode}"\nkey = "{key}"\non = "toggle"\nfrom = ["locomotion"]\n'
            )
            self.models[mode] = paths[f"{mode}/actor.onnx"]
            self.contracts[mode] = paths[f"{mode}/layout.json"].read_text()
            if mode == "dance_waist":
                self.trajectories[mode] = paths[f"{mode}/waist.motion.trajectory.json"].read_text()

    def file(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Jumper bundle reference missing or outside bundle: {relative}")
        expected = self.manifest.get("files", {}).get(relative)
        if (
            not expected
            or path.stat().st_size != expected["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Jumper bundle digest mismatch: {relative}")
        return path

    def new_fsm(self, *, operators=False):
        gait = next(
            (t.get("params", {}) for t in self.contract["observation"]["terms"] if t["name"] == "gait_phase"), {}
        )
        return self.extension.Fsm(
            self.config,
            self.contracts,
            {
                "joint_names": self.wire,
                "joint_pos_lo": [self.contract["joint_limits"][n][0] for n in self.wire],
                "joint_pos_hi": [self.contract["joint_limits"][n][1] for n in self.wire],
                "output_rate_hz": self.hz,
                "gait_period": gait.get("period"),
                "gait_gate_threshold": gait.get("command_threshold"),
            },
            trajectories=self.trajectories,
            operators=operators,
        )

    def use_operators(self, enabled: bool):
        """Change host input mode only at a settled locomotion handoff.

        The upstream native API fixes this flag at construction. Keeping the
        official contracts intact avoids remapping pad axes or overriding hooks.
        """
        if enabled != self.operators:
            if self.fsm.mode != "locomotion":
                raise RuntimeError("Jumper input handoff requires locomotion")
            self.fsm = self.new_fsm(operators=enabled)
            self.key_events.clear()
            self.pressed_keys.clear()
            self.arm_key = None
            self.trigger_axes.clear()
            self.send_trigger_frame = False
            self.operators = enabled

    def tap_mode(self, mode: str):
        key = self.mode_keys.get(mode)
        if key is None:
            raise ValueError(f"Jumper controller has no mode key: {mode}")
        # Both edges must be observed on different controller ticks.
        modifier = self.modifiers.get(mode)
        if modifier:
            self.key_events.append((modifier, True))
        self.key_events.extend(((key, True), (key, False)))
        if modifier:
            self.key_events.append((modifier, False))

    def claw(self, side: str, closed: bool):
        key = self.claw_keys.get(side)
        if key is None:
            raise ValueError(f"Jumper controller has no claw key: {side}")
        if self.trigger_axes:
            self.trigger_axes.clear()
            self.send_trigger_frame = True
        self.key_events.append((key, closed))

    def claw_aperture(self, side: str, opening: float) -> None:
        # Release the keyboard's previous squeeze before driving the native LT/RT input.
        self.claw(side, False)
        self.trigger_axes = {self.claw_triggers[side]: 1.0 - opening} if opening < 1 else {}
        self.send_trigger_frame = True

    def cancel_keys(self) -> None:
        """Discard pending input and release observed motion/modifier keys, preserving the claw."""
        self.key_events.clear()
        for key in sorted(self.pressed_keys.difference(self.claw_keys.values())):
            self.key_events.append((key, False))
        self.arm_key = None

    def arm_pose(self, side: str, pose: str) -> None:
        if self.arm_key:
            self.key_events.append((self.arm_key, False))
        self.arm_key = None if pose == "rest" else "key_left_" + self.claw_controls[side]["task"]["arm_" + pose][0]
        if self.arm_key:
            self.key_events.append((self.arm_key, True))

    def step_key(self, side: str, direction: str) -> str:
        axes = self.claw_controls[side]["axes"]
        for axis, positive, negative in (
            ("lin_vel_x", "forward", "backward"),
            ("lin_vel_y", "left", "right"),
            ("ang_vel_z", "turn_left", "turn_right"),
            ("pitch", "pitch_down", "pitch_up"),
            ("roll", "roll_right", "roll_left"),
            ("twist", "twist_left", "twist_right"),
        ):
            if direction in (positive, negative):
                return str(axes[axis]["+" if direction == positive else "-"][0])
        raise ValueError("Unknown five-foot control")

    def infer(self, mode: str, observation):
        path = self.models[mode]
        session = self.sessions[path]
        action = session.run(None, {session.get_inputs()[0].name: np.asarray([observation], dtype=np.float32)})[0][0]
        if not np.isfinite(action).all():
            raise RuntimeError("Jumper policy produced non-finite actions")
        return action

    def verify_reference(self) -> dict:
        path = self.file(self.manifest["reference"])
        report = self.new_fsm().check_reference(path.read_text())
        worst = 0.0
        for frame in json.loads(path.read_text())["frames"]:
            if "infer" in frame:
                worst = max(worst, float(np.max(np.abs(self.infer(frame["infer"], frame["obs"]) - frame["act"]))))
        if any(
            (report.observation > 1e-6, report.target > 1e-6, report.divergent, report.mode_mismatches, worst > 1e-5)
        ):
            raise ValueError("Jumper controller/ONNX reference replay failed")
        return {
            "frames": report.frames,
            "inferences": report.inferences,
            "observation_error": report.observation,
            "target_error": report.target,
            "onnx_error": worst,
            "scope": "recorded locomotion frames only; not hardware validation",
        }

    def tick(self, q, qd, tau, quat, gyro, command, now_us: int):
        self.fsm.set_state(q.tolist(), qd.tolist(), tau.tolist(), quat.tolist(), gyro.tolist(), now_us)
        if self.posture and not self.operators and self.fsm.mode == "locomotion":
            axes = {"lin_vel_x": command[0], "lin_vel_y": command[1], "ang_vel_z": command[2], **self.posture}
            self.fsm.set_command_axes(list(axes), list(axes.values()), now_us)
        else:
            self.fsm.set_command(*command, now_us)
        if self.send_trigger_frame or self.trigger_axes:
            self.fsm.set_pad_frame([], self.trigger_axes, connected=bool(self.trigger_axes), now_us=now_us)
            self.send_trigger_frame = False
        if self.key_events:
            key, pressed = self.key_events.popleft()
            self.fsm.set_key(key, pressed, now_us)
            if pressed:
                self.pressed_keys.add(key)
            else:
                self.pressed_keys.discard(key)
        mode = self.fsm.tick(now_us)
        if mode is not None:
            self.fsm.resume(self.infer(mode, self.fsm.observation()).tolist())
        return np.asarray(self.fsm.positions()), np.asarray(self.fsm.kp()), np.asarray(self.fsm.kd())
