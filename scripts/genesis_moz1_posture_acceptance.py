"""Record third-person MOZ1 motion with measured articulated-posture evidence.

Run with the Genesis environment from the repository root. This drives the same
RuntimeScene/Moz1Base as the workbench, on a flat floor with self collision on.
Outputs are simulation evidence only; the PD parameters are not hardware limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

DT = 1 / 120
FPS = 30
# Motion pulses are refreshed below; every pulse retains the normal 0.25 s expiry.
PHASES = [
    ("Hold", (0, 0, 0), 2),
    ("Forward", (0.2, 0, 0), 2),
    ("Brake", (0, 0, 0), 1),
    ("Backward", (-0.2, 0, 0), 2),
    ("Brake", (0, 0, 0), 1),
    ("Left", (0, 0.2, 0), 2),
    ("Brake", (0, 0, 0), 1),
    ("Right", (0, -0.2, 0), 2),
    ("Brake", (0, 0, 0), 1),
    ("Turn CCW", (0, 0, 0.4), 2),
    ("Brake", (0, 0, 0), 1),
    ("Turn CW", (0, 0, -0.4), 2),
    ("Brake", (0, 0, 0), 1),
    ("Translate + turn", (0.16, 0.10, 0.3), 2),
    ("Brake", (0, 0, 0), 1),
    ("Max forward", (0.3, 0, 0), 2),
    ("Brake", (0, 0, 0), 1),
    ("Max turn", (0, 0, 0.6), 2),
    ("Final hold", (0, 0, 0), 2),
]


def observe(robot: Any) -> dict:
    """Read every upper DOF and root from physics, at the physics sampling rate."""
    entity, hold = robot.entity, robot.upper_body
    q = entity.get_dofs_position(hold.dofs)[0].cpu().numpy()
    dq = entity.get_dofs_velocity(hold.dofs)[0].cpu().numpy()
    quat = entity.get_quat()[0].cpu().numpy()
    rotation = Rotation.from_quat(quat, scalar_first=True)
    contacts = entity.get_contacts(with_entity=entity)
    valid = contacts["valid_mask"][0]
    forces = contacts["force_a"][0][valid].cpu().numpy()
    pairs = zip(contacts["link_a"][0][valid].cpu().tolist(), contacts["link_b"][0][valid].cpu().tolist(), strict=True)
    return {
        "q_rad": q.tolist(),
        "dq_rad_s": dq.tolist(),
        "error_rad": (q - hold.target).tolist(),
        "motor_effort_nm": entity.get_dofs_control_force(hold.motor_dofs)[0].cpu().tolist(),
        "base_position_m": entity.get_pos()[0].cpu().tolist(),
        "base_quaternion_wxyz": quat.tolist(),
        "base_velocity_m_s": entity.get_vel()[0].cpu().tolist(),
        "base_angular_velocity_rad_s": entity.get_ang()[0].cpu().tolist(),
        "base_tilt_deg": math.degrees(math.acos(float(np.clip(rotation.apply([0, 0, 1])[2], -1, 1)))),
        "base_yaw_rad": float(rotation.as_euler("xyz")[2]),
        "self_contact_count": int(len(forces)),
        "self_contact_max_force_n": float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0,
        "self_contact_pairs": sorted(
            {tuple(sorted((entity.solver.links[a].name, entity.solver.links[b].name))) for a, b in pairs}
        ),
    }


def annotate(rgb: np.ndarray, *, phase: str, sample: dict, peak: float, trajectory: list) -> np.ndarray:
    """Add readable, time-aligned measurements to an actual third-person frame."""
    import cv2

    frame: np.ndarray = np.array(rgb, copy=True)
    cv2.rectangle(frame, (0, 0), (1280, 112), (20, 27, 36), -1)
    lines = [
        f"MOZ1 | Self-collision ON | {phase} | {sample['time_s']:05.2f} s",
        f"36 movable upper joints | PD hold + 14 passive mimic followers | Error now: {np.degrees(np.max(np.abs(sample['error_rad']))):.3f} deg",
        f"Peak joint error: {peak:.3f} deg | Base tilt: {sample['base_tilt_deg']:.3f} deg | Physics 120 Hz / video 30 fps",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (24, 29 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (230, 239, 246), 1, cv2.LINE_AA)
    # A fixed world-space path inset makes translation apparent on the plain floor.
    cv2.rectangle(frame, (1030, 450), (1264, 704), (20, 27, 36), -1)
    cv2.putText(frame, "Measured base XY", (1041, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 230, 240), 1, cv2.LINE_AA)
    for offset in range(-100, 101, 50):
        cv2.line(frame, (1147 + offset, 486), (1147 + offset, 686), (60, 66, 75), 1)
        cv2.line(frame, (1047, 586 + offset), (1247, 586 + offset), (60, 66, 75), 1)
    points = np.asarray([[1147 + 100 * p[0], 586 - 100 * p[1]] for p in trajectory], np.int32)
    if len(points) > 1:
        cv2.polylines(frame, [points], False, (65, 205, 235), 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(points[-1]), 4, (245, 190, 70), -1)
    cv2.putText(frame, "Grid: 0.5 m", (1041, 699), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 220, 230), 1, cv2.LINE_AA)
    return frame


def phase_summary(name: str, twist: tuple, before: dict, samples: list[dict]) -> dict:
    errors = np.abs([s["error_rad"] for s in samples])
    tail = samples[-60:]
    end = samples[-1]
    delta_yaw = end["base_yaw_rad"] - before["base_yaw_rad"]
    delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
    return {
        "name": name,
        "command_body_twist": twist,
        "duration_s": len(samples) * DT,
        "max_joint_error_deg": float(np.degrees(np.max(errors))),
        "per_joint_max_error_deg": np.degrees(np.max(errors, axis=0)).tolist(),
        "max_base_tilt_deg": max(s["base_tilt_deg"] for s in samples),
        "displacement_m": (np.array(end["base_position_m"]) - before["base_position_m"]).tolist(),
        "yaw_change_deg": math.degrees(delta_yaw),
        "tail_max_joint_speed_rad_s": float(np.max(np.abs([s["dq_rad_s"] for s in tail]))),
        "tail_max_base_speed_m_s": max(float(np.linalg.norm(s["base_velocity_m_s"])) for s in tail),
        "tail_max_base_angular_speed_rad_s": max(float(np.linalg.norm(s["base_angular_velocity_rad_s"])) for s in tail),
        "max_motor_effort_nm": np.max(np.abs([s["motor_effort_nm"] for s in samples]), axis=0).tolist(),
        "max_self_contact_count": max(s["self_contact_count"] for s in samples),
        "max_self_contact_force_n": max(s["self_contact_max_force_n"] for s in samples),
        "self_contact_pairs": sorted({tuple(pair) for s in samples for pair in s["self_contact_pairs"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/genesis-moz1-upper-body/acceptance"))
    parser.add_argument("--robot-id", default="library://spiritai/moz1/moz1_omni_gripper_double_narrow")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("report.json", "third-person.mp4", "telemetry.jsonl")):
        raise FileExistsError("Use a fresh output directory to retain previous acceptance evidence")

    import av
    import genesis as gs
    import gs_nyx.nyx_py_sdk as nps

    # Like the file worker, avoid initializing the core / openjiuwen import hooks.
    directory = Path(__file__).resolve().parents[1] / "jiuwensymbiosis/simulators/genesis"
    sys.path.insert(0, str(directory))
    try:
        from camera import PreviewSettings
        from scene_runtime import RuntimeScene
    finally:
        sys.path.pop(0)

    # Worker dependencies use local lazy imports throughout their lifetime.
    sys.path.insert(0, str(directory))
    runtime = container = None
    initialized = sdk_started = False
    report: dict[str, Any] = {"passed": False, "phases": [], "sample_rate_hz": 120, "video_fps": FPS}
    try:
        gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)
        initialized = True
        nps.startup()
        sdk_started = True
        settings = PreviewSettings(scene_id=args.robot_id, robot_mode="moz1_wheels", fps=FPS, spp=32)
        runtime = RuntimeScene(settings, out)
        robot = runtime.robot
        if not robot.collision_settings()["self_collision"] or not robot.collision_settings()["enabled"]:
            raise RuntimeError("Self-collision acceptance requires actual solver collision flags enabled")
        # Presentation-only exposure and framing; the physical world is retained.
        runtime.close_renderer()
        runtime.info["preview_exposure_ev"] = 1.25
        runtime.build_renderer(settings)
        pose = {"pos": [2.5, -3.4, 2.2], "lookat": [0.30, -0.30, 1.05], "up": [0, 0, 1]}
        runtime.update_camera(pose)
        report.update(
            genesis_version=version("genesis-world"),
            nyx_version=version("gs-nyx"),
            scene=runtime.info,
            observer_pose=pose,
            initial_robot=robot.snapshot(),
            source_sha256=hashlib.sha256(robot.source.read_bytes()).hexdigest(),
            physics_dt_s=DT,
            physics_substeps=4,
            limits={
                "joint_error_deg": 1.0,
                "base_tilt_deg": 2.0,
                "stopped_joint_speed_rad_s": 0.03,
                "stopped_base_speed_m_s": 0.008,
                "stopped_base_angular_speed_rad_s": 0.02,
            },
        )
        moving = [j for j in robot.entity.joints if j.n_dofs == 1]
        report["articulation"] = {
            "total_dofs": robot.entity.n_dofs,
            "one_dof_joints": len(moving),
            "upper_dofs": len(robot.upper_body.dofs),
            "upper_motor_dofs": len(robot.upper_body.motor_dofs),
        }
        if report["articulation"] != {"total_dofs": 86, "one_dof_joints": 80, "upper_dofs": 36, "upper_motor_dofs": 22}:
            raise RuntimeError("Unexpected MOZ1 articulation; do not record a locked or mismatched model")
        print("RuntimeScene ready; recording actual physics", flush=True)
        container = av.open(str(out / "third-person.mp4"), "w")
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        trajectory, total_steps, peak = [], 0, 0.0
        with (out / "telemetry.jsonl").open("w") as trace:
            for phase_index, (name, twist, duration) in enumerate(PHASES):
                samples, before = [], observe(robot)
                for step in range(round(duration / DT)):
                    if step % 15 == 0:
                        robot.command(*twist, duration_s=0.25)
                    runtime.step()
                    total_steps += 1
                    sample = observe(robot)
                    sample.update(
                        time_s=total_steps * DT, physics_step=runtime.steps, phase=name, phase_index=phase_index
                    )
                    trace.write(json.dumps(sample, allow_nan=False) + "\n")
                    samples.append(sample)
                    error = float(np.degrees(np.max(np.abs(sample["error_rad"]))))
                    peak = max(peak, error)
                    if (
                        error > report["limits"]["joint_error_deg"]
                        or sample["base_tilt_deg"] > report["limits"]["base_tilt_deg"]
                    ):
                        robot.stop()
                        raise RuntimeError(
                            f"Stability limit exceeded: {name}, error={error}, tilt={sample['base_tilt_deg']}"
                        )
                    if total_steps % 4 == 0:
                        trajectory.append(sample["base_position_m"][:2])
                        rgb = runtime.renderer.read_rgb("observer").cpu().numpy()
                        rgb = annotate(rgb, phase=name, sample=sample, peak=peak, trajectory=trajectory)
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        if step == 3:
                            from PIL import Image

                            Image.fromarray(rgb).save(out / f"phase-{phase_index:02}.jpg")
                summary = phase_summary(name, twist, before, samples)
                report["phases"].append(summary)
                print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}), flush=True)
                if any(twist):
                    if np.linalg.norm(twist[:2]) and np.linalg.norm(summary["displacement_m"][:2]) < 0.15:
                        raise RuntimeError(f"No measured translation in {name}")
                    if twist[2] and abs(summary["yaw_change_deg"]) < 15:
                        raise RuntimeError(f"No measured turn in {name}")
                else:
                    if (
                        summary["tail_max_joint_speed_rad_s"] > 0.03
                        or summary["tail_max_base_speed_m_s"] > 0.008
                        or summary["tail_max_base_angular_speed_rad_s"] > 0.02
                    ):
                        raise RuntimeError(f"Measured stop not confirmed in {name}")
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        report["final_robot"] = robot.snapshot()
        report["max_joint_error_deg"] = peak
        report["max_base_tilt_deg"] = max(p["max_base_tilt_deg"] for p in report["phases"])
        report["max_effort_fraction"] = float(
            np.max(np.max([p["max_motor_effort_nm"] for p in report["phases"]], axis=0) / robot.upper_body.effort)
        )
        report["recording_duration_s"] = total_steps * DT
        report["recording_frames"] = total_steps // 4
        report["reset"] = runtime.reset()
        for _ in range(120):
            runtime.step()
        report["after_reset"] = robot.snapshot()
        if report["after_reset"]["upper_body_state"]["max_abs_error_rad"] > math.radians(1):
            raise RuntimeError("Posture holding failed after scene reset")
        report["source_unchanged"] = hashlib.sha256(robot.source.read_bytes()).hexdigest() == report["source_sha256"]
        report["passed"] = report["source_unchanged"] and report["max_effort_fraction"] < 0.95
    finally:
        if container is not None:
            container.close()
        try:
            if runtime is not None:
                runtime.robot.stop()
                runtime.close()
        finally:
            try:
                if sdk_started:
                    nps.shutdown()
            finally:
                if initialized:
                    gs.destroy()
                sys.path.pop(0)
                (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if not report["passed"]:
        raise RuntimeError("Acceptance did not pass; inspect report.json")
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "max_joint_error_deg", "max_base_tilt_deg", "recording_duration_s")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
