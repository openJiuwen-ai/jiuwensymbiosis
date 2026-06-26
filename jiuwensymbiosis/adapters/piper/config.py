# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper adapter config."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorServerConfig:
    """How to reach (or spawn) the open-vocabulary detection server
    (GroundingDINO + SAM2), serving the ``/segment`` contract.
    """

    url: str = "http://127.0.0.1:8114"
    spawn: bool = True
    host: str = "127.0.0.1"
    port: int = 8114
    device: str = "cuda"
    startup_timeout_s: float = 300.0
    # --- GroundingDINO (text→box) + SAM2 (box→mask) knobs.
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    use_sam2: bool = True


@dataclass
class PiperConfig:
    # --- arm (single-arm; LEFT arm only)
    can_port: str = "can_left"
    # MOVE speed percentage (0-100) passed to MotionCtrl_2; start slow on real HW.
    move_speed: int = 50
    # Tool-tip offset from the flange along base -Z (mm).
    tool_offset_mm: float = 135.8

    # --- workspace constants
    calib_path: str | None = None
    home_lift_mm: float = 250.0
    z_safe_margin_mm: float = -10.0
    # 6-DoF home pose used only when no calib_path is given (mm/deg, FLANGE frame).
    home_pose_xyzrxryrz_mm_deg: list[float] = field(default_factory=lambda: [200.0, 0.0, 400.0, 0.0, 90.0, 0.0])
    # Calibration anchor object pose (used only when no calib_path).
    calib_object_xyzrxryrz_mm_deg: list[float] | None = None
    z_min_safe_mm: float = 50.0
    home_use_init_pose: bool = False

    # --- cartesian workspace box (mm). Clamped before
    #     every EndPoseCtrl so the firmware-chosen IK solution can't wander out
    #     of the front hemisphere. None on a side disables that bound.
    x_min_mm: float | None = 0.0
    x_max_mm: float | None = 700.0
    y_min_mm: float | None = -500.0
    y_max_mm: float | None = 500.0
    z_max_mm: float | None = 800.0

    # --- camera (optional; None disables)
    camera_serial: str | None = None
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30

    # --- gripper (parallel; piper supports width + force, 0.001mm / 0.001 N·m).
    gripper_open_mm: float = 70.0  # commanded width when "open"
    gripper_effort: int = 1000  # 0.001 N·m units (=1 N·m)
    gripper_settle_s: float = 0.8  # wait after a GripperCtrl before next motion

    # --- detection correction. The eye-in-hand back-projection (tf_flange_cam,
    #     initial value not yet re-calibrated on this robot)
    #     over-estimates the object Z by a roughly constant amount at the
    #     observation pose. Touch-calibration on 2026-06-08 found ~+57mm. This
    #     offset is ADDED to every detected base-frame Z (use a negative value
    #     to pull detections down). A proper hand-eye re-calibration would make
    #     this unnecessary.
    z_correction_mm: float = 0.0

    # --- grasp depth. Offset (mm) from the DETECTED TOP surface to the point the
    #     gripper should close at, so the parallel fingers straddle the object BODY
    #     (a top-down parallel grasp can't grab a flat top). NEGATIVE = below the
    #     top. ``get_grasp_info_simple`` returns a ready ``grasp_z`` =
    #     ``max(detected_top + this, z_min_safe)`` so the agent descends to it
    #     directly and never computes the grasp depth itself.
    grasp_z_offset_mm: float = -25.0

    # --- stacking place offset. When releasing a held object ON TOP of a target,
    #     the gripper tip goes to ``target_top + chip_thickness_mm`` so the held
    #     object's bottom rests on the target's top. So this = the held object's
    #     tip-to-bottom distance (= object_height - grasp depth). Same role as
    #     slot_pick's ``chip_thickness_mm``. ``get_grasp_info_simple`` returns a
    #     ready ``place_z = detected_top + chip_thickness_mm`` so the agent descends
    #     to it directly and never computes the stack height.
    chip_thickness_mm: float = 75.0

    # --- task knobs
    detector: DetectorServerConfig = field(default_factory=DetectorServerConfig)
    task_prompt: str | None = None
    name: str = "piper"

    # ----------------------------------------------------------------- loaders
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PiperConfig:
        """Accept either a flat dict OR an ``env.cfg.low_level.*`` dict.

        Honors ``env.cfg.prompt`` and ``api_servers``, so YAMLs port across with minimal churn.
        """
        ll = data.get("env", {}).get("cfg", {}).get("low_level", {}) if isinstance(data.get("env"), dict) else None
        prompt = data.get("env", {}).get("cfg", {}).get("prompt") if isinstance(data.get("env"), dict) else None
        api_servers = data.get("api_servers") or []
        detector_cfg = _extract_detector_from_api_servers(api_servers)

        if isinstance(ll, dict) and ll:
            kw = {k: v for k, v in ll.items() if not k.startswith("_")}
        else:
            kw = dict(data)
        if "camera_resolution" in kw:
            kw["camera_resolution"] = tuple(kw["camera_resolution"])

        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in kw.items() if k in valid}
        clean["detector"] = detector_cfg
        if "CAMERA_SERIAL" in os.environ:
            clean["camera_serial"] = os.environ["CAMERA_SERIAL"]
        if prompt is not None:
            clean["task_prompt"] = prompt
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PiperConfig:
        """Load config from a YAML file, resolving relative calib_path."""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)
        if cfg.calib_path and not Path(cfg.calib_path).is_absolute():
            candidate = (path.parent / cfg.calib_path).resolve()
            if candidate.exists():
                cfg.calib_path = str(candidate)
        return cfg


def _extract_detector_from_api_servers(api_servers: list[Any]) -> DetectorServerConfig:
    """If the YAML lists the detection server, copy its connection + model knobs.
    Recognizes the entry by ``_target_`` containing
    ``grounding_dino_sam2_server`` (or ``gdino``).
    """
    for s in api_servers or []:
        if not isinstance(s, dict):
            continue
        target = s.get("_target_", "").lower()
        if "grounding_dino" not in target and "gdino" not in target:
            continue
        host = s.get("host", "127.0.0.1")
        port = int(s.get("port", 8114))
        defaults = DetectorServerConfig()
        return DetectorServerConfig(
            url=f"http://{host}:{port}",
            spawn=True,
            host=host,
            port=port,
            device=s.get("device", "cuda"),
            gdino_model_id=os.environ.get("GDINO_MODEL_ID") or s.get("gdino_model_id", defaults.gdino_model_id),
            sam2_model_id=os.environ.get("SAM2_MODEL_ID") or s.get("sam2_model_id", defaults.sam2_model_id),
            box_threshold=float(s.get("box_threshold", defaults.box_threshold)),
            text_threshold=float(s.get("text_threshold", defaults.text_threshold)),
            use_sam2=bool(s.get("use_sam2", defaults.use_sam2)),
        )
    defaults = DetectorServerConfig()
    return DetectorServerConfig(
        gdino_model_id=os.environ.get("GDINO_MODEL_ID") or defaults.gdino_model_id,
        sam2_model_id=os.environ.get("SAM2_MODEL_ID") or defaults.sam2_model_id,
    )
