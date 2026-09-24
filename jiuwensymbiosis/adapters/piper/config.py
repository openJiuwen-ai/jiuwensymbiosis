# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper adapter config."""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from jiuwensymbiosis.adapters._common.config import load_yaml_config
from jiuwensymbiosis.perception.config import DetectorConfig, parse_detector_config
from jiuwensymbiosis.perception.config import DetectorServerConfig as DetectorServerConfig

logger = logging.getLogger(__name__)


@dataclass
class PiperConfig:
    path_or_id_fields: ClassVar[tuple[str, ...]] = ("gdino_model_id", "sam2_model_id")
    path_fields: ClassVar[tuple[str, ...]] = ("calib_path", "module_path", "tts_module_path")

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

    # --- joint soft limits (Piper uses degree; unit must match move_joint).
    joint_limits: dict[str, tuple[float, float]] | None = None

    # --- camera (optional; None disables)
    camera_serial: str | None = None
    camera_resolution: tuple[int, int] = (640, 480)
    camera_fps: int = 30
    # Camera topology, and the single authority the calibration subsystem reads
    # to pick its output frame. Piper's camera is wrist-mounted, so calibration
    # solves ``T_flange_cam``.
    camera_mount: Literal["eye_in_hand"] = "eye_in_hand"

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
    #     the gripper tip goes to ``target_top + place_z_offset_mm`` so the held
    #     object's bottom rests on the target's top. So this = the held object's
    #     tip-to-bottom distance (= object_height - grasp depth).
    #     ``get_grasp_info_simple`` returns a ready ``place_z = detected_top +
    #     place_z_offset_mm`` so the agent descends to it directly and never
    #     computes the stack height. (This is the knob for placement height.)
    #     Legacy alias ``chip_thickness_mm`` is still accepted by ``from_dict``.
    place_z_offset_mm: float = 75.0

    # --- task knobs
    detector: DetectorConfig = field(default_factory=DetectorConfig)
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
        detector_cfg = parse_detector_config(data)

        if isinstance(ll, dict) and ll:
            kw = {k: v for k, v in ll.items() if not k.startswith("_")}
        else:
            kw = dict(data)
        if "camera_resolution" in kw:
            kw["camera_resolution"] = tuple(kw["camera_resolution"])
        if "camera_mount" in kw and kw["camera_mount"] != "eye_in_hand":
            raise ValueError(
                f"PiperConfig: camera_mount must be 'eye_in_hand' (the camera is wrist-mounted), "
                f"got {kw['camera_mount']!r}."
            )
        if "joint_limits" in kw:
            raw_limits = kw["joint_limits"]
            if not isinstance(raw_limits, dict):
                kw["joint_limits"] = None
            else:
                normalised: dict[str, tuple[float, float]] = {}
                for k, v in raw_limits.items():
                    if not isinstance(v, (list, tuple)) or len(v) != 2:
                        continue
                    try:
                        normalised[str(k)] = (float(v[0]), float(v[1]))
                    except (TypeError, ValueError):
                        continue
                kw["joint_limits"] = normalised if normalised else None

        if "chip_thickness_mm" in kw:
            if "place_z_offset_mm" in kw:
                raise ValueError(
                    "PiperConfig: place_z_offset_mm and deprecated chip_thickness_mm cannot both be "
                    "configured; specify only place_z_offset_mm."
                )
            logger.warning(
                "PiperConfig: chip_thickness_mm is deprecated; migrate the configuration to place_z_offset_mm."
            )
            kw["place_z_offset_mm"] = kw.pop("chip_thickness_mm")

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
        return load_yaml_config(cls, path)


def _extract_detector_from_api_servers(api_servers: list[Any]) -> DetectorConfig:
    """Compatibility helper; canonical parsing lives in perception.config."""
    return parse_detector_config({"api_servers": api_servers})
