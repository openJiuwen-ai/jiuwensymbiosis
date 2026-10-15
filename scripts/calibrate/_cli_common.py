# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared CLI plumbing for the calibration console scripts.

This is the single home for the generic application-flow pieces the calibration
CLIs need: profile loading, adapter/session loading, board construction, workflow
dependency assembly, logging and the exit-code contract.  The per-mount console
scripts (``hand_eye_calib.py`` and its aliases) only own argument parsing and
user-readable output; they never import solver internals or hand-assemble an
artifact schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jiuwensymbiosis.calibration import CalibrationRunOptions
from jiuwensymbiosis.calibration.domain.quality import ObservabilityThresholds, TargetConsistencyThresholds
from jiuwensymbiosis.calibration.integration.integration import (
    CalibrationAdapterSpec,
    SolvedCalibration,
    load_adapter_spec,
    validate_adapter_reload,
)
from jiuwensymbiosis.calibration.workflows.profile import load_profile
from jiuwensymbiosis.calibration.workflows.workflows import WorkflowDependencies
from jiuwensymbiosis.utils.logging import configure_logging as _configure_framework_logging
from scripts.calibrate.handeye_board import BoardSpec

if TYPE_CHECKING:
    from jiuwensymbiosis.agent.session import RobotSession

logger = logging.getLogger("jiuwensymbiosis.calibration.cli_common")


# =============================================================================
# Profile (the calibration: YAML block — script metadata only)
# =============================================================================
def observability_thresholds_from(profile: dict[str, Any], args: Any) -> ObservabilityThresholds:
    """Build thresholds: profile defaults, CLI overrides win (same name)."""
    defaults = ObservabilityThresholds()
    fields = {
        "min_relative_rotation_deg": defaults.min_relative_rotation_deg,
        "min_axis_separation_deg": defaults.min_axis_separation_deg,
        "min_max_rotation_deg": defaults.min_max_rotation_deg,
        "min_translation_baseline_mm": defaults.min_translation_baseline_mm,
        "min_camera_translation_baseline_mm": defaults.min_camera_translation_baseline_mm,
        "duplicate_rotation_deg": defaults.duplicate_rotation_deg,
        "duplicate_translation_mm": defaults.duplicate_translation_mm,
    }
    obs_profile = profile.get("observability") or {}
    for k, default in fields.items():
        v = getattr(args, k, None)
        if v is None:
            v = obs_profile.get(k, default)
        fields[k] = float(v)
    return ObservabilityThresholds(**fields)


def solution_quality_thresholds_from(profile: dict[str, Any]) -> TargetConsistencyThresholds:
    """Build the target-consistency thresholds from the solution_quality block."""
    defaults = TargetConsistencyThresholds()
    sq_profile = profile.get("solution_quality") or {}
    return TargetConsistencyThresholds(
        max_translation_std_mm=float(
            sq_profile.get("max_flange_target_translation_std_mm", defaults.max_translation_std_mm)
        ),
        max_rotation_spread_deg=float(
            sq_profile.get("max_flange_target_rotation_spread_deg", defaults.max_rotation_spread_deg)
        ),
    )


# =============================================================================
# Adapter spec loading (no sidecar, no agent) — calibration-owned discovery
# =============================================================================
def load_adapter_spec_and_session(yaml_path: str | Path) -> tuple[CalibrationAdapterSpec, RobotSession]:
    """Read the adapter module + build the explicit CalibrationAdapterSpec.

    Returns ``(spec, session_unconnected)``. The session is built through the
    adapter's ``session_factory.from_yaml`` (the single loader — no separate
    ``load_config``), with sidecars OFF (no detector). The CLI owns the session
    lifecycle via ``with session:``; the calibration-owned adapter wrapper is
    built over the connected session Env. The camera mount is read once from
    ``device.camera_mount`` (single authority).
    """
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    adapter_module = str(data.get("calibration", {}).get("adapter_module") or "")
    if not adapter_module:
        raise ValueError(
            f"{yaml_path}: missing calibration.adapter_module (the adapter's "
            "Python module, e.g. jiuwensymbiosis.adapters.so101)."
        )
    spec = load_adapter_spec(adapter_module)
    session = spec.session_factory.from_yaml(yaml_path, include_sidecars=False)
    return spec, session


def build_board(args: Any) -> BoardSpec:
    """Build a BoardSpec from the CLI board arguments."""
    return BoardSpec(
        kind=args.board,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_size_mm=args.square_size_mm,
        marker_size_mm=args.marker_size_mm if args.board == "charuco" else None,
    )


def board_spec_meta(args: Any) -> dict[str, Any]:
    """Serialise the board spec for archive metadata."""
    board = build_board(args)
    spec: dict[str, Any] = {
        "kind": str(getattr(board, "kind", args.board)),
        "squares_x": int(getattr(board, "squares_x", args.squares_x)),
        "squares_y": int(getattr(board, "squares_y", args.squares_y)),
        "square_size_mm": float(getattr(board, "square_size_mm", args.square_size_mm)),
    }
    marker = getattr(board, "marker_size_mm", None)
    if marker is not None:
        spec["marker_size_mm"] = float(marker)
    return spec


def workflow_dependencies(
    spec: CalibrationAdapterSpec,
    args: Any,
    *,
    live: bool,
) -> WorkflowDependencies:
    """Build workflow dependencies and, for live capture, a board detector."""
    profile = load_profile(Path(args.config) if args.config else None)
    obs_th = observability_thresholds_from(profile, args)
    sq_th = solution_quality_thresholds_from(profile)
    detect_fn = None
    if live:
        board = build_board(args)
        min_corners = int(getattr(args, "min_corners", 6))

        def _detect(frame):
            from handeye_board import detect_board

            return detect_board(frame.rgb, board, frame.intrinsics, frame.distortion, min_corners=min_corners)

        detect_fn = _detect

    def _reload_validator(temporary_json, tf_base_cam, intrinsics, mount, stations, *, t_flange_target=None):
        validate_adapter_reload(
            spec,
            temporary_json,
            SolvedCalibration(tf_base_cam, intrinsics, mount, t_flange_target),
            stations,
        )

    return WorkflowDependencies(
        options=CalibrationRunOptions(
            config=getattr(args, "config", None),
            out=getattr(args, "out", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            n_stations=int(getattr(args, "n_stations", 20)),
            max_joint_step_deg=float(getattr(args, "max_joint_step_deg", 5.0)),
            max_cartesian_step_mm=float(getattr(args, "max_cartesian_step_mm", 10.0)),
            max_cartesian_step_deg=float(getattr(args, "max_cartesian_step_deg", 5.0)),
            method=str(getattr(args, "method", "PARK")),
            cross_check=bool(getattr(args, "cross_check", False)),
        ),
        adapter_package=spec.package,
        reload_validator=_reload_validator,
        observability_thresholds=obs_th,
        target_consistency_thresholds=sq_th,
        detect_fn=detect_fn,
    )


def configure_logging(debug: bool = False) -> None:
    """Configure the calibration CLI through the framework logging entry point.

    The unified CLI's logger is intentionally left propagating to the root so
    the framework's single formatted stream handler can render it.  Repeated
    calls only adjust levels; ``jiuwensymbiosis.utils.logging`` owns handler
    creation and de-duplication.
    """
    level = "DEBUG" if debug else "INFO"
    _configure_framework_logging(level=level)


def resolve_mount_guard(device, expected: str, other_command: str) -> str:
    """Resolve the device mount and enforce an alias entry point's mount.

    Used by the legacy console-script aliases so they share the unified
    application flow but refuse to run the wrong mount. Returns the validated
    mount; raises ``PreflightError`` on a mismatch.
    """
    from jiuwensymbiosis.calibration.workflows.preflight import PreflightError, validate_mount

    mount = validate_mount(str(device.camera_mount), source="device.camera_mount")
    if mount != expected:
        raise PreflightError(
            f"this entry point is for {expected} only; camera_mount={mount!r}. Use {other_command} for the other mount."
        )
    return mount


__all__ = [
    "board_spec_meta",
    "build_board",
    "configure_logging",
    "load_adapter_spec_and_session",
    "observability_thresholds_from",
    "resolve_mount_guard",
    "solution_quality_thresholds_from",
    "workflow_dependencies",
]
