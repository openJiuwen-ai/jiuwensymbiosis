# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Calibration workflow preflight: contract + reproducibility checks, never safety.

All preflight logic lives here; the CLI keeps no ``_preflight_*`` parallel
implementation. ``PreflightError`` is the public failure type; import it from
``jiuwensymbiosis.calibration.workflows.preflight`` (the frozen
``jiuwensymbiosis.calibration`` facade does
not re-export internal preflight helpers).

Two execution phases (order is fixed, §6.2):

1. **Static preflight** (before connecting): archive kind/schema/shape, adapter
   ownership (``spec.package``), mount, trajectory space, required motion
   protocols. ``--dry-run`` stops after dense-trajectory expansion.
2. **Dynamic preflight** (after connect, before first motion): live joint
   metadata (order/unit/periodic) and one ``capture_calibration_frame()`` to
   verify camera + intrinsics + distortion snapshot; the preflight frame is
   discarded (does not form a Station).

Preflight can only reject contract/data inconsistencies. It MUST NOT base
rejection on workspace, soft limits or collision assumptions, and MUST NOT
modify or refuse the operator's trajectory on safety grounds — the Driver/
controller is the sole hardware-safety authority and may refuse a segment
(abort, no auto-home, no direction retry).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from jiuwensymbiosis.calibration._mount import MOUNT_VALUES, Mount, canonical_mount
from jiuwensymbiosis.calibration.domain.models import require_se3_mm
from jiuwensymbiosis.calibration.domain.ports import (
    CalibrationPoseSource,
    CartesianCalibrationMotion,
    JointCalibrationMotion,
    JointState,
    ManualGuidance,
)
from jiuwensymbiosis.calibration.domain.trajectory import TrajectorySpace
from jiuwensymbiosis.utils.logging import get_logger

logger = get_logger("jiuwensymbiosis.calibration.preflight")


class PreflightError(RuntimeError):
    """Raised when a contract or data-reproducibility check fails.

    Never raised for workspace/soft-limit/collision reasons — those are the
    Driver's authority, not the calibration layer's.
    """


def resolve_teaching_mode(env: Any) -> Literal["manual_guidance", "external_snapshot"]:
    """``--collect-poses`` without --import-poses: ManualGuidance or snapshot.

    A missing ManualGuidance is NOT an error — the operator may use an external
    teaching pendant and snapshot. Returns the mode label for the CLI to print.
    """
    if isinstance(env, ManualGuidance):
        return "manual_guidance"
    logger.warning(
        "env %s does not implement ManualGuidance; --collect-poses will fall "
        "back to external-teach snapshot mode (move the arm by an external "
        "teaching pendant, then snapshot).",
        type(env).__name__,
    )
    return "external_snapshot"


def validate_motion_capabilities(env: Any, space: TrajectorySpace) -> None:
    """``--auto`` needs CalibrationPoseSource + the motion protocol for ``space``."""
    if not isinstance(env, CalibrationPoseSource):
        raise PreflightError(
            f"{type(env).__name__} lacks CalibrationPoseSource (get_flange_transform_mm); "
            "cannot read flange SE(3) for FK. Implement the protocol to support calibration."
        )
    if space == "joint":
        if not isinstance(env, JointCalibrationMotion):
            raise PreflightError(
                f"{type(env).__name__} lacks JointCalibrationMotion (get_joint_state / "
                "move_joint_vector); required for trajectory.space=joint. Implement the "
                "protocol or switch trajectory.space to cartesian."
            )
    elif space == "cartesian":
        if not isinstance(env, CartesianCalibrationMotion):
            raise PreflightError(
                f"{type(env).__name__} lacks CartesianCalibrationMotion "
                "(move_to_flange_transform_mm); required for trajectory.space=cartesian."
            )
    else:
        raise PreflightError(f"trajectory.space must be 'joint' or 'cartesian', got {space!r}")


def validate_mount(value: str, *, source: str = "camera_mount") -> Mount:
    """Validate a mount string and return its canonical value.

    ``camera_mount`` has a closed vocabulary — ``eye_in_hand`` or
    ``eye_to_hand``. Any other value (including a stray path or typo) is
    rejected at the boundary so it cannot pollute a published frame.
    """
    try:
        return canonical_mount(value)
    except ValueError as exc:
        raise PreflightError(
            f"{source}={value!r} is not a valid camera mount (expected one of {MOUNT_VALUES}); "
            "the calibration output frame would be ambiguous."
        ) from exc


def validate_eye_to_hand_mount(value: str, *, source: str = "camera_mount") -> Literal["eye_to_hand"]:
    """Validate the mount for the eye-to-hand workflow entry points.

    The calibration solver exposes both frame-explicit modes, but the collection,
    execution, replay and publication workflow in this module is deliberately
    the eye-to-hand use case.  Reject eye-in-hand here rather than allowing an
    archive to be solved and published under the wrong ``T_base_cam`` meaning;
    callers should use ``scripts/calibrate/calibrate_hand_eye.py`` for that
    separate workflow.
    """
    mount = validate_mount(value, source=source)
    if mount != "eye_to_hand":
        raise PreflightError(
            f"{source}={mount!r} is not supported by the eye-to-hand workflow; "
            "use scripts/calibrate/calibrate_hand_eye.py for eye-in-hand calibration."
        )
    return "eye_to_hand"


def validate_mount_consistency(expected: str, actual: str) -> None:
    """Assert two mount values agree (used to cross-check caller vs env/archive).

    Both values are validated first, then compared for equality. Used so a
    stale CLI/config value can never silently override the env or archive
    authority — it can only be rejected.
    """
    canon_expected = validate_mount(expected, source="expected mount")
    canon_actual = validate_mount(actual, source="actual mount")
    if canon_expected != canon_actual:
        raise PreflightError(
            f"expected mount={expected!r} != actual mount={actual!r}; "
            "the calibration output frame would not match the runtime frame."
        )


def validate_archive_ownership(
    archive_meta: dict[str, Any],
    expected_adapter: str | Any,
    mount: str,
    space: TrajectorySpace,
    archive_space: str,
) -> None:
    """Static check: archive must belong to THIS adapter + match mount + space.

    The workflow accepts a primitive adapter package string.  Objects exposing
    ``.package`` remain accepted temporarily for legacy callers, without
    importing the framework-specific adapter-spec type into this module.
    """
    expected_package = (
        expected_adapter if isinstance(expected_adapter, str) else str(getattr(expected_adapter, "package", ""))
    )
    if not expected_package:
        raise PreflightError("expected adapter package is empty; archive ownership cannot be verified.")
    archive_mount = str(archive_meta.get("mount") or "")
    if archive_mount and archive_mount != mount:
        raise PreflightError(
            f"archive mount={archive_mount!r} != config mount={mount!r}; "
            "this waypoint archive was collected for a different camera mount."
        )
    archive_pkg = archive_meta.get("adapter_module")
    if archive_pkg and archive_pkg != "unknown" and archive_pkg != expected_package:
        raise PreflightError(
            f"archive adapter={archive_pkg!r} != this robot={expected_package!r}; "
            "this waypoint archive was collected on a different adapter."
        )
    if space != archive_space:
        raise PreflightError(
            f"trajectory space={space!r} but archive space={archive_space!r}; "
            "the archive's space does not match the required motion protocol."
        )


def validate_joint_metadata(
    archive_joint_order: tuple[str, ...],
    archive_joint_unit: str,
    archive_joint_periodic: tuple[bool, ...],
    live_joint_state: JointState,
) -> None:
    """Post-connect: archive joint order/unit/periodic == live env.

    Compares order (exact tuple equality — columns are positional), unit and
    periodic. A mismatch would make interpolation emit values in the wrong
    unit or wrap direction, which the Env would then interpret per its own
    vendor convention — silent wrong-axis / wrong-unit motion. ``limits``
    coverage is NOT checked — calibration trajectory interpolation does not
    read soft limits (§7.2).
    """
    order = tuple(archive_joint_order)
    live_order = tuple(live_joint_state.order)
    if order != live_order:
        raise PreflightError(
            f"archive joint order {order} != live order {live_order}; "
            "positional mismatch would send the wrong value to each joint."
        )
    archive_unit = str(archive_joint_unit)
    live_unit = str(live_joint_state.unit)
    if archive_unit != live_unit:
        raise PreflightError(
            f"archive joint unit {archive_unit!r} != live unit {live_unit!r}; "
            "interpolation would emit values the Env interprets in a different unit."
        )
    archive_periodic = tuple(bool(x) for x in archive_joint_periodic)
    live_periodic = tuple(bool(x) for x in live_joint_state.periodic)
    if len(archive_periodic) != len(live_periodic):
        raise PreflightError(
            f"archive periodic length {len(archive_periodic)} != live length {len(live_periodic)}; "
            "a periodic-joint set changed between collection and replay."
        )
    if archive_periodic != live_periodic:
        diff = [i for i, (a, b) in enumerate(zip(archive_periodic, live_periodic, strict=True)) if a != b]
        raise PreflightError(
            f"archive periodic {archive_periodic} != live periodic {live_periodic} "
            f"(joints {diff}); the wrap direction for a periodic joint differs, "
            "so the same stored coordinates would take a different physical arc."
        )


def validate_cartesian_trajectory(tfs: np.ndarray) -> None:
    """Pure SE(3) data validation for cartesian waypoints — no workspace/IK.

    Checks: archive space, array shape ``(N, 4, 4)`` with ``N >= 2``, each matrix
    finite + homogeneous + SO(3) (via :func:`require_se3_mm`). Does NOT read
    ``env.workspace_bounds`` or ``z_min_safe``; reachability is the Driver's call.
    """
    arr = np.asarray(tfs, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise PreflightError(f"cartesian_tfs must be (N,4,4), got {arr.shape}")
    if arr.shape[0] < 2:
        raise PreflightError(f"cartesian_tfs needs N>=2, got {arr.shape[0]}")
    for i, tf in enumerate(arr):
        try:
            require_se3_mm(tf, field=f"cartesian_tfs[{i}]")
        except ValueError as exc:
            raise PreflightError(str(exc)) from exc


__all__ = [
    "PreflightError",
    "resolve_teaching_mode",
    "validate_archive_ownership",
    "validate_cartesian_trajectory",
    "validate_eye_to_hand_mount",
    "validate_joint_metadata",
    "validate_mount",
    "validate_mount_consistency",
    "validate_motion_capabilities",
]
