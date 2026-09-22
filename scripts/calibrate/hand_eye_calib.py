#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unified body-agnostic hand-eye calibration CLI (single application flow).

This is the single Python application flow for the ``collect-poses`` / ``auto`` /
``replay`` hand-eye calibration modes, shared by the eye-in-hand and eye-to-hand
console entry points.  The mount is resolved from ``device.camera_mount`` (the
single session authority), never re-declared by the user.

The orchestration lives in the calibration ``collect_waypoints``,
``execute_calibration`` and ``replay_calibration`` workflows.  This module only
owns argparse, logging, adapter composition and user-readable error output; it
does not touch ``low_level``, solver internals or the calibration JSON schema.

Exactly one mutually-exclusive mode is required:

  ``--collect-poses OUTPUT [--import-poses INPUT]``
      Without ``--import-poses``: connect + ``ManualGuidance`` teaching (or, on
      a body without ManualGuidance, snapshot an externally-moved arm).
      With ``--import-poses INPUT``: validate + normalise a self-describing
      waypoint archive into OUTPUT; no hardware is connected.

  ``--auto WAYPOINT_ARCHIVE [--config runtime.yaml]``
      Consume a normalised waypoint archive; drive the arm along it, settle,
      capture station frames, dump a station archive, solve + observability.
      LIVE ``--auto`` MUST have ``--config`` (full publication validation).

  ``--replay ARCHIVE [--config runtime.yaml]``
      Fully offline solve + report from a station archive. WITHOUT ``--config``
      it emits an UNLOADABLE REVIEW/candidate report; WITH ``--config`` it
      additionally validates the adapter + mount + runs an adapter reload
      smoke, and only then publishes a formal schema-2 calibration.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jiuwensymbiosis.calibration import (
    CalibrationRunOptions,
    ManualGuidanceRecoveryError,
    collect_waypoints,
    execute_calibration,
    replay_calibration,
)
from jiuwensymbiosis.calibration.workflows.collect import import_waypoints
from jiuwensymbiosis.calibration.workflows.preflight import PreflightError
from jiuwensymbiosis.utils.proxy import clear_proxy_env
from scripts.calibrate._cli_common import (
    configure_logging,
    load_adapter_spec_and_session,
    workflow_dependencies,
)

logger = logging.getLogger("hand_eye_calib")


@dataclass(frozen=True)
class _CalibrationResourceBinding:
    """The admitted resource identities for an already-built calibration session."""

    resources: tuple[str, ...]


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog or "hand_eye_calib.py",
        description="Body-agnostic hand-eye calibration (collect / auto / replay).",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect-poses", metavar="OUTPUT", help="Collect/normalise waypoints to OUTPUT archive.")
    mode.add_argument("--auto", metavar="WAYPOINT_ARCHIVE", help="Drive the arm along a waypoint archive + capture.")
    mode.add_argument("--replay", metavar="STATION_ARCHIVE", help="Offline re-solve from a station archive.")

    ap.add_argument(
        "--import-poses", metavar="INPUT", help="Normalise a self-describing waypoint archive (no hardware)."
    )
    ap.add_argument(
        "--config",
        metavar="runtime.yaml",
        help="Adapter runtime YAML (required for live --auto; optional for --replay).",
    )

    # Board spec (needed for live capture; not for pure --import-poses / --replay).
    ap.add_argument("--board", choices=["charuco", "chessboard"], default="charuco")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-size-mm", type=float, default=15.28)
    ap.add_argument("--marker-size-mm", type=float, default=11.0)
    ap.add_argument(
        "--min-corners",
        type=int,
        default=16,
        help="ChArUco 单帧最少角点数（少于此数的帧丢弃；下限 6，solvePnP 要求；越大质量越稳）",
    )

    ap.add_argument(
        "--n-stations",
        type=int,
        default=20,
        help="Target capture count for --auto (recommended 10-20); an upper bound on "
        "capture attempts, sampled at even dense-path indices incl. first/last.",
    )
    ap.add_argument("--max-joint-step-deg", type=float, default=5.0, help="Joint interpolation step cap (deg).")
    ap.add_argument("--max-cartesian-step-mm", type=float, default=10.0, help="Cartesian translation step cap (mm).")
    ap.add_argument("--max-cartesian-step-deg", type=float, default=5.0, help="Cartesian rotation step cap (deg).")
    ap.add_argument("--method", default="PARK", help="OpenCV calibrateHandEye method.")
    ap.add_argument("--cross-check", action="store_true", help="Cross-check all methods + report max disagreement.")
    ap.add_argument(
        "--out",
        metavar="PATH",
        help="Formal calibration output path (default: calibration.output or tmp/eye_to_hand_calib.json).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate trajectory + limits only; do not move or capture.")
    ap.add_argument(
        "--confirm-estop", action="store_true", help="Confirm the hardware E-stop is reachable before live motion."
    )

    # Observability threshold CLI overrides (same names as the profile keys).
    for name in (
        "min_relative_rotation_deg",
        "min_axis_separation_deg",
        "min_max_rotation_deg",
        "min_translation_baseline_mm",
        "min_camera_translation_baseline_mm",
        "duplicate_rotation_deg",
        "duplicate_translation_mm",
    ):
        ap.add_argument(f"--{name}", type=float, default=None, help=f"Observability threshold override ({name}).")

    ap.add_argument("--debug", action="store_true")
    return ap


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    from jiuwensymbiosis.calibration.domain.solver import MIN_STATIONS

    if args.auto and not args.config:
        ap.error("live --auto requires --config (full publication validation).")
    if args.import_poses is not None and not args.collect_poses:
        ap.error("--import-poses only valid with --collect-poses.")
    if args.n_stations < MIN_STATIONS:
        ap.error(f"--n-stations must be >= {MIN_STATIONS} (the solve minimum).")


# Exit codes: 0 = success (formal publication, or a dry-run whose contracts all
# validated), 1 = error, 2 = preflight failure, 3 = REVIEW/candidate (a report
# was written but it is NOT a loadable calibration).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PREFLIGHT = 2
EXIT_REVIEW = 3
_UNSAFE_CLEANUP_ERRORS = (ManualGuidanceRecoveryError,)


def _exit_code(outcome) -> int:
    """Map a workflow outcome to a process exit code.

    ``artifact_path is None`` alone does not mean failure: a successful
    ``--dry-run`` validates the trajectory and data contracts without producing
    an artifact. Conversely a candidate report IS an artifact but is not a
    publishable calibration, so it must not exit 0.
    """
    if outcome.candidate:
        return EXIT_REVIEW
    return EXIT_OK


def _mode_label(args: argparse.Namespace) -> str:
    if args.collect_poses:
        return "collect-poses" + ("+import" if args.import_poses else "")
    if args.auto:
        return "auto"
    return "replay"


def _prepare_calibration_binding(config_path: str | Path, spec: Any, session: Any) -> _CalibrationResourceBinding:
    """Derive admission resources from the config that built ``session``.

    The calibration spec's ``session_factory.from_yaml`` remains the only
    typed-config loader and keeps ownership of adapter-specific relative-path
    semantics. Resource identities are derived from that session's
    ``env.cfg``. The YAML is read only for the supplemental top-level
    ``physical_device_id`` metadata; no second config factory or session is
    created here.
    """
    import yaml

    from jiuwensymbiosis.adapters._common.resources import complete_resource_keys

    source = Path(config_path)
    metadata = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, dict):
        raise TypeError(f"{source}: calibration config root must be a mapping")

    env = getattr(session, "env", None)
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        raise TypeError("calibration session must expose its loaded typed config as session.env.cfg")
    resource_keys = getattr(spec.session_factory, "resource_keys", None)
    if not callable(resource_keys):
        raise TypeError("calibration session_factory must expose resource_keys(cfg)")

    resources = complete_resource_keys(
        resource_keys(cfg),
        cfg,
        physical_device_id=metadata.get("physical_device_id"),
        include_sidecars=False,
    )
    return _CalibrationResourceBinding(resources=resources)


def main(
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
    require_mount: str | None = None,
    resource_manager=None,
) -> int:
    clear_proxy_env()
    ap = _build_parser(prog)
    args = ap.parse_args(argv)
    _validate_args(ap, args)
    configure_logging(args.debug)

    label = _mode_label(args)
    logger.info("hand-eye calibration: mode=%s", label)

    try:
        if args.collect_poses and args.import_poses:
            # Offline waypoint normalisation — no adapter spec, no hardware.
            import_waypoints(args.import_poses, args.collect_poses, expected_mount=require_mount)
            return EXIT_OK
        if args.collect_poses:
            if not args.config:
                ap.error("--collect-poses (live) requires --config.")
            spec, session = load_adapter_spec_and_session(args.config)
            dependencies = workflow_dependencies(spec, args, live=False)
            from jiuwensymbiosis.runtime.admission import admitted_session

            binding = _prepare_calibration_binding(args.config, spec, session)
            with admitted_session(
                binding,
                session=session,
                resource_manager=resource_manager,
                operation="calibration-collect",
                unsafe_cleanup_errors=_UNSAFE_CLEANUP_ERRORS,
            ):
                device = spec.make_calibration_device(session.env)
                _guard(device, require_mount)
                collect_waypoints(
                    device,
                    args.collect_poses,
                    dependencies.options,
                    adapter_package=dependencies.adapter_package,
                    mount=device.camera_mount,
                )
            return EXIT_OK
        if args.auto:
            spec, session = load_adapter_spec_and_session(args.config)
            dependencies = workflow_dependencies(spec, args, live=True)
            if not args.dry_run and not args.confirm_estop:
                raise RuntimeError("--auto live motion: pass --confirm-estop to confirm the E-stop is reachable.")
            from jiuwensymbiosis.runtime.admission import admitted_session

            binding = _prepare_calibration_binding(args.config, spec, session)
            with admitted_session(
                binding,
                session=session,
                resource_manager=resource_manager,
                operation="calibration-auto",
                unsafe_cleanup_errors=_UNSAFE_CLEANUP_ERRORS,
            ):
                device = spec.make_calibration_device(session.env)
                _guard(device, require_mount)
                outcome = execute_calibration(
                    device,
                    args.auto,
                    mount=device.camera_mount,
                    dependencies=dependencies,
                )
            if outcome.candidate:
                logger.warning("run-auto produced a REVIEW/candidate (no formal calibration published).")
            return _exit_code(outcome)
        if args.replay:
            if args.config:
                spec, session = load_adapter_spec_and_session(args.config)
                dependencies = workflow_dependencies(spec, args, live=False)
                # Config mount is a consistency assertion only — the archive is the
                # publish authority. The device mount is read from the built (but
                # unconnected) env's config, so no connect is needed here.
                device = spec.make_calibration_device(session.env)
                _guard(device, require_mount)
                outcome = replay_calibration(
                    args.replay,
                    mount=device.camera_mount,
                    dependencies=dependencies,
                )
            else:
                # No config means REVIEW/candidate only. Archive metadata is the
                # sole mount authority in the replay workflow.
                options = CalibrationRunOptions(
                    config=None,
                    out=getattr(args, "out", None),
                    method=str(getattr(args, "method", "PARK")),
                    cross_check=bool(getattr(args, "cross_check", False)),
                )
                outcome = replay_calibration(args.replay, options, mount=require_mount)
            return _exit_code(outcome)
        ap.error("no mode selected.")
    except PreflightError as exc:
        logger.error("preflight failed: %s", exc)
        return EXIT_PREFLIGHT
    except Exception as exc:
        logger.error("calibration failed: %s", exc)
        return EXIT_ERROR
    return EXIT_OK


def _guard(device, require_mount: str | None) -> None:
    """Enforce an alias entry point's mount expectation (no-op for the unified entry)."""
    if require_mount is None:
        return
    from scripts.calibrate._cli_common import resolve_mount_guard

    resolve_mount_guard(device, require_mount, other_command="jiuwensymbiosis-calibrate-hand-eye")


if __name__ == "__main__":
    raise SystemExit(main())
