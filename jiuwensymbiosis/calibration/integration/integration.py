# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Calibration-owned framework integration for robot adapters.

The domain and workflow modules in :mod:`jiuwensymbiosis.calibration` are
body-agnostic.  This module is the deliberately small integration boundary
that connects those workflows to a JiuwenSymbiosis session, configuration,
runtime artifact loader and calibration hardware-port view.

Core adapter packages do not import this module.  Discovery therefore lives
on the calibration side of the boundary and is lazy: importing the domain
package or a robot adapter does not import either built-in adapter's session
composition until a caller asks for its spec.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from jiuwensymbiosis.calibration.artifacts.artifacts import save_eye_to_hand_calibration
from jiuwensymbiosis.calibration.domain.quality import target_consistency_report
from jiuwensymbiosis.calibration.domain.solver import save_calibration
from jiuwensymbiosis.calibration.workflows.preflight import validate_mount
from jiuwensymbiosis.calibration_schema import EYE_IN_HAND_FRAME_FIELD, frame_field_for_mount

_ADAPTER_PACKAGE_PREFIX = "jiuwensymbiosis.adapters."
_CALIBRATION_ADAPTER_PREFIX = "jiuwensymbiosis.calibration.adapters."


def adapter_package(cfg_or_module: Any) -> str:
    """Normalize a Jiuwen adapter config, module or module path to a package."""
    if isinstance(cfg_or_module, str):
        module = cfg_or_module
    elif isinstance(cfg_or_module, ModuleType):
        module = cfg_or_module.__name__
    else:
        module = str(
            getattr(cfg_or_module, "__module__", "") or getattr(cfg_or_module, "__name__", "") or cfg_or_module
        )
    if not module.startswith(_ADAPTER_PACKAGE_PREFIX):
        raise ValueError(
            f"adapter module {module!r} is not under {_ADAPTER_PACKAGE_PREFIX!r}; "
            "cannot derive a canonical adapter package."
        )
    prefix_len = len(_ADAPTER_PACKAGE_PREFIX)
    name = module[prefix_len:].split(".", 1)[0]
    if not name:
        raise ValueError(f"adapter module {module!r} has no package name segment.")
    return f"{_ADAPTER_PACKAGE_PREFIX}{name}"


@runtime_checkable
class SessionFactory(Protocol):
    """Callable returned by ``make_builder`` with its attached loaders."""

    def __call__(self, cfg: Any, *, include_sidecars: bool = True) -> Any:
        pass

    def from_yaml(self, path: str | Path, *, include_sidecars: bool = True) -> Any:
        pass

    def from_dict(self, data: dict[str, Any], *, include_sidecars: bool = True) -> Any:
        pass

    def resource_keys(self, cfg: Any) -> Iterable[str]:
        """Return device identities for the already loaded typed config."""
        pass


@runtime_checkable
class CalibrationArtifactLoader(Protocol):
    """Adapter's canonical runtime calibration loader."""

    def __call__(
        self,
        path: str | Path,
        *,
        mount: Literal["eye_in_hand", "eye_to_hand"],
    ) -> dict[str, Any]:
        pass


@runtime_checkable
class CalibrationDeviceFactory(Protocol):
    """Build a calibration-owned adapter wrapper over a session Env."""

    def __call__(self, env: Any) -> Any:
        pass


@dataclass(frozen=True)
class CalibrationAdapterSpec:
    """Explicit integration contract for one robot adapter.

    ``session_factory`` (with ``from_yaml`` / ``from_dict``) is the single way
    to load configuration and build a session; there is deliberately no separate
    ``load_config`` field, so the config loader and session builder cannot drift
    and the camera mount is resolved once from the constructed device.
    """

    package: str
    session_factory: SessionFactory
    load_calibration_artifact: CalibrationArtifactLoader
    make_calibration_device: CalibrationDeviceFactory


def load_adapter_spec(adapter_module: str | ModuleType) -> CalibrationAdapterSpec:
    """Load the calibration-owned spec for an adapter package.

    ``adapter_module`` is normally the canonical package string from a YAML
    config, for example ``"jiuwensymbiosis.adapters.piper"``.  A module object
    is accepted as a convenience.  The corresponding calibration wrapper is
    discovered by replacing the core adapter prefix with
    ``jiuwensymbiosis.calibration.adapters.``; no central adapter registry is
    needed.  Specs are imported lazily so the calibration domain remains
    usable without loading hardware adapter code.
    """
    package = adapter_package(adapter_module)
    prefix_len = len(_ADAPTER_PACKAGE_PREFIX)
    adapter_name = package[prefix_len:]
    spec_module_name = f"{_CALIBRATION_ADAPTER_PREFIX}{adapter_name}"
    try:
        module = import_module(spec_module_name)
    except ModuleNotFoundError as exc:
        # Only a missing wrapper itself is a discovery error.  If the wrapper
        # imported successfully far enough to expose a missing optional SDK or
        # another dependency, preserve that original error for the caller.
        if exc.name == spec_module_name:
            raise ValueError(
                f"calibration adapter wrapper {spec_module_name!r} was not found; "
                f"add a module at {spec_module_name.replace('.', '/')}.py "
                "that exposes CALIBRATION_ADAPTER_SPEC."
            ) from exc
        raise
    spec = getattr(module, "CALIBRATION_ADAPTER_SPEC", None)
    if not isinstance(spec, CalibrationAdapterSpec):
        raise TypeError(f"{spec_module_name} must expose CALIBRATION_ADAPTER_SPEC as CalibrationAdapterSpec.")
    if spec.package != package:
        raise ValueError(
            f"{spec_module_name} exposes spec.package={spec.package!r}, "
            f"but the requested adapter package is {package!r}."
        )
    return spec


@dataclass(frozen=True)
class SolvedCalibration:
    """One solve's publishable output: the camera pose, its frame, and intrinsics.

    ``tf_camera`` is the solved camera pose in the frame ``mount`` selects — the
    two mounts publish it under different schema fields (``T_base_cam`` for
    eye-to-hand, ``T_flange_cam`` for eye-in-hand). The four values are produced
    together by the solver and consumed together by every saver and loader, so
    they travel as one value rather than as a parameter list to keep in order.
    """

    tf_camera: np.ndarray
    intrinsics: np.ndarray
    mount: Literal["eye_in_hand", "eye_to_hand"]
    t_flange_target: np.ndarray | None = None


def validate_adapter_reload(
    spec: CalibrationAdapterSpec,
    temporary_json: str | Path,
    solved: SolvedCalibration,
    stations: list,
) -> None:
    """Round-trip a formal artifact through the adapter's runtime loader.

    Keeping the saver and loader frame field in lockstep is the point of this
    calibration-owned reload gate.
    """
    tf_base_cam = solved.tf_camera
    intrinsics = solved.intrinsics
    t_flange_target = solved.t_flange_target
    validated_mount = validate_mount(solved.mount, source="mount")
    tmp_path = Path(temporary_json)
    if validated_mount == "eye_to_hand":
        save_eye_to_hand_calibration(
            tmp_path,
            tf_base_cam,
            intrinsics,
            mount=validated_mount,
            t_flange_target=t_flange_target,
        )
    else:
        # ``save_eye_to_hand_calibration`` intentionally publishes the
        # eye-to-hand ``T_base_cam`` field.  Eye-in-hand adapters load the
        # same pose under ``T_flange_cam``; use the frame-explicit saver here
        # so the smoke test exercises the adapter's real public contract.
        save_calibration(
            tmp_path,
            tf_base_cam,
            intrinsics,
            None,
            frame_field=EYE_IN_HAND_FRAME_FIELD,
            top_comment="temporary eye-in-hand reload validation artifact",
        )
    calib = spec.load_calibration_artifact(tmp_path, mount=validated_mount)
    field_name = frame_field_for_mount(validated_mount)
    entry = calib.get(field_name) if isinstance(calib, dict) else None
    if not isinstance(entry, dict) or "matrix_4x4" not in entry:
        raise RuntimeError(f"adapter reload: {field_name} missing or malformed in loader output")
    reloaded = np.asarray(entry["matrix_4x4"], dtype=np.float64)
    expected = np.round(np.asarray(tf_base_cam, dtype=np.float64), 6)
    if not np.allclose(reloaded, expected, atol=1e-5):
        raise RuntimeError("adapter reload: matrix round-trip mismatch — loader returned a different camera pose")
    if validated_mount == "eye_to_hand" and stations:
        consistency = target_consistency_report(stations, reloaded, invariant_frame="flange_target")
        if consistency.translation_residual_mm.max > 10.0:
            raise RuntimeError(
                "adapter reload: projection/rigidity failed "
                f"(max translation residual {consistency.translation_residual_mm.max:.2f}mm > 10mm)"
            )


__all__ = [
    "CalibrationAdapterSpec",
    "CalibrationArtifactLoader",
    "CalibrationDeviceFactory",
    "SessionFactory",
    "adapter_package",
    "load_adapter_spec",
    "SolvedCalibration",
    "validate_adapter_reload",
]
