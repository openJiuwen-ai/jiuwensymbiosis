# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Generic hand-eye calibration JSON loader.

Per-vendor adapters get a customized loader via the keyword args:

  load_calibration(path, *, frame_field, legacy_field, env_var)

Each vendor decides which 4x4 SE(3) field it stores the camera pose in.
A 6-DoF arm would use ``tf_flange_cam``
(camera bolted to the flange) or ``tf_base_cam`` (eye-to-hand). The
loader does NOT interpret the frame; it just parses the JSON.

Legacy schema migration:
  Older calibrations (schema_version < 2) used a different field name
  (``legacy_field``). Loading one raises ``LegacyCalibrationError`` unless
  the user opts in via ``env_var``. The opt-in path remaps the legacy field
  into ``frame_field`` so downstream code paths stay uniform; the geometry
  is annotated as ``_legacy_remap_from`` so degraded-accuracy modes can
  detect themselves.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2


class LegacyCalibrationError(RuntimeError):
    """Raised when a calibration uses an older, incompatible schema.

    The user has not opted into the degraded-geometry fallback via the
    env var. The fix is to re-run calibration to produce schema_version
    >= ``CURRENT_SCHEMA_VERSION`` with the new ``frame_field``.
    """


def load_calibration(
    path: str | Path,
    *,
    frame_field: str = "T_J3link_cam",
    legacy_field: str = "T_TCP_cam",
    env_var: str = "JIUWEN_ALLOW_LEGACY_CALIB",
) -> dict[str, Any]:
    """Load a calibration JSON. Returns a dict with numpy arrays where natural.

    Always exposes the camera pose under ``frame_field`` on the returned dict
    (a 4x4 ``np.ndarray``). Other top-level fields (``intrinsics``, ``object``)
    are pre-converted to numpy.

    Args:
      path: JSON file.
      frame_field: Name of the new-schema camera-pose field. The returned dict
        always has this key populated, even when loading a legacy file with
        the opt-in fallback.
      legacy_field: Name of the OLD-schema camera-pose field. Files using
        only this field raise ``LegacyCalibrationError`` unless ``env_var``
        is set.
      env_var: Environment variable whose presence (truthy value) opts the
        user into loading legacy files with degraded-accuracy geometry.
    """
    payload = json.loads(Path(path).read_text())
    version = int(payload.get("schema_version", 1))
    allow_legacy = os.environ.get(env_var, "") not in ("", "0", "false", "False")
    has_new = frame_field in payload
    has_old = legacy_field in payload

    if has_new:
        payload[frame_field]["matrix_4x4"] = np.asarray(payload[frame_field]["matrix_4x4"], dtype=np.float64)
        if version < CURRENT_SCHEMA_VERSION:
            logger.warning(
                "[calib] schema_version=%d but %s present; treating as new schema. "
                "Consider bumping schema_version to %d.",
                version,
                frame_field,
                CURRENT_SCHEMA_VERSION,
            )
    elif has_old:
        msg = (
            f"[calib] {path} uses legacy schema ({legacy_field}, "
            f"schema_version={version}). The geometry model has been upgraded "
            f"— legacy calibrations are NOT compatible because their "
            f"matrix_4x4 encodes a different camera mount frame than the new "
            f"{frame_field}. Re-run calibration to produce schema_version>="
            f"{CURRENT_SCHEMA_VERSION}."
        )
        if not allow_legacy:
            raise LegacyCalibrationError(
                msg + f" (set {env_var}=1 to fall back to the legacy geometry as a degraded path.)"
            )
        logger.warning(
            "%s\n[calib] %s=1 set; remapping %s → %s. Back-projection accuracy "
            "will fall off as the kinematic state drifts from the calibration "
            "snapshot.",
            msg,
            env_var,
            legacy_field,
            frame_field,
        )
        payload[frame_field] = {
            "matrix_4x4": np.asarray(payload[legacy_field]["matrix_4x4"], dtype=np.float64),
            "_legacy_remap_from": legacy_field,
        }
    else:
        raise ValueError(
            f"[calib] {path} is missing both {frame_field} (new) and "
            f"{legacy_field} (legacy) — calibration file is malformed."
        )

    payload["intrinsics"] = np.asarray(payload["intrinsics"], dtype=np.float64)
    payload["object"]["xyz_base_mm"] = np.asarray(payload["object"]["xyz_base_mm"], dtype=np.float64)
    return payload
