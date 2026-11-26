# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared control/status message contract for the preview parent and its worker.

The mmap mailbox only moves bounded JSON; both processes interpret the request
and result fields. This module names those fields and provides explicit
construction, parsing and validity helpers so a protocol change lands in one
place instead of being copied across the parent, the worker and test doubles.

Standard-library only: the standalone worker seeds a minimal package and imports
this module relatively, without package initialization, like :mod:`ipc`.
Importing it must not pull the core package, ``openjiuwen`` or a GPU SDK.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict


# These TypedDicts document the wire shapes. The builders return plain JSON
# objects because both processes pass them through untyped mailbox dictionaries.
class CaptureRequest(TypedDict):
    id: str
    scene_instance_id: str
    config_revision: int
    camera_revision: int
    camera_name: str
    include_depth: bool
    depth_stride: int
    expires_at: float


class CaptureReply(TypedDict, total=False):
    id: str
    error: str
    captured_at: float
    captured_utc: str
    scene_instance_id: str
    config_revision: int
    camera_revision: int
    camera_name: str
    camera: dict
    step: int
    width: int
    height: int
    capture_duration_ms: float


class PickRequest(TypedDict):
    id: str
    action: str
    u: float
    v: float
    scene_instance_id: str
    config_revision: int
    camera_revision: int


class PickReply(TypedDict, total=False):
    id: str
    error: str
    duration_ms: float


class BaseCommandRequest(TypedDict):
    id: str
    vx: float
    vy: float
    wz: float
    scene_instance_id: str
    config_revision: int
    expires_at: float


class BaseCommandReply(TypedDict, total=False):
    id: str
    applied: bool
    error: str


class BaseMotionRequest(TypedDict, total=False):
    id: str
    scene_instance_id: str
    config_revision: int
    lease_until: float
    cancelled: bool


class SceneQueryRequest(TypedDict):
    id: str
    operation: str
    params: dict
    scene_instance_id: str
    config_revision: int
    expires_at: float


class ResetRequest(TypedDict):
    id: str
    scene_instance_id: str
    config_revision: int


_MOTION_BASE_FIELDS = frozenset({"id", "scene_instance_id", "config_revision", "lease_until", "cancelled"})


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _revision(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _required(message: Any, keys: tuple[str, ...], label: str) -> dict:
    if not isinstance(message, dict):
        raise ValueError(f"{label} must be an object")
    missing = [key for key in keys if key not in message]
    if missing:
        raise ValueError(f"{label} is missing {', '.join(missing)}")
    return message


def build_control_message(
    *,
    streaming: bool,
    camera: dict,
    settings: dict,
    config_revision: int,
    stream_epoch: int,
    keyframe_request: int,
    pick: dict | None = None,
    base_command: dict | None = None,
    base_motion: dict | None = None,
    capture: dict | None = None,
    scene_query: dict | None = None,
    reset: dict | None = None,
) -> dict:
    """Assemble the parent-owned control slots; Mailbox stays transport-only."""
    return {
        "streaming": _boolean(streaming, "control.streaming"),
        "camera": camera,
        "settings": settings,
        "config_revision": _revision(config_revision, "control.config_revision"),
        "stream_epoch": _revision(stream_epoch, "control.stream_epoch"),
        "keyframe_request": _revision(keyframe_request, "control.keyframe_request"),
        "pick": pick,
        "base_command": base_command,
        "base_motion": base_motion,
        "capture": capture,
        "scene_query": scene_query,
        "reset": reset,
    }


# ---------------------------------------------------------------------------
# capture_image
# ---------------------------------------------------------------------------
def build_capture_request(
    request_id: str,
    *,
    scene_instance_id: str,
    config_revision: int,
    camera_revision: int,
    camera_name: str,
    include_depth: bool,
    depth_stride: int,
    expires_at: float,
) -> dict[str, Any]:
    """Build a capture request for the existing control slot."""
    if _boolean(include_depth, "capture.include_depth"):
        validate_capture_depth(camera_name, depth_stride)
    return {
        "id": _identifier(request_id, "capture.id"),
        "scene_instance_id": _identifier(scene_instance_id, "capture.scene_instance_id"),
        "config_revision": _revision(config_revision, "capture.config_revision"),
        "camera_revision": _revision(camera_revision, "capture.camera_revision"),
        "camera_name": _identifier(camera_name, "capture.camera_name"),
        "include_depth": include_depth,
        "depth_stride": depth_stride,
        "expires_at": _number(expires_at, "capture.expires_at"),
    }


def validate_capture_depth(camera_name: str, depth_stride: int) -> None:
    """Reject an RGB-D request no worker could serve.

    The parent calls this before touching the worker so an impossible request
    fails immediately; parsing calls it again for messages from older or foreign
    senders, which keeps one definition of "valid RGB-D".
    """
    if camera_name == "observer":
        raise ValueError("RGB-D requires a robot camera and sampling stride in [4, 32]")
    stride_valid = isinstance(depth_stride, int) and not isinstance(depth_stride, bool) and 4 <= depth_stride <= 32
    if not stride_valid:
        raise ValueError("RGB-D requires a robot camera and sampling stride in [4, 32]")


def parse_capture_request(value: Any) -> dict[str, Any]:
    """Validate a worker-bound capture request, applying the documented defaults."""
    message = _required(
        value,
        ("id", "scene_instance_id", "config_revision", "camera_revision", "expires_at"),
        "capture",
    )
    return build_capture_request(
        message["id"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
        camera_revision=message["camera_revision"],
        camera_name=message.get("camera_name", "observer"),
        include_depth=message.get("include_depth", False),
        depth_stride=message.get("depth_stride", 8),
        expires_at=message["expires_at"],
    )


def capture_request_invalid(
    request: dict,
    *,
    config_revision: int,
    scene_instance_id: str,
    camera_revision: int,
    now: float,
) -> bool:
    """Worker admission: stale binding, changed view or expired deadline."""
    observer = request.get("camera_name", "observer") == "observer"
    binding_changed = (
        request.get("config_revision") != config_revision
        or request.get("scene_instance_id") != scene_instance_id
        or (observer and request.get("camera_revision") != camera_revision)
    )
    expires_at = request.get("expires_at")
    return bool(binding_changed or not isinstance(expires_at, (int, float)) or now > expires_at)


def build_capture_reply(
    request_id: str,
    *,
    captured_at: float,
    captured_utc: str,
    scene_instance_id: str,
    config_revision: int,
    camera_revision: int,
    camera_name: str,
    camera: dict,
    step: int,
    width: int,
    height: int,
    capture_duration_ms: float,
    base_sample: dict | None = None,
    body_bounds_m: list | None = None,
    motion_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("capture dimensions must be positive integers")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("capture dimensions must be positive integers")
    reply = {
        "id": _identifier(request_id, "capture.id"),
        "captured_at": _number(captured_at, "capture.captured_at"),
        "captured_utc": _identifier(captured_utc, "capture.captured_utc"),
        "scene_instance_id": _identifier(scene_instance_id, "capture.scene_instance_id"),
        "config_revision": _revision(config_revision, "capture.config_revision"),
        "camera_revision": _revision(camera_revision, "capture.camera_revision"),
        "camera_name": _identifier(camera_name, "capture.camera_name"),
        "camera": camera,
        "step": _revision(step, "capture.step"),
        "width": width,
        "height": height,
        "capture_duration_ms": _number(capture_duration_ms, "capture.capture_duration_ms"),
    }
    if base_sample is not None:
        if body_bounds_m is None:
            raise ValueError("RGB-D capture requires body bounds")
        reply.update(base_sample=base_sample, body_bounds_m=body_bounds_m, motion_id=motion_id)
    return reply


def build_capture_error(request_id: str, reason: str = "capture_failed") -> dict[str, Any]:
    """An id-carrying worker error for a rejected capture request."""
    return {"id": _identifier(request_id, "capture.id"), "error": _identifier(reason, "capture.error")}


def capture_reply_matches(reply: dict, request: dict, *, now: float, max_age_s: float = 2.0) -> bool:
    """Parent admission: the reply must echo this request's binding and be fresh."""
    binding = ("scene_instance_id", "config_revision", "camera_revision", "camera_name")
    return all(reply.get(key) == request[key] for key in binding) and now - reply.get("captured_at", 0) <= max_age_s


# ---------------------------------------------------------------------------
# select (pick / clear)
# ---------------------------------------------------------------------------
def build_pick_request(
    request_id: str,
    *,
    action: str,
    u: float,
    v: float,
    scene_instance_id: str,
    config_revision: int,
    camera_revision: int,
) -> dict[str, Any]:
    """Build a pick/clear request after validating the action and pixel."""
    if action not in {"pick", "clear"}:
        raise ValueError("pick action must be pick or clear")
    for label, coordinate in (("u", u), ("v", v)):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) or not 0 <= coordinate <= 1:
            raise ValueError(f"pick.{label} must be within [0, 1]")
    return {
        "id": _identifier(request_id, "pick.id"),
        "action": action,
        "u": float(u),
        "v": float(v),
        "scene_instance_id": _identifier(scene_instance_id, "pick.scene_instance_id"),
        "config_revision": _revision(config_revision, "pick.config_revision"),
        "camera_revision": _revision(camera_revision, "pick.camera_revision"),
    }


def parse_pick_request(value: Any) -> dict[str, Any]:
    """Validate a worker-bound pick request."""
    message = _required(
        value,
        ("id", "action", "u", "v", "scene_instance_id", "config_revision", "camera_revision"),
        "pick",
    )
    built = build_pick_request(
        message["id"],
        action=message["action"],
        u=message["u"],
        v=message["v"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
        camera_revision=message["camera_revision"],
    )
    return built


def pick_binding_error(
    request: dict, *, config_revision: int, scene_instance_id: str, camera_revision: int
) -> str | None:
    """Return ``scene``/``camera`` for a stale request, or ``None``.

    ``clear`` deliberately does not require a current camera revision; the
    caller chooses its own user-facing message for each reason.
    """
    if request.get("config_revision") != config_revision or request.get("scene_instance_id") != scene_instance_id:
        return "scene"
    if request.get("action") == "pick" and request.get("camera_revision") != camera_revision:
        return "camera"
    return None


def build_pick_reply(request_id: str, *, duration_ms: float) -> dict[str, Any]:
    """An id-carrying pick reply with the worker's measured duration."""
    return {
        "id": _identifier(request_id, "pick.id"),
        "duration_ms": round(_number(duration_ms, "pick.duration_ms"), 2),
    }


# ---------------------------------------------------------------------------
# base_command (manual bounded pulse)
# ---------------------------------------------------------------------------
def build_base_command_request(
    request_id: str,
    *,
    vx: float,
    vy: float,
    wz: float,
    scene_instance_id: str,
    config_revision: int,
    expires_at: float,
) -> dict[str, Any]:
    """Build one bounded manual base pulse."""
    return {
        "id": _identifier(request_id, "base_command.id"),
        "vx": _number(vx, "base_command.vx"),
        "vy": _number(vy, "base_command.vy"),
        "wz": _number(wz, "base_command.wz"),
        "scene_instance_id": _identifier(scene_instance_id, "base_command.scene_instance_id"),
        "config_revision": _revision(config_revision, "base_command.config_revision"),
        "expires_at": _number(expires_at, "base_command.expires_at"),
    }


def parse_base_command_request(value: Any) -> dict[str, Any]:
    """Validate a worker-bound manual base pulse."""
    message = _required(
        value,
        ("id", "vx", "vy", "wz", "scene_instance_id", "config_revision", "expires_at"),
        "base_command",
    )
    built = build_base_command_request(
        message["id"],
        vx=message["vx"],
        vy=message["vy"],
        wz=message["wz"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
        expires_at=message["expires_at"],
    )
    return built


def base_command_valid(
    request: dict | None, *, config_revision: int, scene_instance_id: str, streaming: bool, now: float
) -> bool:
    """Worker admission for one manual pulse; no video connection means no pulse."""
    if not request or not streaming:
        return False
    bound = request.get("scene_instance_id") == scene_instance_id and request.get("config_revision") == config_revision
    return bool(bound and request.get("expires_at", 0) > now)


def build_base_command_reply(request_id: str, *, applied: bool, error: str | None = None) -> dict[str, Any]:
    """An id-carrying result for one manual base pulse."""
    reply: dict[str, Any] = {"id": request_id, "applied": applied}
    if error is not None:
        reply["error"] = error
    return reply


# ---------------------------------------------------------------------------
# base_motion (relative motion, policy actions and arm plans)
# ---------------------------------------------------------------------------
def build_base_motion_request(
    request_id: str,
    *,
    scene_instance_id: str,
    config_revision: int,
    lease_until: float,
    cancelled: bool = False,
    **payload: Any,
) -> dict:
    """Common motion envelope; the payload keeps each action's own fields."""
    reserved = sorted(_MOTION_BASE_FIELDS.intersection(payload))
    if reserved:
        raise ValueError(f"base_motion payload must not redefine {', '.join(reserved)}")
    return {
        "id": _identifier(request_id, "base_motion.id"),
        "scene_instance_id": _identifier(scene_instance_id, "base_motion.scene_instance_id"),
        "config_revision": _revision(config_revision, "base_motion.config_revision"),
        **payload,
        "lease_until": _number(lease_until, "base_motion.lease_until"),
        "cancelled": _boolean(cancelled, "base_motion.cancelled"),
    }


def parse_base_motion_request(value: Any) -> dict[str, Any]:
    """Validate the motion envelope; the action payload passes through unchanged."""
    message = _required(
        value,
        ("id", "scene_instance_id", "config_revision", "lease_until", "cancelled"),
        "base_motion",
    )
    payload = {key: item for key, item in message.items() if key not in _MOTION_BASE_FIELDS}
    return build_base_motion_request(
        message["id"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
        lease_until=message["lease_until"],
        cancelled=message["cancelled"],
        **payload,
    )


def build_rejected_reply(kind: str, request_id: Any, *, error: str | None = None, **extra: Any) -> dict[str, Any]:
    """A request-level rejection that intentionally does not validate a malformed id."""
    reply: dict[str, Any] = {"id": request_id, "error": error or f"{kind}_failed"}
    reply.update(extra)
    return reply


def base_motion_binding(request: dict) -> tuple[str, int]:
    """The scene/config a motion request is bound to."""
    return (
        _identifier(request.get("scene_instance_id"), "base_motion.scene_instance_id"),
        _revision(request.get("config_revision"), "base_motion.config_revision"),
    )


# ---------------------------------------------------------------------------
# scene_query and reset
# ---------------------------------------------------------------------------
def build_scene_query_request(
    request_id: str,
    *,
    operation: str,
    params: dict,
    scene_instance_id: str,
    config_revision: int,
    expires_at: float,
) -> dict[str, Any]:
    """Build a read-only scene query with its binding and deadline."""
    if not isinstance(params, dict):
        raise ValueError("scene_query.params must be an object")
    return {
        "id": _identifier(request_id, "scene_query.id"),
        "operation": _identifier(operation, "scene_query.operation"),
        "params": dict(params),
        "scene_instance_id": _identifier(scene_instance_id, "scene_query.scene_instance_id"),
        "config_revision": _revision(config_revision, "scene_query.config_revision"),
        "expires_at": _number(expires_at, "scene_query.expires_at"),
    }


def parse_scene_query_request(value: Any) -> dict[str, Any]:
    """Validate a worker-bound scene query."""
    message = _required(
        value,
        ("id", "operation", "params", "scene_instance_id", "config_revision", "expires_at"),
        "scene_query",
    )
    built = build_scene_query_request(
        message["id"],
        operation=message["operation"],
        params=message["params"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
        expires_at=message["expires_at"],
    )
    return built


def binding_mismatch(request: dict, *, config_revision: int, scene_instance_id: str) -> bool:
    """True when a request names a different world revision or scene instance."""
    return request.get("config_revision") != config_revision or request.get("scene_instance_id") != scene_instance_id


def build_reset_request(request_id: str, *, scene_instance_id: str, config_revision: int) -> dict[str, Any]:
    """Build a reset request bound to the world it resets."""
    return {
        "id": _identifier(request_id, "reset.id"),
        "scene_instance_id": _identifier(scene_instance_id, "reset.scene_instance_id"),
        "config_revision": _revision(config_revision, "reset.config_revision"),
    }


def parse_reset_request(value: Any) -> dict[str, Any]:
    """Validate a worker-bound reset request."""
    message = _required(value, ("id", "scene_instance_id", "config_revision"), "reset")
    built = build_reset_request(
        message["id"],
        scene_instance_id=message["scene_instance_id"],
        config_revision=message["config_revision"],
    )
    return built


__all__ = [
    "BaseCommandReply",
    "BaseCommandRequest",
    "BaseMotionRequest",
    "CaptureReply",
    "CaptureRequest",
    "PickReply",
    "PickRequest",
    "ResetRequest",
    "SceneQueryRequest",
    "base_command_valid",
    "base_motion_binding",
    "binding_mismatch",
    "build_base_command_reply",
    "build_base_command_request",
    "build_base_motion_request",
    "build_capture_error",
    "build_rejected_reply",
    "build_capture_reply",
    "build_capture_request",
    "build_control_message",
    "build_pick_reply",
    "build_pick_request",
    "build_reset_request",
    "build_scene_query_request",
    "capture_reply_matches",
    "capture_request_invalid",
    "validate_capture_depth",
    "parse_base_command_request",
    "parse_base_motion_request",
    "parse_capture_request",
    "parse_pick_request",
    "parse_reset_request",
    "parse_scene_query_request",
    "pick_binding_error",
]
