"""Shared loopback transport and native protocol checks, without Isaac imports."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jiuwensymbiosis.simulators.robosim.config import RobosimConfig
from jiuwensymbiosis.simulators.robosim.operations import CommandCoordinator, RobosimError, coordinator_for


def read_robot(request: Callable[..., dict]) -> dict | None:
    """Represent only the native robot_not_loaded response as an empty world."""
    try:
        return request("GET", "/robot/state")
    except RobosimError as exc:
        if exc.code != "robot_not_loaded":
            raise
        return None


def ensure_robot(request: Callable[..., dict], robot_type: str, state: dict | None, *, load: bool = False) -> dict:
    """Validate a fresh state, loading only an explicitly allowed empty world."""
    loaded = state is None
    if state is None:
        if not load:
            raise RobosimError("Load a robot first", code="robot_not_loaded")
        state = request(
            "POST", "/robot", {"robot_type": robot_type, "init_pose": {"position": {"x": 0, "y": 0, "z": 0}}}
        )
    if state.get("robot_type") != robot_type:
        raise RobosimError(
            "Robot loading was not confirmed" if loaded else "后台已加载其它机器人，不会自动替换。",
            code="robosim_load_unconfirmed" if loaded else "robosim_robot_mismatch",
            outcome_unknown=loaded,
        )
    return state


def identity_token(identity: dict, *, scene: bool = False) -> tuple:
    """Project a validated snapshot to robot identity, optionally bound to a scene."""
    keys = (
        ("service_instance_id", "robot_instance_id", "scene_instance_id")
        if scene
        else ("service_instance_id", "robot_instance_id")
    )
    return tuple(identity.get(key) for key in keys)


def read_identity(request: Callable[..., dict]) -> dict:
    """Read and validate one bridge snapshot; callers retain their operation scope."""
    try:
        identity = request("GET", "/symbiosis/workspace")
    except RobosimError as exc:
        if getattr(exc.__cause__, "code", None) == 404:
            raise RobosimError(
                "Restart using the current Symbiosis launcher to enable workspace identity",
                code="robosim_identity_required",
            ) from exc
        raise
    service_id, robot_id = identity_token(identity)
    if identity.get("bridge_version") != 1 or not isinstance(service_id, str) or not service_id:
        raise RobosimError("Restart using the current Symbiosis launcher", code="robosim_identity_required")
    if robot_id is not None and (not isinstance(robot_id, str) or not robot_id):
        raise RobosimError("Invalid robot identity", code="robosim_protocol_error")
    return identity


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "RoboSim redirects are disabled", headers, fp)


class RobosimClient:
    """Decode native envelopes without replaying writes or following redirects."""

    def __init__(self, cfg: RobosimConfig, *, coordinator: CommandCoordinator | None = None) -> None:
        self.cfg = cfg
        self.commands = coordinator or coordinator_for(cfg.service_url)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._closed = False

    def close(self) -> None:
        """Release this client only; unresolved command ownership remains shared."""
        self._closed = True

    def request(self, method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
        """Make one request to a relative path on the configured loopback origin."""
        if self._closed:
            raise RobosimError("Client is closed", code="robosim_disconnected")
        method = method.upper()
        parsed = urlsplit(path)
        if (
            method not in {"GET", "POST", "PATCH", "DELETE"}
            or not path.startswith("/")
            or parsed.netloc
            or parsed.scheme
            or parsed.fragment
        ):
            raise ValueError("Expected a supported method and a relative RoboSim path")
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        writing = method != "GET"
        request = Request(
            self.cfg.service_url.rstrip("/") + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with self.commands.command(f"{method} {path}") if writing else nullcontext():
            try:
                with self._opener.open(request, timeout=self.cfg.request_timeout_s) as response:
                    result = json.load(response)
            except HTTPError as exc:
                try:
                    error = json.loads(exc.read())
                except (ValueError, OSError):
                    error = {}
                detail = error.get("detail", error) if isinstance(error, dict) else {}
                code = detail.get("failure_reason") if isinstance(detail, dict) else None
                raise RobosimError(
                    f"RoboSim {path}: HTTP {exc.code}",
                    code=code or "robosim_http_error",
                    outcome_unknown=writing and exc.code not in {400, 401, 403, 404, 405, 409, 422},
                ) from exc
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                raise RobosimError(
                    f"RoboSim {path}: transport failed; execution outcome is unknown",
                    code="robosim_transport_error",
                    outcome_unknown=writing,
                ) from exc
            except (ValueError, UnicodeError) as exc:
                raise RobosimError(
                    "Expected JSON response", code="robosim_protocol_error", outcome_unknown=writing
                ) from exc
            return self._decode(result, writing=writing)

    @staticmethod
    def _decode(result: Any, *, writing: bool) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RobosimError("Expected JSON object", code="robosim_protocol_error", outcome_unknown=writing)
        if "success" not in result:
            return result
        if result["success"] is False:
            raise RobosimError(
                result.get("error") or "RoboSim action failed",
                code=result.get("failure_reason") or "robosim_action_failed",
            )
        if result["success"] is not True or not isinstance(result.get("data", {}), (dict, type(None))):
            raise RobosimError("Invalid RoboSim envelope", code="robosim_protocol_error", outcome_unknown=writing)
        return result.get("data") or {}
