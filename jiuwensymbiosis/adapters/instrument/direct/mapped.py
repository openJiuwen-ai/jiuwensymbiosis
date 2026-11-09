# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""Small, explicit wire mappings; no executable code is loaded from documents."""

from __future__ import annotations

import json
import math
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from urllib import error, request
from urllib.parse import urlsplit

from jiuwensymbiosis.adapters.instrument.direct.errors import CommandNotSent, DriverError
from jiuwensymbiosis.adapters.instrument.direct.vocabulary import CONDITIONS

from .mapping_rules import _MISSING, check_response, matches, select


def _render(value, parameters):
    if isinstance(value, dict):
        if set(value) == {"$parameter"}:
            return parameters[value["$parameter"]]
        return {key: _render(item, parameters) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, parameters) for item in value]
    return value


def _unique_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON response key: {key}")
        value[key] = item
    return value


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MappedProtocolAdapter:
    """A command succeeds only as a protocol operation, never as a physical cycle."""

    supports_cancel = False

    def __init__(self, profile: dict, registry):
        self.profile, self.registry = deepcopy(profile), registry
        self.connection = self.profile["connection"]
        self.actions = {
            name: deepcopy(registry.definitions[name])
            for name in (profile.get("operation_ids") or [profile["operation_id"]])
        }
        self.transport = next(iter(self.actions.values()))["mapping"]["transport"]
        self.timeout = self.connection.get("timeout_s", 3.0)
        self._serial = None
        self._lock = threading.RLock()
        self._last = None
        self._pending = None
        self._opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        self.driver = self
        self.mutation_attempts = 0

    def snapshot(self) -> dict:
        protocol = self.profile.get("protocol", {})
        if not protocol:
            return {"device_state": "unknown", "status_available": False, "source": "unknown"}
        with self._lock:
            value = self._observe(protocol["status_action"])
            state = select(value, protocol["state_path"])
            # Serial status queries may return numeric codes. The profile maps
            # their JSON spelling explicitly (e.g. "0.0"), never truthiness.
            code = state if isinstance(state, str) else json.dumps(state) if type(state) in (int, float) else None
            state = protocol["state_values"].get(code, "unknown")
            if state == "idle" and not matches(value, protocol["idle_checks"]):
                state = "unknown"
            # Sampling metadata must not turn every read into a state revision.
            # Action responses and completion polling retain the original payload.
            stable = deepcopy(value)
            for path in protocol.get("volatile_paths", []):
                parent = stable
                for key in path[:-1]:
                    parent = parent.get(key) if isinstance(parent, dict) else None
                if isinstance(parent, dict):
                    parent.pop(path[-1], None)
            return {
                "device_state": state,
                "status_available": True,
                "source": "hardware",
                "observed_at": datetime.now(UTC).isoformat(),
                "state_scope": "gateway",
                "physical_completion_verified": False,
                "gateway": stable,
                # Only requested, registered conditions from this fresh status
                # response can satisfy admission. Strings/numbers are not facts.
                **{
                    condition: value[condition]
                    for condition in self.profile.get("required_confirmations", ())
                    if condition in CONDITIONS and isinstance(value.get(condition), bool)
                },
            }

    def _observe(self, action):
        mapping = self.actions[action]["mapping"]
        call = self._serial_request if mapping["transport"] == "serial_line" else self._http_request
        return call(mapping, {}, read_only=True)[0]

    def _guard(self, rule, parameters):
        value = self._observe(rule["observe"])
        if not matches(value, rule["checks"], parameters):
            raise DriverError("PRECONDITION", "Documented gateway guard did not match")
        return value

    @staticmethod
    def can_cancel(action=None) -> bool:
        return False

    def start(self, action: str, parameters: dict) -> dict:
        if action not in self.actions:
            raise DriverError("UNSUPPORTED_OPERATION", "Action has no installed protocol mapping")
        self.registry.validate_parameters(action, parameters)
        definition = self.actions[action]
        mapping = definition["mapping"]
        with self._lock:
            if self._pending is not None:
                raise DriverError("DEVICE_BUSY", "An earlier mapped action still has an unresolved outcome")
            try:
                for guard in mapping.get("guards", []):
                    self._guard(guard, parameters)
                completion = mapping.get("completion")
                baseline = self._observe(completion["observe"]) if completion else None
                if (
                    completion
                    and "event" in completion
                    and not isinstance(select(baseline, completion["event"]["path"], parameters), list)
                ):
                    raise DriverError("PROTOCOL", "Missing baseline event history; command was not sent")
                if (
                    completion
                    and "changed" in completion
                    and select(baseline, completion["changed"], parameters) is _MISSING
                ):
                    raise DriverError("PROTOCOL", "Missing baseline state; command was not sent")
            except Exception as exc:
                # Observations may perform reads, but action I/O has not begun.
                # Pending from an earlier action is rejected outside this block.
                code = exc.code if isinstance(exc, DriverError) else "PRE_DISPATCH"
                raise CommandNotSent(code, "Action rejected before dispatch") from exc
            # Persist in-memory pending intent before I/O: uncertain writes cannot be repeated.
            if completion:
                self._pending = {"completion": completion, "baseline": baseline, "parameters": deepcopy(parameters)}
            if self.transport == "serial_line":
                data, raw = self._serial_request(mapping, parameters, read_only=definition["read_only"])
            else:
                data, raw = self._http_request(mapping, parameters, read_only=definition["read_only"])
            self._last = self._result(data, raw, parameters, running=completion is not None)
            return deepcopy(self._last)

    @staticmethod
    def _result(data, raw, parameters, *, running=False):
        return {
            "status": "running" if running else "succeeded",
            "device_state": "running" if running else "unknown",
            "executed": True,
            "actual_parameters": deepcopy(parameters),
            "data": {**data, "physical_completion_verified": False, "completion_scope": "protocol_response"},
            "evidence": [
                {
                    "field": "command_accepted" if running else "protocol_response_verified",
                    "value": True,
                    "source": "hardware",
                    "observed_at": datetime.now(UTC).isoformat(),
                    "response": raw,
                    "note": "Gateway/protocol evidence; no independently sensed physical completion claim.",
                }
            ],
        }

    def query_action(self, action: str, parameters: dict) -> dict:
        if action not in self.registry.read_only:
            raise DriverError("UNSUPPORTED_OPERATION", "Only read-only mappings may use query_action")
        return self.start(action, parameters)

    def poll(self) -> dict:
        with self._lock:
            if self._pending is not None:
                pending = self._pending
                rule, parameters = pending["completion"], pending["parameters"]
                value = self._observe(rule["observe"])
                fresh = False
                if "event" in rule:
                    event_rule = rule["event"]
                    events = select(value, event_rule["path"], parameters)
                    if not isinstance(events, list):
                        raise DriverError("PROTOCOL", "Completion event history missing", uncertain=True)
                    baseline = select(pending["baseline"], event_rule["path"], parameters)
                    old = {json.dumps(e, sort_keys=True) for e in baseline}
                    for event in events:
                        if json.dumps(event, sort_keys=True) not in old and matches(
                            event, event_rule["match"], parameters
                        ):
                            if not matches(event, event_rule["success"], parameters):
                                raise DriverError(
                                    "PROTOCOL", "New matching action receipt did not confirm success", uncertain=True
                                )
                            fresh = True
                else:
                    current = select(value, rule["changed"], parameters)
                    fresh = current is not _MISSING and current != select(
                        pending["baseline"], rule["changed"], parameters
                    )
                completed = fresh and matches(value, rule["checks"], parameters)
                if completed:
                    for entry in rule.get("additional", []):
                        if not matches(self._observe(entry["observe"]), entry["checks"], parameters):
                            completed = False
                data = {"observation": value}
                if "hold_s" in rule:
                    duration = _render(rule["hold_s"], parameters)
                    if completed:
                        pending.setdefault("hold_started", time.monotonic())
                    elif "hold_started" in pending:
                        raise DriverError(
                            "PROTOCOL", "Gateway completion conditions changed during hold", uncertain=True
                        )
                    elapsed = max(0.0, time.monotonic() - pending["hold_started"]) if "hold_started" in pending else 0.0
                    completed = completed and elapsed >= duration
                    data.update(
                        hold_duration_s=duration,
                        hold_elapsed_s=elapsed,
                        hold_basis="middleware_monotonic_timer_after_fresh_gateway_receipt",
                    )
                self._last = self._result(data, value, parameters, running=not completed)
                if completed:
                    self._pending = None
            if self._last is None:
                raise DriverError("NOT_STARTED", "No protocol response is available")
            return deepcopy(self._last)

    def _serial_request(self, mapping, parameters, *, read_only):
        if self._serial is None:
            import serial

            self._serial = serial.Serial(
                self.connection["port"],
                baudrate=self.connection["baudrate"],
                timeout=self.timeout,
                write_timeout=self.timeout,
                exclusive=True,
            )
        parts = []
        for part in mapping["request"]:
            if isinstance(part, str):
                parts.append(part)
            else:
                value = parameters[part["parameter"]]
                fmt = part.get("format", "str")
                parts.append(str(value) if fmt == "str" else format(value, fmt))
        wire = ("".join(parts) + {"LF": "\n", "CRLF": "\r\n", "NONE": ""}[mapping["line_ending"]]).encode("ascii")
        if not read_only:
            self.mutation_attempts += 1
        try:
            self._serial.reset_input_buffer()
            self._serial.write(wire)
            raw = self._serial.read_until(b"\n", 4097)
        except OSError as exc:
            raise DriverError(
                "COMMUNICATION", "Serial outcome unknown; command is never retried", uncertain=True
            ) from exc
        if not raw.endswith(b"\n") or len(raw) > 4096:
            raise DriverError("PROTOCOL", "Missing or oversized complete response line", uncertain=True)
        try:
            line = raw.decode("ascii").strip()
        except UnicodeError as exc:
            raise DriverError("PROTOCOL", "Response is not ASCII", uncertain=True) from exc
        rule = mapping["response"]
        data = {}
        if "contains" in rule:
            if rule["contains"] not in line:
                raise DriverError("PROTOCOL", "Documented response acknowledgement missing", uncertain=True)
        else:
            if not line.startswith(rule["number_prefix"]):
                raise DriverError("PROTOCOL", "Documented numeric response prefix missing", uncertain=True)
            try:
                prefix_length = len(rule["number_prefix"])
                value = float(line[prefix_length:].strip())
            except ValueError as exc:
                raise DriverError("PROTOCOL", "Response value is not numeric", uncertain=True) from exc
            if not math.isfinite(value):
                raise DriverError("PROTOCOL", "Response value is nonfinite", uncertain=True)
            data[rule["field"]] = value
        return data, line

    def _http_request(self, mapping, parameters, *, read_only):
        base = self.connection["base_url"].rstrip("/")
        parsed = urlsplit(base)
        invalid_origin = parsed.scheme not in {"http", "https"} or not parsed.hostname
        has_suffix = parsed.query or parsed.fragment
        if invalid_origin or parsed.username is not None or has_suffix:
            raise DriverError("CONFIGURATION", "base_url must be an explicit HTTP(S) origin without credentials")
        body = (
            None
            if mapping["method"] == "GET"
            else json.dumps(_render(mapping.get("body", {}), parameters), allow_nan=False).encode()
        )
        path = mapping["path"]
        if isinstance(path, list):
            path = "".join(str(_render(segment, parameters)) for segment in path)
        headers = {"Content-Type": "application/json"}
        req = request.Request(base + path, body, headers, method=mapping["method"])
        if not read_only:
            self.mutation_attempts += 1
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                raw = response.read(65537)
                status = response.status
        except error.HTTPError as exc:
            # A gateway response is distinct from a network failure, but does
            # not prove whether a physical write happened before the error.
            status = exc.code
            exc.close()
            raise DriverError(
                "HTTP_STATUS", f"Gateway returned HTTP {status}; command is never retried", uncertain=True
            ) from exc
        except OSError as exc:
            raise DriverError(
                "COMMUNICATION", "HTTP outcome unknown; command is never retried", uncertain=True
            ) from exc
        if status != 200 or len(raw) > 65536:
            raise DriverError("PROTOCOL", "Unexpected HTTP status or oversized response", uncertain=True)
        try:
            value = json.loads(raw, object_pairs_hook=_unique_json_keys)
        except ValueError as exc:
            raise DriverError("PROTOCOL", "Response is not JSON", uncertain=True) from exc
        if not isinstance(value, dict):
            raise DriverError("PROTOCOL", "HTTP response must be an object", uncertain=True)
        # Reject nonfinite JSON even when fields outside the assertions carry it.
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as exc:
            raise DriverError("PROTOCOL", "Nonfinite HTTP response", uncertain=True) from exc
        check_response(value, mapping["response"], parameters)
        return value, value

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
