# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""Bounded declarative JSON selection and assertions. No scripts or remote schemas."""

from __future__ import annotations

import re

from jiuwensymbiosis.adapters.instrument.direct.errors import DriverError

_MISSING = object()
_NAME = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")


def render(value, parameters):
    if isinstance(value, dict):
        if set(value) == {"$parameter"}:
            return parameters[value["$parameter"]]
        return {k: render(v, parameters) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, parameters) for v in value]
    return value


def substitutions(value):
    if isinstance(value, dict):
        if "$parameter" in value:
            if set(value) != {"$parameter"} or not isinstance(value["$parameter"], str):
                raise ValueError("invalid parameter substitution")
            return {value["$parameter"]}
        return set().union(*(substitutions(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(substitutions(v) for v in value)) if value else set()
    return set()


def validate_path(path):
    if not isinstance(path, list) or not 1 <= len(path) <= 16:
        raise ValueError("JSON path must contain 1..16 literal keys, indices or unique selectors")
    for part in path:
        if isinstance(part, str) and part and len(part) <= 128:
            continue
        if type(part) is int and 0 <= part <= 10000:
            continue
        if isinstance(part, dict) and set(part) == {"field", "equals"} and isinstance(part["field"], str):
            substitutions(part["equals"])
            continue
        raise ValueError("invalid JSON path segment")


def select(value, path, parameters=None):
    for part in path:
        if isinstance(part, dict):
            expected = render(part["equals"], parameters or {})
            selected = (
                [v for v in value if isinstance(v, dict) and exact(v.get(part["field"], _MISSING), expected)]
                if isinstance(value, list)
                else []
            )
            value = selected[0] if len(selected) == 1 else _MISSING
        elif isinstance(value, dict) and isinstance(part, str):
            value = value.get(part, _MISSING)
        elif isinstance(value, list) and type(part) is int and part < len(value):
            value = value[part]
        else:
            return _MISSING
    return value


def exact(value, expected):
    return type(value) is type(expected) and value == expected


def validate_checks(checks, *, allow_empty=False):
    if not isinstance(checks, list) or len(checks) > 50:
        raise ValueError("checks must be a bounded nonempty array")
    if not checks and not allow_empty:
        raise ValueError("checks must be a bounded nonempty array")
    for rule in checks:
        if not isinstance(rule, dict) or "path" not in rule or len(rule) != 2:
            raise ValueError("each check requires path and exactly one assertion")
        validate_path(rule["path"])
        if "equals" in rule:
            substitutions(rule["equals"])
        elif "one_of" in rule and isinstance(rule["one_of"], list) and 1 <= len(rule["one_of"]) <= 50:
            pass
        elif rule.get("type") in {"string", "number", "integer", "boolean", "object", "array", "null"}:
            pass
        else:
            raise ValueError("unsupported assertion")


def matches(value, checks, parameters=None):
    for rule in checks:
        item = select(value, rule["path"], parameters)
        if item is _MISSING:
            return False
        if "equals" in rule and not exact(item, render(rule["equals"], parameters or {})):
            return False
        if "one_of" in rule and not any(exact(item, v) for v in rule["one_of"]):
            return False
        if "type" in rule:
            kinds = {
                "string": (str,),
                "number": (int, float),
                "integer": (int,),
                "boolean": (bool,),
                "object": (dict,),
                "array": (list,),
                "null": (type(None),),
            }
            if type(item) not in kinds.get(rule["type"], ()):
                return False
    return True


def check_response(value, rule, parameters=None):
    checks = rule.get("checks", [{"path": [rule.get("field")], "equals": rule.get("equals")}])
    if not matches(value, checks, parameters):
        raise DriverError("PROTOCOL", "Documented response assertion did not match", uncertain=True)


def validate_http_mapping(mapping, parameters, *, read_only):
    path = mapping.get("path")
    used = substitutions(mapping.get("body"))
    if isinstance(path, list):
        if not path or len(path) > 30:
            raise ValueError("HTTP path segment list is empty or oversized")
        static = ""
        for part in path:
            if isinstance(part, str):
                static += part
            elif isinstance(part, dict) and set(part) == {"$parameter"}:
                name = part["$parameter"]
                if name not in parameters:
                    raise ValueError("HTTP path parameters require bounded nonnegative integers")
                parameter = parameters[name]
                if parameter["type"] != "integer" or parameter["minimum"] < 0 or "maximum" not in parameter:
                    raise ValueError("HTTP path parameters require bounded nonnegative integers")
                used.add(name)
                static += "0"
            else:
                raise ValueError("invalid HTTP path segment")
        path = static
    if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
        raise ValueError("HTTP path must be a fixed absolute path or bounded integer segments")
    if any(c in path for c in "\r\n?#{}\\") or "/../" in path or any(ord(c) < 32 for c in path):
        raise ValueError("HTTP path must be a fixed absolute path or bounded integer segments")
    if mapping["method"] == "GET" and mapping.get("body") not in ({}, None):
        raise ValueError("GET body is unsupported")
    if mapping.get("unwrap", "none") != "none":
        raise ValueError("MCP envelopes require the instrument access: mcp SDK adapter")
    response = mapping.get("response")
    if isinstance(response, dict) and set(response) == {"field", "equals"}:
        if not isinstance(response["field"], str) or not _NAME.fullmatch(response["field"]):
            raise ValueError("HTTP response field must be a top-level identifier")
    elif isinstance(response, dict) and set(response) == {"checks"}:
        validate_checks(response["checks"], allow_empty="completion" in mapping and not read_only)
    else:
        raise ValueError("HTTP response requires explicit field/equals or checks")
    guards = mapping.get("guards", [])
    if not isinstance(guards, list) or len(guards) > 10:
        raise ValueError("guards must be a bounded array")
    for guard in guards:
        validate_observation(guard)
    completion = mapping.get("completion")
    if completion is not None:
        if (
            read_only
            or not isinstance(completion, dict)
            or set(completion) - {"observe", "checks", "event", "changed", "additional", "hold_s"}
        ):
            raise ValueError("invalid completion mapping")
        validate_observation({k: completion[k] for k in ("observe", "checks") if k in completion})
        if ("event" in completion) == ("changed" in completion):
            raise ValueError("completion requires exactly one freshness rule: event or changed")
        if "event" in completion:
            event = completion["event"]
            if not isinstance(event, dict) or set(event) != {"path", "match", "success"}:
                raise ValueError("event requires path/match/success")
            validate_path(event["path"])
            validate_checks(event["match"])
            validate_checks(event["success"])
        else:
            validate_path(completion["changed"])
        additional = completion.get("additional", [])
        if not isinstance(additional, list) or len(additional) > 10:
            raise ValueError("invalid additional observations")
        for entry in additional:
            validate_observation(entry)
        if "hold_s" in completion:
            hold = completion["hold_s"]
            if isinstance(hold, dict) and set(hold) == {"$parameter"}:
                name = hold["$parameter"]
                spec = parameters.get(name, {})
                if (
                    spec.get("type") not in {"integer", "number"}
                    or not 0 < spec.get("minimum", 0) <= spec.get("maximum", 0) <= 86400
                ):
                    raise ValueError("hold_s requires a bounded positive duration parameter")
                used.add(name)
            elif type(hold) not in {int, float} or not 0 < hold <= 86400:
                raise ValueError("hold_s must be a positive duration up to 86400 seconds")
    if substitutions(mapping) - set(parameters):
        raise ValueError("mapping references undeclared parameters")
    return used


def validate_observation(value):
    if not isinstance(value, dict) or set(value) != {"observe", "checks"} or not isinstance(value["observe"], str):
        raise ValueError("observation requires observe action and checks")
    validate_checks(value["checks"])


def validate_profile(device, registry):
    protocol = device.get("protocol", {})
    if set(protocol) - {"status_action", "state_path", "state_values", "idle_checks", "volatile_paths"}:
        raise ValueError("unsupported mapped device protocol settings")
    if protocol:
        if not {"status_action", "state_path", "state_values", "idle_checks"} <= set(protocol):
            raise ValueError("mapped observation requires status action, state path/values and idle checks")
        validate_path(protocol["state_path"])
        values = protocol["state_values"]
        if (
            not isinstance(values, dict)
            or not values
            or any(v not in {"idle", "busy", "running", "fault", "offline", "unknown"} for v in values.values())
        ):
            raise ValueError("invalid state mapping")
        validate_checks(protocol["idle_checks"])
        volatile = protocol.get("volatile_paths", [])
        if not isinstance(volatile, list) or len(volatile) > 32:
            raise ValueError("volatile_paths must contain at most 32 metadata paths")
        protected = [protocol["state_path"], *(check["path"] for check in protocol["idle_checks"])]
        for path in volatile:
            validate_path(path)
            if any(not isinstance(segment, str) for segment in path):
                raise ValueError("volatile metadata paths require literal object keys")
            if any(path[: len(other)] == other or other[: len(path)] == path for other in protected):
                raise ValueError("volatile metadata cannot hide state or idle conditions")
        if substitutions(protocol):
            raise ValueError("snapshot cannot require runtime parameters")
    names = device.get("operation_ids") or [device["operation_id"]]
    references = [protocol["status_action"]] if protocol else []
    for name in names:
        mapping = registry.definitions[name]["mapping"]
        references.extend(g["observe"] for g in mapping.get("guards", []))
        completion = mapping.get("completion")
        if completion:
            references.append(completion["observe"])
            references.extend(g["observe"] for g in completion.get("additional", []))
    for name in references:
        if name not in names:
            raise ValueError("observation action must belong to the same device")
        definition = registry.definitions[name]
        mapping = definition["mapping"]
        invalid_observation = not definition["read_only"] or definition["parameters_schema"].get("properties")
        if invalid_observation or mapping["transport"] not in {"http_json", "serial_line"} or mapping.get("guards"):
            raise ValueError("observations require parameter-free read-only mapped actions without guards")
