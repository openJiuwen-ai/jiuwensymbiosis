# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Derived from the instrument integration; licensed under Apache-2.0.

"""Per-service instrument extensions: data contracts and bounded protocol mappings."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .contracts import ensure_json_value

_RESERVED = frozenset(
    {
        "get_capabilities",
        "list_devices",
        "get_snapshot",
        "probe_device",
        "get_task",
        "confirm_task",
        "cancel_task",
        "emergency_stop_task",
        "get_feedback",
        "execution_status",
        "invoke",
    }
)
_NAME = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "minimum",
    "maximum",
    "multipleOf",
    "enum",
    "const",
    "description",
    "title",
    "default",
    "minLength",
    "maxLength",
}


def validate_parameter_schema(schema: dict) -> None:
    """Accept flat closed schemas with documented numeric limits when provided."""
    ensure_json_value(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"Invalid parameter schema: {exc.message}") from exc
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("parameters_schema must be a closed object")
    if set(schema) - _SCHEMA_KEYS:
        raise ValueError("unsupported parameter schema keyword; references and expressions are not supported")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or len(properties) > 50:
        raise ValueError("parameters_schema must have at most 50 properties")
    required = schema.get("required", [])
    if not isinstance(required, list) or set(required) - properties.keys() or len(set(required)) != len(required):
        raise ValueError("required must contain unique declared parameter names")
    for name, item in properties.items():
        if not _NAME.fullmatch(name) or not isinstance(item, dict) or set(item) - _SCHEMA_KEYS:
            raise ValueError(f"unsupported parameter declaration: {name}")
        kind = item.get("type")
        if kind not in {"integer", "number", "string", "boolean"}:
            raise ValueError(f"parameter {name} must have a scalar type")
        if kind in {"integer", "number"} and "minimum" not in item:
            raise ValueError(f"parameter {name} lacks a documented numeric minimum")
        if kind in {"integer", "number"} and "maximum" in item and item["minimum"] > item["maximum"]:
            raise ValueError(f"parameter {name} has inverted bounds")
        if kind == "string" and not item.get("enum"):
            raise ValueError(f"parameter {name} requires a finite documented string enum")
        if "default" in item:
            try:
                Draft202012Validator(item).validate(item["default"])
            except ValidationError as exc:
                raise ValueError(f"Invalid parameter default: {exc.message}") from exc


def validate_output_schema(schema: dict[str, Any]) -> None:
    """Validate JSON result declarations without granting external resolution."""
    ensure_json_value(schema)
    if not isinstance(schema, dict) or len(json.dumps(schema)) > 200_000:
        raise ValueError("output_schema must be a bounded schema object")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$id"} and (not isinstance(item, str) or not item.startswith("#")):
                    raise ValueError("instrument schemas cannot resolve external URLs or files")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)
    Draft202012Validator.check_schema(schema)


def _substitutions(value: Any) -> set[str]:
    if isinstance(value, dict):
        if "$parameter" in value:
            if set(value) != {"$parameter"} or not isinstance(value["$parameter"], str):
                raise ValueError("parameter substitution must contain only $parameter")
            return {value["$parameter"]}
        return set().union(*(_substitutions(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_substitutions(v) for v in value)) if value else set()
    return set()


def validate_mapping(mapping: dict, parameters_schema: dict, *, read_only: bool) -> None:
    """Reject unsupported transports, dynamic endpoints and missing response rules."""
    if not isinstance(mapping, dict):
        raise ValueError("each extension action requires an executable mapping")
    transport = mapping.get("transport")
    parameters = parameters_schema.get("properties", {})
    used = set()
    if transport == "serial_line":
        if any(p["type"] in {"integer", "number"} and "maximum" not in p for p in parameters.values()):
            raise ValueError("serial numeric parameters require documented numeric bounds")
        if set(mapping) - {"transport", "request", "response", "line_ending"}:
            raise ValueError("unknown serial mapping fields")
        if mapping.get("line_ending") not in {"LF", "CRLF", "NONE"}:
            raise ValueError("serial mapping requires an explicit line_ending")
        parts = mapping.get("request")
        if not isinstance(parts, list) or not parts or len(parts) > 50:
            raise ValueError("serial request must be a nonempty bounded segment list")
        for part in parts:
            if isinstance(part, str):
                invalid_length = not part or len(part) > 256
                if invalid_length or any(c in part for c in "\r\n\x00") or not part.isascii():
                    raise ValueError("serial literal must be short single-line ASCII")
            elif isinstance(part, dict) and set(part) <= {"parameter", "format"} and "parameter" in part:
                name = part["parameter"]
                if name not in parameters or part.get("format", "str") not in {"str", "03d", "d"}:
                    raise ValueError("unknown serial parameter or format")
                if part.get("format") in {"03d", "d"} and parameters[name]["type"] != "integer":
                    raise ValueError("integer wire format requires an integer parameter")
                if parameters[name]["type"] == "string" and any(
                    any(c in text for c in "\r\n\x00") or not text.isascii() for text in parameters[name]["enum"]
                ):
                    raise ValueError("serial parameter enum must be single-line ASCII")
                used.add(name)
            else:
                raise ValueError("unsupported serial request segment")
        response = mapping.get("response", {})
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        if set(response) == {"contains"}:
            if not isinstance(response["contains"], str) or not response["contains"].strip():
                raise ValueError("response contains must be a nonempty string")
        elif set(response) == {"number_prefix", "field"}:
            if not isinstance(response["number_prefix"], str) or not response["number_prefix"]:
                raise ValueError("number_prefix is required")
            if not isinstance(response["field"], str) or not _NAME.fullmatch(response["field"]):
                raise ValueError("numeric response field must be an identifier")
        else:
            raise ValueError("serial mapping requires contains or number_prefix/field response evidence")
    elif transport == "http_json":
        if set(mapping) - {"transport", "method", "path", "body", "response", "unwrap", "guards", "completion"}:
            raise ValueError("unknown HTTP mapping fields")
        if mapping.get("method") not in {"GET", "POST"} or (read_only and mapping["method"] != "GET"):
            raise ValueError("HTTP mapping requires GET/POST; read-only mappings require GET")
        from jiuwensymbiosis.adapters.instrument.direct.mapping_rules import validate_http_mapping

        used = validate_http_mapping(mapping, parameters, read_only=read_only)

    else:
        raise ValueError(f"unsupported protocol transport: {transport!r}")
    if used != set(parameters):
        raise ValueError("each parameter must be explicitly bound exactly by the protocol mapping")
    if set(parameters) != set(parameters_schema.get("required", [])):
        raise ValueError("mapped parameters must be required; defaults are recommendations only")


@dataclass(frozen=True)
class InstrumentOperation:
    """Read-only view of a validated instrument declaration; no execution state."""

    version: str
    _definition_json: str

    @classmethod
    def from_definition(cls, definition: dict[str, Any], version: str) -> InstrumentOperation:
        return cls(version, json.dumps(definition, sort_keys=True, allow_nan=False))

    @property
    def definition(self) -> dict[str, Any]:
        return json.loads(self._definition_json)

    @property
    def operation_id(self) -> str:
        return self.definition["name"]

    @property
    def description(self) -> str:
        return self.definition["description"]

    @property
    def request_schema(self) -> dict[str, Any]:
        return self.definition["parameters_schema"]

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.definition.get("output_schema", {"type": "object"})

    @property
    def required_conditions(self) -> tuple[str, ...]:
        if self.definition["read_only"]:
            return ()
        return tuple(self.definition.get("allowed_device_states", ["idle"]))

    @property
    def completion_rule(self) -> str:
        return self.definition["completion_rule"]


class ActionRegistry:
    """An explicit installation catalog; loading one cannot alter another."""

    def __init__(self, extensions: list[dict] | None = None):
        ensure_json_value(extensions)
        self.operations: dict[str, InstrumentOperation] = {}
        self.definitions: dict[str, dict] = {}
        self.documented_commands: dict[str, list[dict]] = {}
        self.action_extensions: dict[str, str] = {}
        read_only: set[str] = set()
        seen_extensions = set()
        for extension in extensions or []:
            if (
                set(extension) - {"extension_id", "version", "actions", "coverage", "source_text"}
                or not {"extension_id", "version", "actions"} <= extension.keys()
            ):
                raise ValueError("extension requires extension_id/version/actions and optional coverage")
            identifier = extension["extension_id"]
            if not isinstance(identifier, str) or not _NAME.fullmatch(identifier) or identifier in seen_extensions:
                raise ValueError("extension_id must be a unique lowercase identifier")
            seen_extensions.add(identifier)
            if not isinstance(extension["version"], str) or not extension["version"].strip():
                raise ValueError("extension version is required")
            actions = extension["actions"]
            if not isinstance(actions, list) or not 1 <= len(actions) <= 100:
                raise ValueError("extension must declare 1..100 actions")
            for definition in actions:
                required = {
                    "name",
                    "description",
                    "parameters_schema",
                    "read_only",
                    "source",
                    "completion_rule",
                    "mapping",
                }
                if (
                    not isinstance(definition, dict)
                    or not required <= set(definition)
                    or set(definition) - required - {"allowed_device_states", "output_schema"}
                ):
                    raise ValueError("action definition has missing or unknown fields")
                states = definition.get("allowed_device_states", ["idle"])
                if states not in (["idle"], ["idle", "busy"]):
                    raise ValueError("allowed_device_states must be idle or idle/busy")
                if definition["read_only"] and "allowed_device_states" in definition:
                    raise ValueError("read-only actions do not need allowed_device_states")
                name = definition["name"]
                if not isinstance(name, str) or not _NAME.fullmatch(name):
                    raise ValueError(f"invalid or duplicate action name: {name!r}")
                if name in self.operations or name in _RESERVED:
                    raise ValueError(f"invalid or duplicate action name: {name!r}")
                if not isinstance(definition["read_only"], bool):
                    raise ValueError("read_only must be a boolean")
                if definition["completion_rule"] != "protocol_response_verified":
                    raise ValueError("mapping v1 supports verified protocol responses, not physical-cycle completion")
                source = definition["source"]
                if not isinstance(source, dict) or set(source) != {"document", "section", "sha256", "excerpt"}:
                    raise ValueError("source requires document/section/sha256/excerpt")
                if any(not isinstance(v, str) or not v.strip() for v in source.values()) or not re.fullmatch(
                    r"[0-9a-f]{64}", source["sha256"]
                ):
                    raise ValueError("source evidence fields and SHA-256 are required")
                schema = definition["parameters_schema"]
                validate_parameter_schema(schema)
                validate_mapping(definition["mapping"], schema, read_only=definition["read_only"])
                if "output_schema" in definition:
                    validate_output_schema(definition["output_schema"])
                self.operations[name] = InstrumentOperation.from_definition(definition, extension["version"])
                self.definitions[name] = deepcopy(definition)
                self.action_extensions[name] = identifier
                if definition["read_only"]:
                    read_only.add(name)
            source_text = extension.get("source_text")
            if source_text is not None:
                if not isinstance(source_text, str) or not source_text or len(source_text.encode()) > 5_000_000:
                    raise ValueError("extension source_text must be bounded UTF-8 text")
                digest = hashlib.sha256(source_text.encode()).hexdigest()
                for definition in actions:
                    source = definition["source"]
                    if (
                        source["sha256"] != digest
                        or source["excerpt"] not in source_text
                        or source["section"] not in source_text
                    ):
                        raise ValueError("embedded extension source does not match its action provenance")
            coverage = extension.get("coverage", [])
            if not isinstance(coverage, list):
                raise ValueError("extension coverage must be an array")
            for row in coverage:
                if not isinstance(row, dict) or set(row) != {"command", "action", "status", "evidence"}:
                    raise ValueError("coverage requires command/action/status/evidence")
                if any(not isinstance(row[key], str) or not row[key].strip() for key in ("command", "evidence")):
                    raise ValueError("coverage command and evidence must be nonempty")
                if row["status"] == "implemented":
                    if self.action_extensions.get(row["action"]) != identifier:
                        raise ValueError("coverage refers to an action outside its extension")
                elif row["status"] != "blocked" or row["action"] is not None:
                    raise ValueError("blocked coverage requires action=null")
            self.documented_commands[identifier] = deepcopy(coverage)
        self.read_only = frozenset(read_only)

    def validate_parameters(self, action: str, parameters: dict) -> None:
        """Validate extension parameters without coercion or implicit defaults."""
        ensure_json_value(parameters)
        schema = self.definitions[action]["parameters_schema"]
        try:
            Draft202012Validator(schema).validate(parameters)
        except ValidationError as exc:
            raise ValueError(f"Invalid {action} parameters: {exc.message}") from exc
        for name, value in parameters.items():
            if schema["properties"][name]["type"] == "integer" and type(value) is not int:
                raise ValueError(f"{name} must be an integer, not a float or boolean")
