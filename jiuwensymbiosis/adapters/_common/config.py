# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Source-aware adapter configuration, shared by YAML, runtime and validation.

Factories declare ``path_fields`` for their own source-mapping keys, including
legacy nested layouts. No adapter-specific field names belong in this module.
The factory still owns defaults, validation and environment overrides.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

import yaml

__all__ = ["parse_config", "load_yaml_config"]

ConfigT = TypeVar("ConfigT", covariant=True)


class _ConfigParser(Protocol[ConfigT]):
    # fmt: off
    def from_dict(self, data: dict[str, Any]) -> ConfigT:
        ...
    # fmt: on


def parse_config(
    factory: _ConfigParser[ConfigT],
    data: Mapping[str, Any],
    *,
    source_dir: str | Path | None = None,
) -> tuple[dict[str, Any], ConfigT]:
    """Return isolated normalized source data and one parsed effective config.

    With a source directory, declared paths expand environment variables and
    ``~`` and resolve against that directory even when the target is absent.
    A source-less mapping keeps the existing ``from_dict`` semantics. Factories
    without path fields need no declaration. Parsing occurs exactly once; the
    factory cannot mutate the returned source mapping.
    """
    if not isinstance(data, Mapping):
        raise ValueError("adapter configuration must be a mapping")
    fields = getattr(factory, "path_fields", ())
    if not isinstance(fields, (tuple, list)) or any(not isinstance(key, str) or not key for key in fields):
        raise TypeError("config factory path_fields must be a sequence of non-empty field names")
    conditional_fields = getattr(factory, "path_or_id_fields", ())
    if not isinstance(conditional_fields, (tuple, list)) or any(
        not isinstance(key, str) or not key for key in conditional_fields
    ):
        raise TypeError("config factory path_or_id_fields must be a sequence of non-empty field names")
    directory = Path(source_dir).expanduser().resolve() if source_dir is not None else None

    def resolve(value: Any) -> Any:
        if directory is None or not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            return value
        path = Path(os.path.expandvars(os.fspath(value))).expanduser()
        return str((path if path.is_absolute() else directory / path).resolve())

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: resolve(item)
                if directory is not None
                and (
                    key in fields
                    or (
                        key in conditional_fields
                        and isinstance(item, str)
                        and item.startswith(("./", "../", "/", "~", "$"))
                    )
                )
                else visit(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return value

    normalized = visit(copy.deepcopy(dict(data)))
    return normalized, factory.from_dict(copy.deepcopy(normalized))


def load_yaml_config(factory: _ConfigParser[ConfigT], path: str | Path) -> ConfigT:
    """Load a YAML through the same source-aware parser as edited snapshots."""
    source = Path(path).expanduser().resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    return parse_config(factory, {} if data is None else data, source_dir=source.parent)[1]
