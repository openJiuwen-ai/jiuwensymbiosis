# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.slot_pick.detect."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.tools.slot_pick.detect import (
    _coerce_float,
    _coerce_int,
    _coerce_bool,
    _coerce_pose,
    _coerce_optional_pose,
    _stop,
    _call_ok,
    _position_from_detection,
    _detect_object,
)


class TestCoerceFloat:
    def test_valid(self):
        assert _coerce_float(1.5, "x") == 1.5

    def test_string(self):
        assert _coerce_float("3.14", "x") == pytest.approx(3.14)

    def test_int(self):
        assert _coerce_float(3, "x") == 3.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="x"):
            _coerce_float("abc", "x")


class TestCoerceInt:
    def test_valid(self):
        assert _coerce_int(3, "n") == 3

    def test_string(self):
        assert _coerce_int("5", "n") == 5

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _coerce_int(0, "n")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _coerce_int(-1, "n")


class TestCoerceBool:
    def test_true_strings(self):
        for s in ("1", "true", "yes", "y", "on"):
            assert _coerce_bool(s, "f") is True

    def test_false_strings(self):
        for s in ("0", "false", "no", "n", "off"):
            assert _coerce_bool(s, "f") is False

    def test_bool_passthrough(self):
        assert _coerce_bool(True, "f") is True
        assert _coerce_bool(False, "f") is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="f"):
            _coerce_bool("maybe", "f")


class TestCoercePose:
    def test_valid(self):
        result = _coerce_pose([1, 2, 3, 4])
        assert result == (1.0, 2.0, 3.0, 4.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must be \\[x, y, z, r\\]"):
            _coerce_pose([1, 2, 3], "f")

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError):
            _coerce_pose("not_a_list", "f")


class TestCoerceOptionalPose:
    def test_none(self):
        assert _coerce_optional_pose(None, "f") is None

    def test_empty_string(self):
        assert _coerce_optional_pose("", "f") is None

    def test_valid(self):
        result = _coerce_optional_pose([1, 2, 3, 4], "f")
        assert result == (1.0, 2.0, 3.0, 4.0)


class TestStop:
    def test_structure(self):
        r = _stop("pick", "not_found", fallback_recommended=True)
        assert r["ok"] is False
        assert r["stage"] == "pick"
        assert r["reason"] == "not_found"
        assert r["fallback_recommended"] is True

    def test_extra_kwargs(self):
        r = _stop("x", "y", fallback_recommended=False, extra_key=42)
        assert r["extra_key"] == 42


class TestCallOk:
    def test_dict_ok_true(self):
        assert _call_ok({"ok": True}) is True

    def test_dict_ok_false(self):
        assert _call_ok({"ok": False}) is False

    def test_dict_missing_ok(self):
        assert _call_ok({}) is True

    def test_non_dict(self):
        assert _call_ok("not a dict") is True


class TestPositionFromDetection:
    def test_valid(self):
        result = _position_from_detection({"position": [100, 200, 300]}, stage="test")
        assert result == (100.0, 200.0, 300.0)

    def test_missing_position(self):
        result = _position_from_detection({}, stage="test")
        assert isinstance(result, dict)
        assert result["ok"] is False

    def test_short_position(self):
        result = _position_from_detection({"position": [100, 200]}, stage="test")
        assert isinstance(result, dict)
        assert result["ok"] is False


class TestDetectObject:
    def test_success(self, mock_api):
        result = _detect_object(mock_api, "box")
        assert result.get("ok") is True
        assert result.get("object") == "box"
        assert "selection_method" in result
