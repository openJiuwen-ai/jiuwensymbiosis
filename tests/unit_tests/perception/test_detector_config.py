# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Deployment choices cannot accidentally start or load local models."""

import pytest

from jiuwensymbiosis.perception.config import DetectorServerConfig, parse_detector_config


def test_missing_detector_is_disabled():
    assert parse_detector_config({}).mode == "disabled"


@pytest.mark.parametrize("mode", ["local", "remote"])
@pytest.mark.parametrize(
    "settings,expected", [({}, None), ({"max_frame_age_s": None}, None), ({"max_frame_age_s": 10}, 10)]
)
def test_frame_age_limit_is_optional(mode, settings, expected):
    raw = {"mode": mode, **settings}
    if mode == "remote":
        raw["endpoint"] = {"url": "http://vision"}
    assert parse_detector_config({"detector": raw}).max_frame_age_s == expected


@pytest.mark.parametrize("spawn", [False, True])
def test_legacy_frame_age_limit_defaults_off_and_can_be_selected(spawn):
    raw = {"_target_": "gdino", "spawn": spawn}
    assert parse_detector_config({"api_servers": [raw]}).max_frame_age_s is None
    assert DetectorServerConfig(spawn=spawn).max_frame_age_s is None
    assert parse_detector_config({"api_servers": [{**raw, "max_frame_age_s": 10}]}).max_frame_age_s == 10
    assert DetectorServerConfig(spawn=spawn, max_frame_age_s=10).max_frame_age_s == 10


def test_remote_ignores_local_model_environment(monkeypatch):
    monkeypatch.setenv("GDINO_MODEL_ID", "/missing/local/model")
    config = parse_detector_config({"detector": {"mode": "remote", "endpoint": {"url": "http://gpu:8114"}}})
    assert config.local is None
    assert not config.spawn
    assert config.url == "http://gpu:8114"


def test_nested_legacy_explicit_no_spawn_is_remote():
    config = parse_detector_config(
        {"env": {"cfg": {"api_servers": [{"_target_": "gdino", "host": "gpu", "port": 8114, "spawn": False}]}}}
    )
    assert config.mode == "remote"
    assert config.url == "http://gpu:8114"


@pytest.mark.parametrize(
    "detector",
    [
        {"mode": "remote", "endpoint": {"url": "http://gpu"}, "device": "cuda"},
        {"mode": "remote", "endpoint": {"url": "http://gpu"}, "local": {}},
        {"mode": "local", "endpoint": {"url": "http://gpu"}},
        {"mode": "local", "local": {"spwan": True}},
        {"mode": "local", "local": {"use_sam2": "false"}},
        {"mode": "local", "local": {"port": True}},
        {"mode": "disabled", "max_frame_age_s": float("inf")},
        {"mode": "disabled", "max_frame_age_s": 0},
        {"mode": "disabled", "max_frame_age_s": -1},
        {"mode": "disabled", "max_frame_age_s": float("nan")},
        {"mode": "disabled", "max_frame_age_s": True},
        {"mode": "disabled", "max_frame_age_s": "10"},
    ],
)
def test_invalid_deployment_fails_at_parse(detector):
    with pytest.raises(ValueError):
        parse_detector_config({"detector": detector})


def test_old_new_detector_conflict():
    with pytest.raises(ValueError, match="conflict"):
        parse_detector_config({"detector": {"mode": "disabled"}, "api_servers": [{"_target_": "gdino"}]})


def test_python_remote_compatibility_does_not_validate_local_bind():
    config = DetectorServerConfig(host="gpu", port=8114, spawn=False)
    assert config.url == "http://gpu:8114"
    assert config.local is None
