# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.adapters._common.builder."""

from __future__ import annotations

import inspect

import yaml

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.env.mock import MockArmEnv
from tests.mocks.mock_api import MockApi


class _TestEnv(MockArmEnv):
    def __init__(self, cfg=None):
        super().__init__()


class _TestApi(MockApi):
    def __init__(self, env, **kwargs):
        super().__init__(env)


class TestMakeBuilder:
    def test_direct_call(self):
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        cfg = PiperConfig()
        session = builder(cfg)
        assert isinstance(session, RobotSession)
        assert session._connected is False

    def test_from_dict(self):
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        session = builder.from_dict({"can_port": "can_left"})
        assert isinstance(session, RobotSession)

    def test_from_yaml(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"can_port": "can_left"}), encoding="utf-8")
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        session = builder.from_yaml(p)
        assert isinstance(session, RobotSession)


class TestBuilderSignature:
    """A builder is a plain function carrying ``.from_yaml``/``.from_dict`` —
    not a class instance. This avoids both the ``@staticmethod __call__`` smell
    (an instance masquerading as a function) and the inverse linter warning
    (an instance method that ignores ``self``)."""

    def test_builder_is_a_function_not_a_class_instance(self):
        from types import FunctionType

        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        assert isinstance(builder, FunctionType)
        assert not isinstance(builder, type)

    def test_call_signature_inspectable(self):
        # A plain function's signature directly exposes ``cfg``.
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        sig = inspect.signature(builder)
        assert "cfg" in sig.parameters

    def test_from_yaml_and_from_dict_are_attributes(self):
        builder = make_builder(PiperConfig, _TestEnv, _TestApi)
        assert callable(getattr(builder, "from_yaml", None))
        assert callable(getattr(builder, "from_dict", None))


class _RecordingApi(MockApi):
    """Captures the kwargs it was constructed with."""

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.received_kwargs = kwargs


class _FlatConfig:
    """Minimal cfg with same-named fields and a renamed source field."""

    def __init__(self, a, b, deep_url):
        self.a = a
        self.b = b
        self.deep_url = deep_url
        self.name = "flat"


class TestApiKwargsFromCfgList:
    def test_same_name_passthrough(self):
        builder = make_builder(
            _FlatConfig,
            _TestEnv,
            _RecordingApi,
            api_kwargs_from_cfg=["a", "b"],
        )
        cfg = _FlatConfig(a=1, b="two", deep_url="x")
        session = builder(cfg)
        assert session.api.received_kwargs == {"a": 1, "b": "two"}

    def test_rename_mapping(self):
        # "cfg_attr:api_kwarg" maps cfg.deep_url → api_kwarg "url".
        builder = make_builder(
            _FlatConfig,
            _TestEnv,
            _RecordingApi,
            api_kwargs_from_cfg=["a", "deep_url:url"],
        )
        cfg = _FlatConfig(a=1, b="x", deep_url="http://d")
        session = builder(cfg)
        assert session.api.received_kwargs == {"a": 1, "url": "http://d"}

    def test_callable_form_still_works(self):
        # Backward compat: a function is still accepted.
        def _kw(cfg):
            return {"a": cfg.a, "b": cfg.b}

        builder = make_builder(
            _FlatConfig,
            _TestEnv,
            _RecordingApi,
            api_kwargs_from_cfg=_kw,
        )
        cfg = _FlatConfig(a=7, b="z", deep_url="x")
        session = builder(cfg)
        assert session.api.received_kwargs == {"a": 7, "b": "z"}


class TestMakeDetectorSidecar:
    def _cfg(self, spawn):
        from types import SimpleNamespace

        det = SimpleNamespace(
            spawn=spawn,
            host="127.0.0.1",
            port=8114,
            device="cpu",
            startup_timeout_s=1.0,
            gdino_model_id="m1",
            sam2_model_id="m2",
            box_threshold=0.3,
            text_threshold=0.2,
            use_sam2=True,
            url="http://127.0.0.1:8114",
        )
        return SimpleNamespace(detector=det, name="t")

    def test_returns_none_when_not_spawning(self, monkeypatch):
        from jiuwensymbiosis.adapters._common.builder import make_detector_sidecar

        builder = make_detector_sidecar()
        cfg = self._cfg(spawn=False)
        assert builder(cfg) is None

    def test_returns_callable_factory_when_spawning(self, monkeypatch):
        # Don't actually start a subprocess: stub detector_subprocess.
        import jiuwensymbiosis.adapters._common.detector_sidecar as ds_mod

        called = {}

        class _FakeCM:
            def __enter__(self):
                called["entered"] = True
                return self

            def __exit__(self, *a):
                called["exited"] = True

        def _fake(**kwargs):
            called["kwargs"] = kwargs
            return _FakeCM()

        monkeypatch.setattr(ds_mod, "detector_subprocess", _fake)

        from jiuwensymbiosis.adapters._common.builder import make_detector_sidecar

        builder = make_detector_sidecar()
        cfg = self._cfg(spawn=True)
        starter = builder(cfg)
        assert callable(starter)
        cm = starter()
        with cm:
            pass
        assert called["kwargs"]["host"] == "127.0.0.1"
        assert called["kwargs"]["port"] == 8114
        assert called["entered"] is True
        assert called["exited"] is True
