# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""local_models:本机模型缓存目录识别与探测(纯逻辑,用 tmp 目录)。"""

from __future__ import annotations

from jiuwensymbiosis.gui import local_models


def _make_gdino(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"x")


def _make_sam2(path):
    _make_gdino(path)
    (path / "processor_config.json").write_text("{}", encoding="utf-8")


def test_looks_like_gdino_dir(tmp_path):
    good = tmp_path / "g"
    _make_gdino(good)
    assert local_models.looks_like_gdino_dir(good)
    assert not local_models.looks_like_gdino_dir(tmp_path / "missing")


def test_looks_like_sam2_needs_processor_config(tmp_path):
    only_gdino = tmp_path / "g"
    _make_gdino(only_gdino)
    assert not local_models.looks_like_sam2_dir(only_gdino)  # 缺 processor_config.json
    full = tmp_path / "s"
    _make_sam2(full)
    assert local_models.looks_like_sam2_dir(full)


def test_detect_local_model_scans_hf_and_modelscope(tmp_path, monkeypatch):
    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    snap = tmp_path / "hf" / "models--IDEA-Research--grounding-dino-base" / "snapshots" / "abc"
    _make_gdino(snap)
    found = local_models.detect_local_model(local_models.GDINO_REPO, local_models.looks_like_gdino_dir)
    assert found == snap


def test_detect_local_model_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(local_models, "HF_HUB", tmp_path / "hf")
    monkeypatch.setattr(local_models, "MODELSCOPE", tmp_path / "ms")
    assert local_models.detect_local_model(local_models.SAM2_REPO, local_models.looks_like_sam2_dir) is None
