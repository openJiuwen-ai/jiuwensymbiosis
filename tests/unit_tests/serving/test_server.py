# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.serving.grounding_dino_sam2_server — schema and endpoint tests."""

from __future__ import annotations

import base64

import numpy as np
import pytest

try:
    from jiuwensymbiosis.serving.grounding_dino_sam2_server import (
        SegmentRequest,
        _box_to_mask,
        _encode_mask,
        _normalize_prompt,
        app,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@skip_no_torch
class TestSchemas:
    def test_segment_request(self):
        req = SegmentRequest(image_base64="abc", text_prompt="a box")
        assert req.image_base64 == "abc"
        assert req.text_prompt == "a box"


@skip_no_torch
class TestNormalizePrompt:
    def test_lowercase(self):
        assert _normalize_prompt("A Box") == "a box ."

    def test_adds_period(self):
        assert _normalize_prompt("box") == "box ."

    def test_already_has_period(self):
        result = _normalize_prompt("a box.")
        assert result.endswith(".")


@skip_no_torch
class TestEncodeMask:
    def test_bool_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:7] = True
        result = _encode_mask(mask)
        decoded = base64.b64decode(result)
        assert len(decoded) == 10 * 10


@skip_no_torch
class TestBoxToMask:
    def test_basic(self):
        mask = _box_to_mask(np.array([10.0, 20.0, 50.0, 60.0]), 100, 100)
        assert mask.shape == (100, 100)
        assert mask[20:60, 10:50].any()


@skip_no_torch
class TestEndpoints:
    @pytest.mark.asyncio
    async def test_health(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data

    @pytest.mark.asyncio
    async def test_segment_missing_fields(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/segment", json={})
            assert resp.status_code == 422
