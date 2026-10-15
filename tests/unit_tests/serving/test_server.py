# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.serving.grounding_dino_sam2_server — schema and endpoint tests."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from jiuwensymbiosis.serving.grounding_dino_sam2_server import (
        _box_to_mask,
        _normalize_prompt,
        app,
    )

    HAS_SERVER_DEPS = True
except ImportError:
    HAS_SERVER_DEPS = False


skip_no_server_deps = pytest.mark.skipif(not HAS_SERVER_DEPS, reason="server dependencies not installed")


@skip_no_server_deps
class TestNormalizePrompt:
    def test_lowercase(self):
        assert _normalize_prompt("A Box") == "a box ."

    def test_adds_period(self):
        assert _normalize_prompt("box") == "box ."

    def test_already_has_period(self):
        result = _normalize_prompt("a box.")
        assert result.endswith(".")


@skip_no_server_deps
class TestBoxToMask:
    def test_basic(self):
        mask = _box_to_mask(np.array([10.0, 20.0, 50.0, 60.0]), 100, 100)
        assert mask.shape == (100, 100)
        assert mask[20:60, 10:50].any()


@skip_no_server_deps
class TestEndpoints:
    @pytest.mark.asyncio
    async def test_health(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/health", headers={"X-Request-ID": "health-check"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["schema_version"] == 1
            assert data["request_id"] == "health-check"
            assert data["result"]["service"] == "vision"
            assert data["result"]["status"] in {"ready", "loading"}

    @pytest.mark.asyncio
    async def test_segment_missing_fields(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/v1/segment", json={})
            assert resp.status_code == 422
