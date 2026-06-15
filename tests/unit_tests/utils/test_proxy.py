# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.utils.proxy."""

from __future__ import annotations

import os

from jiuwensymbiosis.utils.proxy import clear_proxy_env


class TestClearProxyEnv:
    def test_removes_proxy_vars(self):
        os.environ["HTTP_PROXY"] = "http://proxy:8080"
        os.environ["HTTPS_PROXY"] = "http://proxy:8080"
        os.environ["ALL_PROXY"] = "socks5://proxy:1080"
        result = clear_proxy_env()
        assert "HTTP_PROXY" in result
        assert "HTTPS_PROXY" in result
        assert "ALL_PROXY" in result
        assert "HTTP_PROXY" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        assert "ALL_PROXY" not in os.environ

    def test_sets_no_proxy_or_keeps_existing(self):
        # clear_proxy_env uses setdefault, so if NO_PROXY already set it stays
        clear_proxy_env()
        assert "NO_PROXY" in os.environ or "no_proxy" in os.environ

    def test_idempotent(self):
        clear_proxy_env()
        result = clear_proxy_env()
        assert result == {}

    def test_returns_popped(self):
        os.environ["HTTP_PROXY"] = "http://test:1234"
        result = clear_proxy_env()
        assert result.get("HTTP_PROXY") == "http://test:1234"

    def test_case_insensitive(self):
        os.environ["http_proxy"] = "http://lo:1"
        os.environ["https_proxy"] = "http://lo:2"
        os.environ["all_proxy"] = "http://lo:3"
        result = clear_proxy_env()
        # Both upper and lower case should be cleared
        cleared_lower = {k.lower() for k in result}
        assert "http_proxy" in cleared_lower or "HTTP_PROXY" in result
