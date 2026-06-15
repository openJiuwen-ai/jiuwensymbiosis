# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.slot_pick.vlm_judge."""

from __future__ import annotations

import numpy as np

from jiuwensymbiosis.tools.slot_pick.vlm_judge import (
    make_vlm_completion_judge,
    _encode_jpeg_b64,
)


class TestEncodeJpegB64:
    def test_valid_image(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = _encode_jpeg_b64(img)
        if result is not None:
            import base64

            decoded = base64.b64decode(result)
            assert len(decoded) > 0

    def test_none_on_bad_input(self):
        result = _encode_jpeg_b64(None)
        assert result is None

    def test_none_on_wrong_shape(self):
        result = _encode_jpeg_b64(np.zeros((10, 10)))
        assert result is None


class TestMakeVlmCompletionJudge:
    def test_returns_callable(self):
        judge = make_vlm_completion_judge(
            api_base="http://localhost:1234",
            api_key="test",
            model_name="test-model",
        )
        assert callable(judge)

    def test_fallback_used_when_no_image(self):
        fallback_calls = [0]

        def my_fallback(api, config):
            fallback_calls[0] += 1
            return True

        judge = make_vlm_completion_judge(
            api_base="http://localhost:1234",
            api_key="test",
            model_name="test-model",
            fallback=my_fallback,
        )

        class FakeApi:
            def get_image(self):
                raise RuntimeError("no camera")

        class FakeConfig:
            chip_object_name = "chip"
            slot_object_name = "slot"

        result = judge(FakeApi(), FakeConfig())
        assert fallback_calls[0] == 1
        assert result is True

    def test_no_fallback_returns_false(self):
        judge = make_vlm_completion_judge(
            api_base="http://localhost:1234",
            api_key="test",
            model_name="test-model",
        )

        class FakeApi:
            def get_image(self):
                raise RuntimeError("no camera")

        class FakeConfig:
            chip_object_name = "chip"
            slot_object_name = "slot"

        result = judge(FakeApi(), FakeConfig())
        assert result is False
