# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration test stubs — skipped by default, require real hardware/GPU."""

import pytest

pytestmark = pytest.mark.integration


class TestPiperLowLevelIntegration:
    """Requires piper_sdk + real CAN-connected Piper arm."""

    def test_can_connect(self):
        pytest.skip("Requires real Piper hardware")

    def test_move_to_pose_blocking(self):
        pytest.skip("Requires real Piper hardware")

    def test_joint_motion(self):
        pytest.skip("Requires real Piper hardware")


class TestRealSenseCameraIntegration:
    """Requires pyrealsense2 + physically connected RealSense camera."""

    def test_grab_frames(self):
        pytest.skip("Requires RealSense camera")


class TestGroundingDinoSam2Integration:
    """Requires CUDA GPU + model weights."""

    def test_segment_endpoint(self):
        pytest.skip("Requires CUDA GPU + model weights")
