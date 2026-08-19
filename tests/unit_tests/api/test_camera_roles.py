# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which camera answers a question is decided by the FRAME, not by the camera's name.

The framework used to name two sensor roles — a "coarse" one that gave bearings and a
"precise" one that gave metric 3-D — and wire each path to a fixed camera. That is not a
property of the hardware: a body can have one RGBD camera and want both answers from it,
and a body can have two cameras where the first one asked cannot answer. What actually
decides is whether the grabbed frame carries depth. These tests pin that.
"""

from __future__ import annotations

from jiuwensymbiosis.perception.frame import CameraFrame
from tests.mocks.mock_dual_arm import MockDualArmApi, MockDualArmEnv, WorldBox

_CRATE = WorldBox("crate", (1400.0, 0.0, 300.0), (400.0, 500.0, 600.0))


class _TwoCameraEnv(MockDualArmEnv):
    """cruzr's shape — one camera without depth, one with — under deliberately neutral names."""

    cameras = ("nodepth", "rgbd")


def _api(env_cls=MockDualArmEnv):
    env = env_cls([_CRATE])
    env.connect()
    return MockDualArmApi(env)


def _strip_depth_from(api, camera_without_depth):
    """Make one named camera answer with an RGB-only frame, like a stereo pair read raw."""
    real, asked = api.env.grab_calibrated_frame, []

    def grab(camera=None):
        asked.append(camera)
        frame = real(camera)
        return CameraFrame(rgb=frame.rgb, depth_m=None) if camera == camera_without_depth else frame

    api._grab_calibrated_frame = grab
    return asked


class TestTheMetricPathPicksByDepth:
    def test_a_camera_without_depth_is_skipped_for_the_one_that_has_it(self):
        api = _api(_TwoCameraEnv)
        asked = _strip_depth_from(api, "nodepth")

        out = api.locate_for_grasp("crate")

        assert asked == ["nodepth", "rgbd"], "every camera should be asked, in order"
        assert out["ok"] is True, "the RGBD one could answer, so the measurement must succeed"

    def test_the_single_camera_default_still_works(self):
        """A body that never says which cameras it has has exactly one, and it answers."""
        api = _api()
        assert api.env.cameras == (None,)
        assert api.locate_for_grasp("crate")["ok"] is True

    def test_no_camera_carries_depth_says_so(self):
        """The reason names what was missing — 'no depth anywhere' is actionable,
        a blanket 'no camera' on a body that plainly has one is not."""
        api = _api(_TwoCameraEnv)
        _strip_depth_from(api, "nodepth")
        api._grab_calibrated_frame = lambda camera=None: CameraFrame(
            rgb=api.env.grab_calibrated_frame(camera).rgb, depth_m=None
        )

        out = api.locate_for_grasp("crate")

        assert out["ok"] is False
        assert out["reason"] == "no_depth"
