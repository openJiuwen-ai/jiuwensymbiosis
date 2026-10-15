# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Home reporting and motion share one physical target across calibration modes."""

from __future__ import annotations

from math import sqrt

import pytest

from jiuwensymbiosis.adapters.piper import lowlevel
from jiuwensymbiosis.adapters.piper.api import PiperApi
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.piper.env import PiperEnv
from jiuwensymbiosis.errors import SafetyViolationError
from jiuwensymbiosis.rails.recovery import recover_session


@pytest.fixture
def home_warnings(monkeypatch):
    calls, warnings = [], []
    original = lowlevel.logger.warning

    def capture(message, *args, **kwargs):
        if "home target" in message:
            warnings.append((list(calls), message % args))
        original(message, *args, **kwargs)

    monkeypatch.setattr(lowlevel.logger, "warning", capture)
    return calls, warnings


@pytest.mark.parametrize(
    "calibration, home, bounds, reason",
    [
        ("none", [200, 20, 100, 180, 30, 0], {}, "TIP z="),
        ("anchorless", [200, 20, 100, 180, 30, 0], {}, "TIP z="),
        ("none", [701, 20, 300, 180, 30, 0], {}, "FLANGE x="),
        ("none", [10, 20, 300, 180, 30, 0], {}, "TIP x="),
        ("none", [200, 20, 801, 180, 30, 0], {}, "FLANGE z="),
        ("none", [200.0007, 20, 300, 0, 90, 0], {"tool_offset_mm": 0, "x_max_mm": 200.0008}, "FLANGE x="),
    ],
)
def test_static_home_limits_warn_before_can_without_blocking_connection(
    piper_factory, home_warnings, calibration, home, bounds, reason
):
    calls, warnings = home_warnings
    robot = piper_factory(calibration=calibration, home_pose_xyzrxryrz_mm_deg=home, sdk_calls=calls, **bounds)
    assert len(warnings) == 1
    calls_at_warning, message = warnings[0]
    assert calls_at_warning == []
    assert "configured home target" in message
    assert reason in message
    assert "automatic recovery homing will be rejected" in message
    assert calls[:4] == ["can_precheck", "construct", "connect", "enable"]
    assert robot.env.home_pose.as_tuple() == tuple(home)
    assert robot.env.get_observation().pose is not None
    with pytest.raises(SafetyViolationError, match=reason):
        robot.env.home()
    assert robot.arm.commands == []


@pytest.mark.parametrize(
    "config, source",
    [
        ({"home_use_init_pose": True, "initial_pose": (200, 20, 100, 180, 30, 0)}, "startup"),
        ({"calibration": "anchored", "home_lift_mm": -95}, "anchor-derived"),
    ],
)
def test_live_home_limits_warn_after_readback_without_blocking_connection(piper_factory, home_warnings, config, source):
    calls, warnings = home_warnings
    robot = piper_factory(sdk_calls=calls, **config)
    assert len(warnings) == 1
    calls_at_warning, message = warnings[0]
    assert calls_at_warning[-1] == "read_pose"
    assert calls_at_warning.count("read_pose") == 1
    assert f"{source} home target" in message
    assert "TIP z=" in message
    assert robot.env.get_observation().pose is not None
    assert recover_session(robot, release=False, home=True, log_prefix="test") == (False, False)
    assert robot.arm.commands == []


@pytest.mark.parametrize(
    "calibration, home_use_init_pose, expected_flange, expected_tip_xyz",
    [
        pytest.param("none", False, (220, 40, 350, 0, 90, 0), (320, 40, 350), id="config"),
        pytest.param("anchorless", False, (220, 40, 350, 0, 90, 0), (320, 40, 350), id="anchorless-config"),
        pytest.param("anchored", False, (210, 30, 300, 180, 30, 90), (210, -20, 300 - 50 * sqrt(3)), id="anchor"),
        pytest.param("none", True, (200, 20, 400, 180, 30, 90), (200, -30, 400 - 50 * sqrt(3)), id="startup"),
        pytest.param(
            "anchorless", True, (200, 20, 400, 180, 30, 90), (200, -30, 400 - 50 * sqrt(3)), id="anchorless-startup"
        ),
        pytest.param(
            "anchored", True, (200, 20, 400, 180, 30, 90), (200, -30, 400 - 50 * sqrt(3)), id="anchor-startup"
        ),
    ],
)
def test_home_tip_readback_preserves_physical_target(
    piper_factory, calibration, home_use_init_pose, expected_flange, expected_tip_xyz, home_warnings
):
    # Distinct positions AND orientations expose selection of the wrong source.
    robot = piper_factory(
        calibration=calibration,
        initial_pose=(200, 20, 400, 180, 30, 90),
        anchor=(210, 30, 100),
        tool_offset_mm=100,
        home_pose_xyzrxryrz_mm_deg=[220, 40, 350, 0, 90, 0],
        home_lift_mm=100,
        home_use_init_pose=home_use_init_pose,
    )
    api, env, arm = robot.api, robot.env, robot.arm
    assert home_warnings[1] == []
    expected_tip = dict(zip(("x", "y", "z", "rx", "ry", "rz"), (*expected_tip_xyz, *expected_flange[3:]), strict=True))

    # Home inspection uses the stored orientation, not the current one, without motion.
    arm.pose = (300000, 60000, 450000, 0, 0, 0)
    home = api.get_home_pose()
    assert home == pytest.approx({**expected_tip, "r": expected_flange[5]})
    assert arm.commands == []
    assert env.home_pose.as_tuple() == pytest.approx(expected_flange)

    api.home()
    assert tuple(value / 1000 for value in arm.commands[-1]) == pytest.approx(expected_flange)
    assert api.get_pose() == pytest.approx(expected_tip)
    assert api.get_home_pose() == pytest.approx(home)


def test_get_home_pose_before_connect_reports_lifecycle_error():
    api = PiperApi(PiperEnv(PiperConfig()))
    with pytest.raises(RuntimeError, match="env not connected"):
        api.get_home_pose()


def test_get_home_pose_after_disconnect_reports_lifecycle_error(piper_factory):
    robot = piper_factory()
    robot.env.disconnect()
    with pytest.raises(RuntimeError, match="env not connected"):
        robot.api.get_home_pose()
