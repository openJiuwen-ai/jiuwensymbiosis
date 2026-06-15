# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Test wrapper around the real MockArmEnv — adds test helpers."""

from jiuwensymbiosis.env.mock import MockArmEnv


class MockArmEnvWrapper(MockArmEnv):
    """Thin wrapper adding assertion helpers on top of MockArmEnv."""

    def assert_connected(self) -> None:
        assert self._connected, "MockArmEnv is not connected"

    def assert_suction(self, expected: bool) -> None:
        assert self._suction == expected, f"suction={self._suction}, expected={expected}"

    def assert_pose_approx(self, x: float, y: float, z: float, r: float | None = None, tol: float = 0.01) -> None:
        assert abs(self._pose["x"] - x) <= tol, f"x: {self._pose['x']} != {x}"
        assert abs(self._pose["y"] - y) <= tol, f"y: {self._pose['y']} != {y}"
        assert abs(self._pose["z"] - z) <= tol, f"z: {self._pose['z']} != {z}"
        if r is not None:
            assert abs(self._pose["r"] - r) <= tol, f"r: {self._pose['r']} != {r}"

    @property
    def move_log(self) -> list[dict]:
        return self._move_log
