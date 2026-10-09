# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Piper integration fixtures: fake CAN only, exercise the real API/Env/driver."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.adapters.piper import lowlevel
from jiuwensymbiosis.adapters.piper.api import PiperApi
from jiuwensymbiosis.adapters.piper.config import PiperConfig
from jiuwensymbiosis.adapters.piper.env import PiperEnv


class _InstantArm:
    def __init__(self, pose, calls):
        self.pose = tuple(round(value * 1000) for value in pose)
        self.commands = []
        self.calls = calls
        self.motion_modes = []

    def ConnectPort(self):
        self.calls.append("connect")

    def DisconnectPort(self):
        self.calls.append("disconnect")

    def EnablePiper(self):
        self.calls.append("enable")
        return True

    def MotionCtrl_2(self, *args):
        self.calls.append("motion_mode")
        self.motion_modes.append(args)

    def GripperCtrl(self, *args):
        self.calls.append("gripper")

    def EndPoseCtrl(self, *values):
        self.calls.append("end_pose")
        self.commands.append(values)
        self.pose = values

    def GetArmEndPoseMsgs(self):
        self.calls.append("read_pose")
        fields = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")
        return SimpleNamespace(end_pose=SimpleNamespace(**dict(zip(fields, self.pose, strict=True))))


@pytest.fixture
def piper_factory(monkeypatch):
    envs = []
    monkeypatch.setattr(lowlevel, "_attach_cmd_log_handler", lambda: None)
    monkeypatch.setattr(lowlevel.time, "sleep", lambda _seconds: None)

    def build(
        *, calibration="none", initial_pose=(200, 20, 400, 180, 30, 0), anchor=(210, 30, 60), sdk_calls=None, **config
    ):
        calls = sdk_calls if sdk_calls is not None else []
        arm = _InstantArm(initial_pose, calls)

        def arm_factory(_port):
            calls.append("construct")
            return arm

        monkeypatch.setattr(lowlevel, "_require_can_interface", lambda _port: calls.append("can_precheck"))
        monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=arm_factory))
        calib = {"T_flange_cam": {"matrix_4x4": np.eye(4)}, "intrinsics": np.eye(3)}
        if calibration == "anchored":
            calib["object"] = {"xyz_base_mm": list(anchor)}
        monkeypatch.setattr(lowlevel, "load_calibration", lambda _path: calib)
        env = PiperEnv(PiperConfig(calib_path=None if calibration == "none" else "stub-calibration.json", **config))
        envs.append(env)
        env.connect()
        return SimpleNamespace(api=PiperApi(env), env=env, arm=arm)

    yield build
    for env in reversed(envs):
        env.disconnect()
