# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real-time perceive→act servo primitives (the fast-path control layer).

Three decoupled pieces, body-agnostic:

* ``StreamingFrameSource`` — background continuous frame capture (持续抓帧).
* ``BackgroundTracker``    — background perception at its own rate (持续感知).
* ``ServoController``       — high-frequency ``control_hz`` control loop with
  slew limiting toward the latest target (实时执行).

``ServoBinding`` wires these to a concrete ``RobotSession``. The fast execution
mode (``agent/fast``) composes them (via ``track_detect``) into a Perceive+Act
loop that runs in-process with **no LLM in the loop**.
"""

from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.agent.fast.realtime.servo import (
    ServoConfig,
    ServoController,
    ServoResult,
)
from jiuwensymbiosis.agent.fast.realtime.streaming import StreamingFrameSource
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker

__all__ = [
    "StreamingFrameSource",
    "BackgroundTracker",
    "ServoController",
    "ServoConfig",
    "ServoResult",
    "ServoBinding",
]
