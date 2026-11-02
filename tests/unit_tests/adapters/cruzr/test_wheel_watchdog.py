"""Run the actual wheel-worker control loops with timed DDS feedback, without ROS."""

import sys
from types import SimpleNamespace as NS

import pytest

from jiuwensymbiosis.adapters.cruzr.ros2 import wheel_worker


@pytest.mark.parametrize("mode", ["spin", "forward", "arc", "translate", "rotate"])
@pytest.mark.parametrize("failure", ["frozen_odom", "timeout"])
def test_feedback_loss_or_timeout_stops_every_control_mode(monkeypatch, mode, failure):
    now = [1.0]
    commands, results, lifecycle = [], [], []
    callbacks = {}
    stamp = [1_000_000_000]

    class JointCmd:
        MODE_VELOCITY = 1

    class Node:
        def create_subscription(self, cls, topic, callback, qos):
            callbacks[topic] = callback

        def create_publisher(self, *args):
            return NS(publish=lambda cmd: commands.append((now[0], tuple(j.velocity for j in cmd.joint_cmd))))

        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=int(now[0] * 1e9)))

        def destroy_node(self):
            lifecycle.append("destroyed")

    def pump(*args, **kwargs):
        # Repeated timestamps while moving must not extend the watchdog. Once
        # zero is published, fresh stationary samples can confirm the stop.
        stopped = bool(commands and commands[-1][1] == (0.0, 0.0))
        if failure == "timeout" or stopped:
            stamp[0] = int(now[0] * 1e9)
        callbacks["/mc/odom"](
            NS(
                header=NS(stamp=NS(sec=stamp[0] // 1_000_000_000, nanosec=stamp[0] % 1_000_000_000)),
                pose=NS(pose=NS(position=NS(x=0.0, y=0.0), orientation=NS(w=1.0, x=0.0, y=0.0, z=0.0))),
                twist=NS(twist=NS(linear=NS(x=0.0, y=0.0), angular=NS(z=0.0))),
            )
        )

    def sleep(dt):
        now[0] += dt

    clock = NS(monotonic=lambda: now[0], time=lambda: now[0], sleep=sleep)
    ros = NS(
        init=lambda **kwargs: lifecycle.append("initialized"),
        create_node=lambda name: Node(),
        spin_once=pump,
        shutdown=lambda: lifecycle.append("shutdown"),
    )
    modules = {
        "rclpy": ros,
        "rclpy.qos": NS(
            QoSProfile=lambda **kwargs: NS(),
            ReliabilityPolicy=NS(BEST_EFFORT=1),
            DurabilityPolicy=NS(VOLATILE=1),
            HistoryPolicy=NS(KEEP_LAST=1),
        ),
        "mc_task_msgs.msg": NS(JointCmd=JointCmd, RobotCommand=lambda: NS(header=NS(stamp=NS()), joint_cmd=[])),
        "nav_msgs.msg": NS(Odometry=object),
        "sensor_msgs.msg": NS(LaserScan=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    confirm_stop = wheel_worker._confirm_stop
    monkeypatch.setattr(wheel_worker, "_confirm_stop", lambda *a: confirm_stop(*a, clock=clock.monotonic, sleep=sleep))
    monkeypatch.setattr(wheel_worker, "time", clock)
    monkeypatch.setattr(wheel_worker, "select", NS(select=lambda *a: ([], [], [])))
    monkeypatch.setattr(wheel_worker, "signal", NS(signal=lambda *a: None, SIGTERM=15, SIGINT=2))
    monkeypatch.setattr(wheel_worker, "_emit", results.append)
    argv = {
        "spin": ["--spin"],
        "forward": ["--forward"],
        "arc": ["--arc", "--arc-dyaw", "1.0"],
        "translate": ["--dx", "1.0"],
        "rotate": ["--dyaw", "1.0"],
    }[mode]
    assert wheel_worker.main([*argv, "--timeout", "0.4"]) == 0
    assert len(results) == 1
    result = results[0]
    assert result["ok"] is False
    expected = (
        "odom_stale"
        if failure == "frozen_odom"
        else {
            "spin": "spin_timeout",
            "forward": "drive_timeout",
            "arc": "arc_timeout",
            "translate": "forward_timeout",
            "rotate": "rotate_timeout",
        }[mode]
    )
    assert result["reason"] == expected
    assert result["settled"] is True and result["stop_confirmation"] == "measured"
    nonzero = [(t, v) for t, v in commands if v != (0.0, 0.0)]
    assert nonzero and max(t for t, _ in nonzero) <= (1.3 if failure == "frozen_odom" else 1.45)
    first_zero = next(i for i, (_, v) in enumerate(commands) if v == (0.0, 0.0))
    assert all(v == (0.0, 0.0) for _, v in commands[first_zero:])
    assert lifecycle == ["initialized", "destroyed", "shutdown"]
