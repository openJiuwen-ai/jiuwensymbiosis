# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr 底盘差速运动 worker（用 /usr/bin/python3 执行）。

闭环控制:先转 ``--dyaw`` (rad),再前进 ``--dx`` (m),各自以 ``/mc/odom`` 反馈停在
目标上(不需要标定轮径/轮距)。轮速指令走 ``RobotCommand``(``driving_wheel_*_joint``,
``MODE_VELOCITY``)发到 ``/mc/sdk/robot_command`` —— 这是底盘真正执行的通道(不是
``/mc/cmd_vel``,那条要靠未使能的厂商 chassis_controller)。前进/后退时读对应单线雷达
正前扇区(忽略近距自身/噪声回波)做急停。以一行 JSON 打印结果。

轮速的算法本身(减速档、转向偏置、弧线曲率伺服、雷达扇区取最近)不含 ROS,放在通用的
``jiuwensymbiosis/motion/diff_drive.py``;本文件只负责订阅/发布和循环。

rclpy / mc_task_msgs 等仅在 ``main`` 内 import,故 ``--help`` 在 conda 下也可用。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import select
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ``diff_drive`` 以"文件路径"方式从通用的 ``jiuwensymbiosis/motion/diff_drive.py`` 加载，
# 绕过 ``jiuwensymbiosis`` 包的 ``__init__``（其 eager 导入 openjiuwen，而本 worker 运行于
# ``/usr/bin/python3``，未安装 openjiuwen）。``diff_drive`` 仅依赖 stdlib，可独立加载。
# 路径按本文件位置（adapters/cruzr/ros2/）上溯到包根，随目录搬迁必须同步改。
_dd_path = Path(__file__).resolve().parents[3] / "motion" / "diff_drive.py"
_dd_spec = importlib.util.spec_from_file_location("cruzr_diff_drive", _dd_path)
dd = importlib.util.module_from_spec(_dd_spec)
_dd_spec.loader.exec_module(dd)


class _StopSpin(Exception):
    """Raised by the SIGTERM/SIGINT handler so the spin loop's ``finally`` halts the wheels."""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cruzr differential-drive base move (wheel velocity + odom feedback).")
    p.add_argument("--dx", type=float, default=0.0, help="Forward metres (closed-loop on odom).")
    p.add_argument("--dyaw", type=float, default=0.0, help="Left turn radians (CCW, closed-loop on odom).")
    p.add_argument("--spin", action="store_true",
                   help="Continuous-search spin: rotate at constant speed (--dyaw sign = direction) until a "
                        "'stop' line arrives on stdin (or stdin EOF / SIGTERM), bounded by --spin-max-rad / --timeout.")
    p.add_argument("--spin-max-rad", type=float, default=6.6,
                   help="Spin self-bound (≈one revolution + margin): stop even if never told to, so a dead "
                        "parent can't leave the base spinning.")
    p.add_argument("--forward", action="store_true",
                   help="Continuous-approach drive: creep straight forward at constant speed (--k-fwd) until a "
                        "'stop' line arrives on stdin (or stdin EOF / SIGTERM), bounded by --fwd-max-m / --timeout; "
                        "front-lidar e-stop.")
    p.add_argument("--fwd-max-m", type=float, default=2.5,
                   help="Forward self-bound (metres): a GENEROUS safety backstop (not the normal stop point — "
                        "the parent stops us on waist-acquire/head-lost) so a dead parent can't leave the base "
                        "driving forever.")
    p.add_argument("--command-topic", default="/mc/sdk/robot_command")
    p.add_argument("--odom-topic", default="/mc/odom")
    p.add_argument("--front-lidar-topic", default="/sensor/lidar/front")
    p.add_argument("--back-lidar-topic", default="/sensor/lidar/back")
    p.add_argument("--left-wheel-joint", default="driving_wheel_left_joint")
    p.add_argument("--right-wheel-joint", default="driving_wheel_right_joint")
    p.add_argument("--k-rot", type=float, default=0.6, help="Wheel rad/s during rotation (far from target).")
    p.add_argument("--k-rot-min", type=float, default=0.25, help="Min wheel rad/s near target (above deadband).")
    p.add_argument("--k-rot-slow-rad", type=float, default=0.5, help="Decelerate within this angle (rad) of target.")
    p.add_argument("--k-fwd", type=float, default=0.8, help="Wheel rad/s during forward drive.")
    p.add_argument("--k-fwd-min", type=float, default=0.25, help="Min wheel rad/s near target (above deadband).")
    p.add_argument("--k-fwd-slow-m", type=float, default=0.25,
                   help="Decelerate within this distance (m) of target (discrete forward phase).")
    p.add_argument("--k-steer", type=float, default=0.0,
                   help="Continuous-forward steering gain: wheel rad/s added/subtracted per rad of head "
                        "bearing fed over stdin, curving the creep toward the target. 0 = drive straight.")
    p.add_argument("--steer-max", type=float, default=0.4,
                   help="Clamp on the per-wheel steering delta (rad/s) so a large bearing can't pivot the base.")
    p.add_argument("--arc", action="store_true",
                   help="Constant-curvature arc: drive forward while curving at target radius --radius, "
                        "servoing curvature from odom Δyaw/Δs (no wheel-radius/track calibration), until the "
                        "accumulated heading change reaches --arc-dyaw. Self-bounds by arc-length / timeout / "
                        "lidar / stdin-EOF / SIGTERM. Used to curve the base onto the box's face-normal line.")
    p.add_argument("--radius", type=float, default=1.0, help="Target arc radius (m, >0).")
    p.add_argument("--arc-dyaw", type=float, default=0.0,
                   help="Signed total heading change over the arc (rad); sign = turn dir (+ = left/CCW).")
    p.add_argument("--arc-k-curv", type=float, default=0.5,
                   help="P-gain: per-wheel differential (rad/s) added per unit odom curvature error (1/m).")
    p.add_argument("--arc-len-safety", type=float, default=1.5,
                   help="Arc-length self-bound = radius·|arc_dyaw|·this (a dead parent can't circle forever).")
    p.add_argument("--yaw-tol", type=float, default=0.05)
    p.add_argument("--pos-tol", type=float, default=0.05)
    p.add_argument("--safe-dist", type=float, default=0.45,
                   help="Stop if a lidar return in the travel sector is closer.")
    p.add_argument("--self-floor", type=float, default=0.25,
                   help="Ignore lidar returns closer than this (self/noise).")
    p.add_argument("--sector-deg", type=float, default=25.0,
                   help="Half-sector around travel dir watched for obstacles.")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--serve", action="store_true",
                   help="Resident mode: init rclpy + odom once, then loop reading '<dx> <dyaw>' request "
                        "lines from stdin, run one discrete rotate+forward move per line, emit one JSON "
                        "result each. 'stop'/EOF/SIGTERM exits. Avoids a fresh process per navigate_relative.")
    return p


def _emit(d: dict) -> None:
    # stdout IS this worker's protocol (parsed by ros2/worker.py), not logging.
    sys.stdout.write(json.dumps(d) + "\n")
    sys.stdout.flush()


def main(argv: Optional[list] = None) -> int:
    a = _build_parser().parse_args(argv)

    import rclpy
    from mc_task_msgs.msg import JointCmd, RobotCommand
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan

    rclpy.init(args=None)
    n = rclpy.create_node("jiuwensymbiosis_cruzr_wheel_worker")
    st: dict[str, Any] = {"yaw": None, "x": None, "y": None, "stamp_ns": None, "front": None, "back": None}

    q = QoSProfile(depth=1)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.durability = DurabilityPolicy.VOLATILE
    q.history = HistoryPolicy.KEEP_LAST

    def on_odom(m):
        obj = m.pose.pose.orientation
        st["yaw"] = math.atan2(2 * (obj.w * obj.z + obj.x * obj.y), 1 - 2 * (obj.y * obj.y + obj.z * obj.z))
        st["x"] = m.pose.pose.position.x
        st["y"] = m.pose.pose.position.y
        st["stamp_ns"] = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)

    sector = math.radians(a.sector_deg)

    def scan_min(msg):
        return dd.sector_min_range(
            msg.ranges, msg.angle_min, msg.angle_increment,
            half_sector_rad=sector, range_min=msg.range_min, range_max=msg.range_max,
            self_floor=a.self_floor)

    n.create_subscription(Odometry, a.odom_topic, on_odom, q)
    n.create_subscription(LaserScan, a.front_lidar_topic, lambda m: st.__setitem__("front", scan_min(m)), q)
    n.create_subscription(LaserScan, a.back_lidar_topic, lambda m: st.__setitem__("back", scan_min(m)), q)
    pub = n.create_publisher(RobotCommand, a.command_topic, 10)

    try:
        t = time.time()
        while st["stamp_ns"] is None and time.time() - t < 4.0:
            rclpy.spin_once(n, timeout_sec=0.1)
        if st["stamp_ns"] is None:
            _emit({"ok": False, "reason": "no_odom"})
            return 0
        offset_ns = n.get_clock().now().nanoseconds - st["stamp_ns"]

        def wheels(vl, vr):
            # Robot-aligned stamp (dev/robot clocks may differ).
            t_ns = n.get_clock().now().nanoseconds - offset_ns
            cmd = RobotCommand()
            cmd.header.stamp.sec = int(t_ns // 1_000_000_000)
            cmd.header.stamp.nanosec = int(t_ns % 1_000_000_000)
            for name, v in ((a.left_wheel_joint, vl), (a.right_wheel_joint, vr)):
                jc = JointCmd()
                jc.name = name
                jc.control_mode = JointCmd.MODE_VELOCITY
                jc.velocity = float(v)
                cmd.joint_cmd.append(jc)
            pub.publish(cmd)

        def stop():
            for _ in range(8):
                wheels(0.0, 0.0)
                rclpy.spin_once(n, timeout_sec=0.0)
                time.sleep(0.05)

        def _do_move(dx: float, dyaw: float, k_rot: float, k_fwd: float, k_rot_slow: float) -> dict:
            """One discrete rotate-then-forward move, closed-loop on odom (rotate by dyaw, then drive
            dx). Shared by the one-shot CLI path and the resident --serve loop; k_rot/k_fwd/k_rot_slow
            are per-move so --serve can honour navigate_relative's gentle-approach overrides.
            """
            res: dict = {"ok": True}
            # rotate phase (closed loop on yaw)
            if abs(dyaw) > a.yaw_tol:
                target = st["yaw"] + dyaw
                t0 = time.time()
                while True:
                    rem = dd.wrap_angle(target - st["yaw"])
                    if abs(rem) < a.yaw_tol:
                        break
                    if time.time() - t0 > a.timeout:
                        res = {"ok": False, "reason": "rotate_timeout", "yaw_reached": round(st["yaw"], 4)}
                        break
                    wheels(*dd.rotate_wheels(rem, k_rot=k_rot, k_rot_min=a.k_rot_min,
                                             k_rot_slow_rad=k_rot_slow))
                    rclpy.spin_once(n, timeout_sec=0.0)
                    time.sleep(0.05)
                stop()
                res["yaw_reached"] = round(st["yaw"], 4)
            # forward phase (closed loop on distance, lidar-gated)
            if res.get("ok") and abs(dx) > a.pos_tol:
                x0, y0 = st["x"], st["y"]
                s = 1.0 if dx > 0 else -1.0
                t0 = time.time()
                while True:
                    dist = math.hypot(st["x"] - x0, st["y"] - y0)
                    if dist >= abs(dx):
                        break
                    if time.time() - t0 > a.timeout:
                        res = {"ok": False, "reason": "forward_timeout", "dist": round(dist, 3)}
                        break
                    lidar = st["front"] if s > 0 else st["back"]
                    if lidar is not None and lidar < a.safe_dist:
                        res = {"ok": False, "reason": "lidar_blocked",
                               "range": round(lidar, 3), "dist": round(dist, 3)}
                        break
                    wheels(*dd.forward_wheels(abs(dx) - dist, s, k_fwd=k_fwd, k_fwd_min=a.k_fwd_min,
                                              k_fwd_slow_m=a.k_fwd_slow_m))
                    rclpy.spin_once(n, timeout_sec=0.0)
                    time.sleep(0.05)
                stop()
                res["dist_traveled"] = round(math.hypot(st["x"] - x0, st["y"] - y0), 3)
            return res

        # ---- continuous-search spin (constant speed, interruptible) ----
        # Rotate in place at constant speed while the parent polls its (slow) detector; the parent
        # writes "stop" to our stdin the moment it sees the target. Multiple independent stops keep
        # the base from ever spinning away: stdin sentinel (normal), stdin EOF (parent died), SIGTERM
        # (fallback signal), and a self-imposed max-angle / timeout. Every exit path runs stop().
        if a.spin:
            def _raise_stop(*_a):
                raise _StopSpin()

            signal.signal(signal.SIGTERM, _raise_stop)
            signal.signal(signal.SIGINT, _raise_stop)
            direction = 1.0 if a.dyaw >= 0 else -1.0   # rem>0 -> CCW/left -> L=-k, R=+k
            prev, acc = st["yaw"], 0.0
            t0 = time.time()
            result = {"ok": True, "stopped": False, "reason": "spin_complete"}
            try:
                while acc < a.spin_max_rad:
                    if time.time() - t0 > a.timeout:
                        result["reason"] = "spin_timeout"
                        break
                    r, _, _ = select.select([sys.stdin], [], [], 0)  # non-blocking: stop requested?
                    if r:
                        line = sys.stdin.readline()
                        if line == "" or line.strip() == "stop":     # EOF (parent gone) or sentinel
                            result["stopped"] = True
                            result["reason"] = "stopped"
                            break
                    wheels(*dd.spin_wheels(direction, a.k_rot))
                    rclpy.spin_once(n, timeout_sec=0.0)
                    time.sleep(0.05)
                    d = dd.wrap_angle(st["yaw"] - prev)
                    acc += abs(d)
                    prev = st["yaw"]
            except _StopSpin:
                result = {"ok": True, "stopped": True, "reason": "sigterm"}
            finally:
                stop()
            result["yaw_turned"] = round(acc, 4)
            _emit(result)
            return 0

        # ---- continuous-approach forward (constant speed, STEERED toward the target, lidar-gated) ----
        # Creep forward while the parent polls its (slow) waist+head detectors. The parent feeds the
        # live head bearing over stdin each poll; we bias the wheels to CURVE toward it (not just drive
        # straight), and it writes "stop" the instant the waist acquires. When the head LOSES the anchor
        # the parent writes "hold": we pause the wheels (hold position) instead of creeping blind on the
        # last bearing — the next numeric bearing resumes driving. Same four independent stops as --spin
        # (stdin sentinel / EOF / SIGTERM / self-bound max-dist+timeout — the last is now a GENEROUS
        # safety backstop, not the normal stopping point: the parent stops us on waist-acquire/head-lost)
        # plus the front-lidar e-stop. Every exit path runs stop(). --k-steer 0 degrades to a straight
        # (still hold-able) creep.
        if a.forward:
            def _raise_stop(*_a):
                raise _StopSpin()

            signal.signal(signal.SIGTERM, _raise_stop)
            signal.signal(signal.SIGINT, _raise_stop)
            x0, y0 = st["x"], st["y"]
            t0 = time.time()
            steer = 0.0                 # live head bearing (rad, + = target left of centre), from stdin
            holding = False             # head lost → pause wheels (don't creep blind on a stale bearing)
            stop_now = False
            result = {"ok": True, "stopped": False, "reason": "drive_complete"}
            try:
                while True:
                    dist = math.hypot(st["x"] - x0, st["y"] - y0)
                    if dist >= a.fwd_max_m:
                        break
                    if time.time() - t0 > a.timeout:
                        result["reason"] = "drive_timeout"
                        break
                    if st["front"] is not None and st["front"] < a.safe_dist:
                        result["ok"] = False
                        result["reason"] = "lidar_blocked"
                        result["range"] = round(st["front"], 3)
                        break
                    # Drain stdin: 'stop'/EOF halts; 'hold' pauses (head lost); a numeric line updates the
                    # live steer bearing AND resumes driving.
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    while r:
                        line = sys.stdin.readline()
                        if line == "" or line.strip() == "stop":     # EOF (parent gone) or sentinel
                            result["stopped"] = True
                            result["reason"] = "stopped"
                            stop_now = True
                            break
                        tok = line.strip()
                        if tok == "hold":                            # head lost the anchor → pause
                            holding = True
                        elif tok:
                            try:
                                steer = float(tok.split()[-1])       # "0.12" or "steer 0.12"
                                holding = False                      # fresh bearing → resume driving
                            except ValueError:
                                pass
                        r, _, _ = select.select([sys.stdin], [], [], 0)
                    if stop_now:
                        break
                    if holding:
                        wheels(0.0, 0.0)                             # paused: hold position, don't advance
                    else:
                        wheels(*dd.steered_wheels(steer, k_fwd=a.k_fwd, k_steer=a.k_steer,
                                                  steer_max=a.steer_max))
                    rclpy.spin_once(n, timeout_sec=0.0)
                    time.sleep(0.05)
            except _StopSpin:
                result = {"ok": True, "stopped": True, "reason": "sigterm"}
            finally:
                stop()
            result["dist_traveled"] = round(math.hypot(st["x"] - x0, st["y"] - y0), 3)
            _emit(result)
            return 0

        # ---- constant-curvature arc (odom-servoed curvature, stops on accumulated heading) ----
        # Drive forward while curving at target radius R, servoing the wheel differential from the
        # odom-measured instantaneous curvature (Δyaw/Δs) so no wheel-radius/track calibration is needed
        # (same odom-closed idea as the rotate/forward phases). The grasp fine-approach uses this to curve
        # the base ONTO the box's face-normal line in one smooth move. Self-bounds: accumulated |Δyaw| ≥
        # |arc_dyaw| (normal stop), arc-length cap, timeout, front-lidar e-stop, stdin EOF, SIGTERM — every
        # exit path runs stop(). Landing-position error is absorbed by the caller's vision-corrected straight-in.
        if a.arc:
            def _raise_stop(*_a):
                raise _StopSpin()

            signal.signal(signal.SIGTERM, _raise_stop)
            signal.signal(signal.SIGINT, _raise_stop)
            s = 1.0 if a.arc_dyaw >= 0 else -1.0           # +1 left/CCW, -1 right/CW
            kappa_t = 1.0 / max(a.radius, 1e-3)            # target |curvature| (1/m)
            l_max = a.radius * abs(a.arc_dyaw) * a.arc_len_safety
            px, py, pyaw = st["x"], st["y"], st["yaw"]
            acc_yaw = acc_s = delta = 0.0
            t0 = time.time()
            result = {"ok": True, "reason": "arc_complete"}
            try:
                while abs(acc_yaw) < abs(a.arc_dyaw):
                    if time.time() - t0 > a.timeout:
                        result = {"ok": False, "reason": "arc_timeout"}
                        break
                    if st["front"] is not None and st["front"] < a.safe_dist:
                        result = {"ok": False, "reason": "lidar_blocked", "range": round(st["front"], 3)}
                        break
                    dyaw_i = dd.wrap_angle(st["yaw"] - pyaw)
                    ds_i = math.hypot(st["x"] - px, st["y"] - py)
                    acc_yaw += dyaw_i
                    acc_s += ds_i
                    px, py, pyaw = st["x"], st["y"], st["yaw"]
                    if acc_s >= l_max:
                        result["reason"] = "arc_len_cap"
                        break
                    kappa_m = dd.measured_curvature(dyaw_i, ds_i, kappa_t)
                    delta = dd.arc_delta(delta, kappa_t, kappa_m, gain=a.arc_k_curv, k_fwd=a.k_fwd)
                    # s=+1 ⇒ right faster ⇒ CCW
                    wheels(*dd.arc_wheels(abs(a.arc_dyaw) - abs(acc_yaw), s, delta, k_fwd=a.k_fwd,
                                          k_fwd_min=a.k_fwd_min, slow_band_rad=a.k_rot_slow_rad))
                    rclpy.spin_once(n, timeout_sec=0.0)
                    time.sleep(0.05)
            except _StopSpin:
                result = {"ok": True, "reason": "sigterm"}
            finally:
                stop()
            result["yaw_turned"] = round(acc_yaw, 4)
            result["dist_traveled"] = round(acc_s, 3)
            _emit(result)
            return 0

        # ---- resident serve mode: idle until a "<dx> <dyaw>" request line, run one discrete move,
        # emit one JSON result, loop. "stop"/EOF/SIGTERM exits. Keeps rclpy + odom warm across moves
        # so each navigate_relative no longer pays a fresh process + DDS discovery + odom-wait.
        if a.serve:
            def _raise_stop(*_a):
                raise _StopSpin()

            signal.signal(signal.SIGTERM, _raise_stop)
            signal.signal(signal.SIGINT, _raise_stop)
            try:
                while True:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    rclpy.spin_once(n, timeout_sec=0.0)   # keep odom fresh while idle
                    if not r:
                        continue
                    line = sys.stdin.readline()
                    if line == "" or line.strip() == "stop":   # EOF (parent gone) or sentinel
                        break
                    # Request: "dx dyaw [k_rot k_fwd k_rot_slow]" — trailing k's override the CLI
                    # defaults per move (navigate_relative's gentle-approach values).
                    toks = line.split()
                    try:
                        dx = float(toks[0])
                        dyaw = float(toks[1]) if len(toks) > 1 else 0.0
                        k_rot = float(toks[2]) if len(toks) > 2 else a.k_rot
                        k_fwd = float(toks[3]) if len(toks) > 3 else a.k_fwd
                        k_rot_slow = float(toks[4]) if len(toks) > 4 else a.k_rot_slow_rad
                    except (ValueError, IndexError):
                        _emit({"ok": False, "reason": "bad_request"})
                        continue
                    _emit(_do_move(dx, dyaw, k_rot, k_fwd, k_rot_slow))
            except _StopSpin:
                pass
            finally:
                stop()
            return 0

        result = _do_move(a.dx, a.dyaw, a.k_rot, a.k_fwd, a.k_rot_slow_rad)
        _emit(result)
        return 0
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
