# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Low-level Cruzr ROS 2 driver.

This module owns all direct ROS 2 interaction. It publishes
``mc_task_msgs/RobotCommand`` to ``/mc/sdk/robot_command`` and keeps a latest
joint-state snapshot from ``/mc/sdk/robot_state``. It runs rclpy in-process, so
the interpreter's Python must match the ROS build (e.g. the conda env on Jazzy).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.perception.frame import CameraFrame
from jiuwensymbiosis.ros2 import worker as ros2_worker
from jiuwensymbiosis.ros2.image_decode import decode_image_msg as decode_image_msg  # re-export

logger = logging.getLogger(__name__)


class CruzrLowLevel:
    """ROS 2 wrapper for Cruzr joint commands."""

    def __init__(self, cfg: CruzrConfig) -> None:
        """Create the ROS 2 node, command publisher, and state subscriber."""
        self.cfg = cfg
        self._camera_obj = None  # 懒创建的 CruzrCamera
        self._nav_obj = None  # 懒创建的 CruzrNav
        self._closed = False
        self._latest_joints: dict[str, float] = {}
        self._lock = threading.RLock()
        self._head_fk_cache: Optional[tuple] = None   # (urdf_path, model, data, frame_id), built on first use

        try:
            import rclpy
            from mc_state_msgs.msg import RobotState
            from mc_task_msgs.msg import JointCmd, RobotCommand
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        except ImportError as exc:
            raise RuntimeError(
                "[Cruzr] in-process rclpy is required but unavailable for the current interpreter "
                f"{sys.executable} (Python {sys.version_info.major}.{sys.version_info.minor}). Source the "
                "ROS 2 / Cruzr_ws environment and run under an interpreter whose Python matches the ROS "
                "build (e.g. the project conda env on Jazzy)."
            ) from exc

        self._rclpy = rclpy
        self._joint_cmd_cls = JointCmd
        self._robot_command_cls = RobotCommand

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        else:
            self._owns_rclpy_context = False

        self._node = rclpy.create_node("jiuwensymbiosis_cruzr")
        self._publisher = self._node.create_publisher(RobotCommand, cfg.command_topic, 10)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        qos.history = HistoryPolicy.KEEP_LAST
        self._subscription = self._node.create_subscription(
            RobotState,
            cfg.state_topic,
            self._on_robot_state,
            qos,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, name="cruzr-ros2-spin", daemon=True)
        self._spin_thread.start()

        logger.info("[Cruzr] connected command_topic=%s state_topic=%s", cfg.command_topic, cfg.state_topic)
        self._await_first_state(5.0)

    def _await_first_state(self, timeout_s: float) -> None:
        """Block until the first joint-state frame arrives, so DDS discovery completes
        before any move (else early moves run open-loop) and the TF buffer has time to
        warm. Logs whether the state stream came up — a quick health check on connect.
        """
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.get_joint_positions():
                logger.info("[Cruzr] joint-state stream live")
                return
            time.sleep(0.05)
        logger.warning("[Cruzr] no joint state within %.1fs of connect; DDS reception may be "
                       "misconfigured (moves will run open-loop, detect may report no_live_tf)", timeout_s)

    @property
    def camera(self):
        """懒创建 CruzrCamera（仅在抓帧时构造，经子进程 worker 抓帧；不影响纯命令路径）。"""
        if self._camera_obj is None:
            self._camera_obj = CruzrCamera(self.cfg)
        return self._camera_obj

    def grab_frames(self, camera: str = "waist"):
        """抓一帧 -> (rgb, depth_m, K, tf_base_cam)；无帧返回 None。"""
        frame = self.camera.grab(camera)
        if frame is None:
            return None
        return frame.rgb, frame.depth_m, frame.intrinsics, frame.tf_base_cam

    def grab_head_frame(self):
        """抓头部一帧【带组织化点云 + base<-optical TF】(仅 grounded 3D 验证用) ->
        ``(rgb, cloud_xyz[H,W,3] | None, tf_base_cam[4x4] | None)``；worker 整体失败返回 None。
        """
        result = self.camera.grab_head_frame()
        if result is None:
            return None
        rgb, cloud, tf_base_cam = result
        if tf_base_cam is None:
            tf_base_cam = self._head_tf_from_joint_state()
            if tf_base_cam is not None:
                logger.info("[CruzrCamera] using URDF FK for base<-head optical TF")
        return rgb, cloud, tf_base_cam

    def _head_tf_from_joint_state(self) -> Optional[np.ndarray]:
        """Compute ``base_link <- stereo_left_rgb_optical_link`` from live joints.

        The robot currently publishes no complete base-to-head TF chain on DDS.
        The command/state channel does provide all joint positions, so use the
        deployed URDF for the pose-dependent portion and append the two calibrated
        zero-translation camera mount rotations from ``stereo_depth.launch.py``.
        """
        joints = self.get_joint_positions()
        if not joints:
            logger.warning("[CruzrCamera] cannot compute head TF: no joint state")
            return None
        try:
            import pinocchio as pin

            cache = self._head_fk_cache
            urdf_path = str(self.cfg.urdf_path)
            if cache is None or cache[0] != urdf_path:
                model = pin.buildModelFromUrdf(urdf_path)
                data = model.createData()
                frame_id = model.getFrameId("head_pitch_link")
                if frame_id >= len(model.frames):
                    raise ValueError("head_pitch_link is absent from URDF")
                cache = (urdf_path, model, data, frame_id)
                self._head_fk_cache = cache
            _, model, data, frame_id = cache

            q = pin.neutral(model)
            for name, position in joints.items():
                joint_id = model.getJointId(str(name))
                if joint_id == 0:
                    continue
                joint = model.joints[joint_id]
                if joint.nq == 1:
                    q[joint.idx_q] = float(position)
                elif joint.nq == 2 and joint.nv == 1:
                    # Pinocchio stores an unbounded revolute joint as cos/sin.
                    q[joint.idx_q] = np.cos(float(position))
                    q[joint.idx_q + 1] = np.sin(float(position))

            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            tf_base_head = np.asarray(data.oMf[frame_id].homogeneous, dtype=np.float64)

            # head_pitch -> stereo_camera is Rx(-90 deg); stereo_camera ->
            # optical is q=(-.5,.5,-.5,.5). Their product is the matrix below.
            tf_head_optical = np.eye(4, dtype=np.float64)
            tf_head_optical[:3, :3] = np.asarray([
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ])
            tf_base_optical = tf_base_head @ tf_head_optical
            tf_base_optical[:3, 3] *= 1000.0  # downstream head geometry uses mm
            return tf_base_optical
        except Exception as exc:  # noqa: BLE001 — fail closed in strict grounded mode
            logger.warning("[CruzrCamera] head TF FK failed: %s", exc)
            return None

    @property
    def nav(self):
        """懒创建 CruzrNav（经子进程 worker 做轮速+odom 闭环）。"""
        if self._nav_obj is None:
            self._nav_obj = CruzrNav(self.cfg)
        return self._nav_obj

    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0, *,
                          k_rot: Optional[float] = None,
                          k_rot_slow_rad: Optional[float] = None,
                          k_fwd: Optional[float] = None) -> dict:
        """相对当前位姿导航（差速轮速 + odom 闭环，经子进程 worker）。
        k_rot/k_rot_slow_rad/k_fwd 非 None 时覆盖全局 base_k_*(approach 精定位用温和值)。
        """
        return self.nav.navigate_relative(
            float(dx_m), float(dy_m), float(dyaw_rad),
            k_rot=k_rot, k_rot_slow_rad=k_rot_slow_rad, k_fwd=k_fwd)

    def navigate_arc(self, radius: float, dyaw: float, *, k_fwd: Optional[float] = None) -> dict:
        """沿定曲率弧前进(边转边走,odom 闭环,经子进程 worker):半径 radius(m)、总转角 dyaw(rad,有符号)。"""
        return self.nav.navigate_arc(float(radius), float(dyaw), k_fwd=k_fwd)

    def start_base_spin(self, direction: float = 1.0) -> Any:
        """启动非阻塞【连续搜索自转】worker（匀速旋转、可中断）；返回句柄供轮询/停止。"""
        return self.nav.start_spin(float(direction))

    def base_spin_running(self, handle: Any) -> bool:
        """连续自转 worker 是否仍在转（未自行结束）。"""
        return self.nav.spin_running(handle)

    def stop_base_spin(self, handle: Any) -> dict:
        """停止连续自转 worker 并回收结果（发 stdin 'stop' 干净停车；含 SIGTERM/kill 兜底）。"""
        return self.nav.stop_spin(handle)

    def start_base_drive(self, *, k_fwd: Optional[float] = None, fwd_max_m: Optional[float] = None) -> Any:
        """启动非阻塞【连续驱近前进】worker（匀速前进、可中断、雷达急停）；返回句柄供轮询/停止。

        ``k_fwd`` / ``fwd_max_m``：末段抓取伺服用的慢速爬行 / 更紧前进自限；``None``=配置默认（粗定位不传即原行为）。
        """
        return self.nav.start_drive(k_fwd=k_fwd, fwd_max_m=fwd_max_m)

    def base_drive_running(self, handle: Any) -> bool:
        """连续前进 worker 是否仍在开（未自行结束）。"""
        return self.nav.drive_running(handle)

    def steer_base_drive(self, handle: Any, bearing_rad: float) -> None:
        """把头部实时方位喂给连续前进 worker，使其朝目标弧线转向（差速）。尽力而为，管道断了则忽略。"""
        return self.nav.steer_drive(handle, bearing_rad)

    def hold_base_drive(self, handle: Any) -> None:
        """头部丢失锚点时让连续前进 worker 暂停轮子（原地保持、不按旧方位盲进）；下次喂方位即恢复。尽力而为。"""
        return self.nav.hold_drive(handle)

    def stop_base_drive(self, handle: Any) -> dict:
        """停止连续前进 worker 并回收结果（发 stdin 'stop' 干净停车；含 SIGTERM/kill 兜底）。"""
        return self.nav.stop_drive(handle)

    def read_hand_ft(self, arm: str) -> dict:
        """读一帧手部力传感器数据。

        Returns {"ok": True, "fx", "fy", "fz", "fmag"} or {"ok": False, "reason": "no_ft"}.
        """
        topic = (self.cfg.left_hand_ft_topic if arm.lower() in ("left", "l")
                 else self.cfg.right_hand_ft_topic)
        from geometry_msgs.msg import WrenchStamped
        state = {"w": None}
        sub = self._node.create_subscription(
            WrenchStamped, topic, lambda m: state.__setitem__("w", m), 10)
        deadline = time.monotonic() + 2.0
        # self._executor runs on self._spin_thread, so the subscription callback fires without spin_once here.
        while state["w"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self._node.destroy_subscription(sub)
        if state["w"] is None:
            return {"ok": False, "reason": "no_ft"}
        f = state["w"].wrench.force
        t = state["w"].wrench.torque
        fmag = (f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5
        return {"ok": True, "fx": f.x, "fy": f.y, "fz": f.z, "fmag": fmag,
                "tx": t.x, "ty": t.y, "tz": t.z}

    # ----------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Stop executor and destroy the ROS 2 node. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for obj in (self._camera_obj, self._nav_obj):   # stop any resident workers first
            try:
                if obj is not None:
                    obj.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[Cruzr] worker close failed: %s", exc)
        try:
            self._executor.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Cruzr] executor shutdown failed: %s", exc)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        try:
            self._executor.remove_node(self._node)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._node.destroy_node()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Cruzr] node destroy failed: %s", exc)
        if self._owns_rclpy_context:
            try:
                self._rclpy.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[Cruzr] rclpy shutdown failed: %s", exc)

    # ---------------------------------------------------------------- callbacks
    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001
            if not self._closed:
                logger.warning("[Cruzr] ROS 2 executor stopped unexpectedly: %s", exc)

    def _on_robot_state(self, msg: Any) -> None:
        names = list(getattr(msg.joint_states, "name", []) or [])
        positions = list(getattr(msg.joint_states, "position", []) or [])
        with self._lock:
            self._latest_joints = {str(name): float(pos) for name, pos in zip(names, positions)}

    # ------------------------------------------------------------------- state
    def get_joint_positions(self) -> dict[str, float]:
        """Return the latest joint positions keyed by joint name.

        Reads the cached ``/mc/sdk/robot_state`` snapshot; empty until the first
        frame arrives.
        """
        with self._lock:
            return dict(self._latest_joints)

    def get_angles(self) -> list[float]:
        """Return latest joint positions as a list, in message order when known."""
        with self._lock:
            return list(self._latest_joints.values())

    # ----------------------------------------------------------------- commands
    def publish_joint_position(self, joint_name: str, position_rad: float) -> None:
        """Publish a single position command."""
        self.publish_joint_positions({joint_name: position_rad})

    def publish_joint_positions(self, joint_positions: dict[str, float]) -> None:
        """Publish one ``RobotCommand`` containing all requested joint positions."""
        if self._closed:
            raise RuntimeError("[Cruzr] driver is closed.")
        cmd = self._robot_command_cls()
        cmd.header.stamp = self._node.get_clock().now().to_msg()
        for name, pos in joint_positions.items():
            joint = self._joint_cmd_cls()
            joint.name = str(name)
            joint.control_mode = self._joint_cmd_cls.MODE_POSITION
            joint.position = float(pos)
            cmd.joint_cmd.append(joint)
        self._publisher.publish(cmd)

    def home(self, *, arm: Optional[str] = None) -> dict[str, Any]:
        """Return the selected arm to the configured home position."""
        joint = self.cfg.joint_name_for_arm(arm)
        self.move_joint_blocking({joint: self.cfg.home_position_rad})
        return {"ok": True, "arm": arm or self.cfg.default_arm, "joint": joint,
                "position_rad": self.cfg.home_position_rad}

    def raise_arm_blocking(
        self,
        *,
        arm: Optional[str] = None,
        return_home: bool = True,
        target_rad: Optional[float] = None,
    ) -> dict[str, Any]:
        """Raise a shoulder pitch joint and optionally lower it back home."""
        joint = self.cfg.joint_name_for_arm(arm)
        target = float(self.cfg.raise_position_for_arm(arm) if target_rad is None else target_rad)
        self._ramp_joint(joint, self.cfg.home_position_rad, target)
        if return_home:
            self._ramp_joint(joint, target, self.cfg.home_position_rad)
        return {
            "ok": True,
            "arm": arm or self.cfg.default_arm,
            "joint": joint,
            "target_rad": target,
            "returned_home": bool(return_home),
        }

    def _ramp_to_targets_native(self, targets: dict[str, float],
                                *, ramp_duration_s: Optional[float] = None) -> bool:
        """Smoothstep-ramp from the measured joint angles to ``targets`` so the arm
        eases into motion instead of snapping to the raw target. Dense duration-based
        setpoints (``ramp_duration_s`` @ ``control_rate_hz``) eased in/out with
        ``t^2(3-2t)``. Joints with no live state hold at their target (never a
        ramp-from-zero, which would itself be a jump). Returns True when every target
        joint had a live start.

        ``ramp_duration_s`` overrides ``cfg.ramp_duration_s`` for this one move — the
        light head uses a short ramp (``cfg.head_ramp_duration_s``) so a "look" is quick,
        while the box-carrying arms keep the gentle default.
        """
        rate = max(float(self.cfg.control_rate_hz), 1.0)
        period = 1.0 / rate
        dur = self.cfg.ramp_duration_s if ramp_duration_s is None else ramp_duration_s
        steps = max(2, int(float(dur) * rate))
        measured = self.get_joint_positions()
        missing = [k for k in targets if k not in measured]
        if missing:
            logger.warning("[Cruzr] native move: no live state for %s; holding at target (no ramp)", missing)
        start = {k: float(measured.get(k, targets[k])) for k in targets}
        for i in range(1, steps + 1):
            t = i / steps
            a = t * t * (3.0 - 2.0 * t)
            self.publish_joint_positions({k: start[k] + a * (targets[k] - start[k]) for k in targets})
            time.sleep(period)
        return not missing

    def move_joint_blocking(
        self,
        joint_positions: dict[str, float] | list[float],
        *,
        joint_names: Optional[list[str]] = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Move joints to target positions by repeatedly publishing until timeout/readback.

        When no state has been received, this still sends commands for a short
        period so the SDK controller receives a stable target.
        """
        targets = self._normalize_joint_targets(joint_positions, joint_names)
        self._ramp_to_targets_native(targets)
        deadline = time.monotonic() + float(timeout_s)
        period = 1.0 / max(float(self.cfg.control_rate_hz), 1.0)
        saw_state = bool(self.get_joint_positions())
        open_loop_deadline = time.monotonic() + max(float(self.cfg.open_loop_hold_s), period)
        while time.monotonic() < deadline:
            self.publish_joint_positions(targets)
            reached = self._targets_reached(targets)
            if reached:
                break
            if not saw_state:
                saw_state = bool(self.get_joint_positions())
                if not saw_state and time.monotonic() >= open_loop_deadline:
                    break
            time.sleep(period)
        time.sleep(max(float(self.cfg.settle_s), 0.0))
        readback = self.get_joint_positions()
        return {
            "ok": True,
            "targets": targets,
            "readback": readback,
            "open_loop": not bool(readback),
        }

    def move_joints_blocking(self, targets: dict[str, float], *, timeout_s: float = 10.0,
                             ramp_duration_s: Optional[float] = None) -> dict:
        """Move multiple joints to targets via a native multi-JointCmd smoothstep ramp.

        ``ramp_duration_s`` overrides the global ``cfg.ramp_duration_s`` for this move only
        (e.g. ``set_head`` passes the shorter ``cfg.head_ramp_duration_s``).
        """
        targets = {str(k): float(v) for k, v in targets.items()}
        ramp_ok = self._ramp_to_targets_native(targets, ramp_duration_s=ramp_duration_s)
        deadline = time.monotonic() + float(timeout_s)
        period = 1.0 / max(float(self.cfg.control_rate_hz), 1.0)
        open_loop_deadline = time.monotonic() + max(float(self.cfg.open_loop_hold_s), period)
        saw_state = bool(self.get_joint_positions())
        while time.monotonic() < deadline:
            self.publish_joint_positions(targets)
            if self._targets_reached(targets):
                break
            if not saw_state:
                saw_state = bool(self.get_joint_positions())
                if not saw_state and time.monotonic() >= open_loop_deadline:
                    break
            time.sleep(period)
        time.sleep(max(float(self.cfg.settle_s), 0.0))
        readback = self.get_joint_positions()
        return {"ok": True, "targets": targets, "readback": readback,
                "open_loop": not bool(readback), "ramp_from_state": ramp_ok}

    def stream_joint_trajectory(self, waypoints: list[dict[str, float]], *,
                                total_duration_s: float) -> dict:
        """Stream a precomputed joint-space path through ``waypoints`` (in order) as ONE
        continuous trajectory at ``control_rate_hz`` over ``total_duration_s``.

        Unlike ``move_joints_blocking`` (a smoothstep ramp from the measured start to a
        SINGLE target), this walks a multi-knot path with a single GLOBAL ease-in/out on
        the path phase, sub-interpolating between adjacent knots at the control rate. A
        Cartesian-shaped path (e.g. the gap-locked place lower) therefore executes smoothly
        in the same wall-clock as one move — no per-knot deceleration or settle. Pass
        ``waypoints[0]`` = the current measured pose so the motion starts without a jump.
        All waypoints must share the same joint keys.
        """
        pts = [{str(k): float(v) for k, v in wp.items()} for wp in waypoints if wp]
        if len(pts) < 2:
            return {"ok": False, "reason": "need_at_least_two_waypoints"}
        keys = list(pts[0].keys())
        n_seg = len(pts) - 1
        rate = max(float(self.cfg.control_rate_hz), 1.0)
        period = 1.0 / rate
        steps = max(2, int(float(total_duration_s) * rate))
        for i in range(1, steps + 1):
            s = i / steps
            a = s * s * (3.0 - 2.0 * s)          # global ease-in/out on the path phase
            x = a * n_seg
            seg = min(int(x), n_seg - 1)         # which knot interval we are in
            f = x - seg                          # position within it
            lo, hi = pts[seg], pts[seg + 1]
            self.publish_joint_positions({k: lo[k] + f * (hi[k] - lo[k]) for k in keys})
            time.sleep(period)
        # hold the final knot briefly so the position controller settles onto it
        hold_end = time.monotonic() + max(float(self.cfg.settle_s), period)
        while time.monotonic() < hold_end:
            self.publish_joint_positions(pts[-1])
            time.sleep(period)
        return {"ok": True, "waypoints": len(pts), "readback": self.get_joint_positions()}

    def set_lifter(self, q_lifter: dict[str, float], *, ramp_duration_s: float | None = None) -> dict:
        """Move the 3 lifter_pitch joints via a multi-joint ramp.

        The lifter is driven by the SAME SDK path as the arms — RobotCommand on
        ``/mc/sdk/robot_command`` (verified on hardware). It is NOT the separate
        ``/mc/lifter/controller`` JointCommand topic (that is the MoveIt
        ros2_control path, which the SDK command route does not feed).

        ``ramp_duration_s`` overrides the ramp for this move only; when omitted it uses the
        dedicated ``cfg.lifter_ramp_duration_s`` (falls back to the global default if unset).
        """
        cfg = getattr(self, "cfg", None)
        dur = ramp_duration_s if ramp_duration_s is not None else getattr(cfg, "lifter_ramp_duration_s", None)
        return self.move_joints_blocking(
            {str(k): float(v) for k, v in q_lifter.items()}, ramp_duration_s=dur)

    def turn_waist_blocking(
        self,
        target_rad: float,
        *,
        hold: dict[str, float],
        waist_joint: str = "waist_yaw_joint",
    ) -> dict:
        """Rotate ``waist_joint`` to the absolute ``target_rad`` in ONE gentle move,
        holding ``hold`` (e.g. the arm joints) fixed.

        Issues a single ``move_joints_blocking`` with the arms + waist target: the
        motion backend ramps every joint from its measured angle with a smoothstep
        (ease-in/out, zero jerk) over ``cfg.ramp_duration_s``, so the held arms stay
        put (current -> current) while the waist turns smoothly. It MUST be one
        command: each ``move_joints_blocking`` runs a full ``ramp_duration_s``
        trajectory, so per-degree chunking would make the turn take ~ramp_duration_s
        EACH — a minute-long crawl for a large turn. Speed/gentleness are governed by
        ``cfg.ramp_duration_s``.
        """
        target_rad = float(target_rad)
        hold = {str(k): float(v) for k, v in (hold or {}).items()}
        res = self.move_joints_blocking({**hold, waist_joint: target_rad})
        return {"ok": True, "joint": waist_joint, "target_rad": target_rad,
                "readback": res.get("readback")}

    # ------------------------------------------------------------------ helpers
    def _ramp_joint(self, joint_name: str, start: float, target: float) -> None:
        step = abs(float(self.cfg.step_rad))
        if step <= 0:
            raise ValueError("[Cruzr] step_rad must be > 0")
        period = 1.0 / max(float(self.cfg.control_rate_hz), 1.0)
        current = float(start)
        target = float(target)
        direction = 1.0 if target >= current else -1.0
        while True:
            reached = (direction > 0 and current >= target) or (direction < 0 and current <= target)
            if reached:
                current = target
                self.publish_joint_position(joint_name, current)
                break
            current += direction * step
            if direction > 0:
                current = min(current, target)
            else:
                current = max(current, target)
            self.publish_joint_position(joint_name, current)
            time.sleep(period)
        time.sleep(max(float(self.cfg.settle_s), 0.0))

    def _normalize_joint_targets(
        self,
        joint_positions: dict[str, float] | list[float],
        joint_names: Optional[list[str]],
    ) -> dict[str, float]:
        if isinstance(joint_positions, dict):
            return {str(k): float(v) for k, v in joint_positions.items()}
        if joint_names is None:
            default_joint = self.cfg.joint_name_for_arm(self.cfg.default_arm)
            if len(joint_positions) != 1:
                raise ValueError("joint_names is required when moving more than one unnamed joint")
            joint_names = [default_joint]
        if len(joint_names) != len(joint_positions):
            raise ValueError("joint_names and joint_positions length mismatch")
        return {str(k): float(v) for k, v in zip(joint_names, joint_positions)}

    def _targets_reached(self, targets: dict[str, float]) -> bool:
        state = self.get_joint_positions()
        if not state:
            return False
        tol = float(self.cfg.reach_tolerance_rad)
        for name, target in targets.items():
            if name not in state:
                return False
            if abs(float(state[name]) - float(target)) > tol:
                return False
        return True


class CruzrCamera:
    """用 ``/usr/bin/python3`` 子进程订阅腰部 RGBD 相机一帧（conda 3.11 无法 import rclpy）。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._worker = ros2_worker.worker_path("jiuwensymbiosis.adapters.cruzr.ros2.camera_worker")
        # build_cmd -> its persistent --serve worker (one per camera command; resident_workers=True)
        self._resident: dict = {}

    def grab(self, camera: str = "waist") -> Optional[CameraFrame]:
        """抓取一帧，失败返回 None（保持 reason=no_camera 回退链）。"""
        if camera == "head":
            return self._grab(self._head_cmd, want_depth=False)
        return self._grab(self._waist_cmd, want_depth=True)

    def grab_head_frame(self):
        """抓头部一帧【带点云 + TF】(仅 grounded 3D 验证用)，返回 plain tuple
        ``(rgb, cloud_xyz[H,W,3] | None, tf_base_cam[4x4] | None)``；worker 整体失败返回 None。

        cloud/tf 任一为 None 表示该项拿不到，上层据此优雅降级回纯 bearing。不用 CameraFrame
        (它是 shared 契约、不携带点云)，故回一个 Cruzr 私有 tuple。
        """
        def _read(out: Path, meta: dict):
            rgb = np.load(out / "color.npy")
            cloud = None
            if meta.get("has_cloud") and (out / "cloud.npy").exists():
                cloud = np.load(out / "cloud.npy")
            tf = meta.get("tf_base_cam")
            tf_base_cam = np.asarray(tf, dtype=np.float64) if tf else None
            return rgb, cloud, tf_base_cam

        return self._run_and_read(self._head_cloud_cmd, _read)

    def _waist_cmd(self, out: Path) -> list:
        return [
            self.cfg.ros_python, str(self._worker),
            "--color-topic", self.cfg.waist_color_topic,
            "--depth-topic", self.cfg.waist_depth_topic,
            "--camera-info-topic", self.cfg.waist_camera_info_topic,
            "--color-msg-type", self.cfg.color_msg_type,
            "--depth-msg-type", self.cfg.depth_msg_type,
            "--depth-scale", str(self.cfg.depth_scale),
            "--timeout-s", str(self.cfg.camera_grab_timeout_s),
            "--output-dir", str(out),
            "--base-frame", getattr(self.cfg, "base_frame", "base_link"),
            "--camera-optical-frame", getattr(self.cfg, "camera_optical_frame",
                                              "waist_front_rgbd_color_optical_frame"),
            "--tf-timeout-s", str(getattr(self.cfg, "camera_tf_timeout_s", 8.0)),
        ]

    def _head_cmd(self, out: Path) -> list:
        # Head search grabs the RIGHT-EYE RAW image — the vendor right-eye topic (shm_msgs/Image2m,
        # yuv422; header+payload still traverse DDS), NOT the left rect-color compressed stream. Pure
        # 2-D: no depth/cloud/TF (grounded "on" is judged by 2-D bbox overlap in the api layer).
        color_topic = getattr(self.cfg, "head_right_topic", self.cfg.head_left_topic)
        color_msg_type = getattr(
            self.cfg, "head_right_msg_type", self.cfg.head_color_msg_type
        )
        return [
            self.cfg.ros_python, str(self._worker),
            "--color-topic", color_topic,
            "--color-msg-type", color_msg_type,
            "--timeout-s", str(self.cfg.camera_grab_timeout_s),
            "--output-dir", str(out),
            "--camera-optical-frame", "",
            "--ensure-rgb",
        ]

    def _head_cloud_cmd(self, out: Path) -> list:
        # Head stereo left + organized point cloud + base<-optical TF, for grounded 3-D verify.
        # Unlike _head_cmd this needs the TF (to put cloud points in the base frame), so it passes
        # the head optical frame + base frame + tf timeout. Still no depth, --ensure-rgb kept.
        color_topic = getattr(self.cfg, "head_aligned_color_topic", self.cfg.head_left_topic)
        color_msg_type = getattr(
            self.cfg, "head_aligned_color_msg_type", self.cfg.head_color_msg_type
        )
        cmd = [
            self.cfg.ros_python, str(self._worker),
            "--color-topic", color_topic,
            "--color-msg-type", color_msg_type,
            "--timeout-s", str(self.cfg.camera_grab_timeout_s),
            "--output-dir", str(out),
            "--base-frame", getattr(self.cfg, "base_frame", "base_link"),
            "--camera-optical-frame", getattr(self.cfg, "head_optical_frame",
                                              "stereo_left_rgb_optical_link"),
            "--tf-timeout-s", str(getattr(self.cfg, "camera_tf_timeout_s", 8.0)),
            "--pointcloud-topic", getattr(self.cfg, "head_pointcloud_topic", ""),
            "--pointcloud-msg-type", getattr(self.cfg, "head_pointcloud_msg_type",
                                             "sensor_msgs/msg/PointCloud2"),
            "--ensure-rgb",
        ]
        if color_topic == self.cfg.head_left_topic:
            cmd.append("--rectify-cruzr-stereo-left")
        return cmd

    def _run_and_read(self, build_cmd, reader):
        """Grab a frame, retrying with a FRESH subprocess when an attempt yields nothing.

        At run start the head stereo camera node / DDS discovery isn't ready yet, so the first
        attempt(s) time out with no color frame — the "only the head, only at the start" symptom
        (the waist starts fast and isn't affected). Each attempt is a brand-new subprocess, so a
        retry re-runs full DDS discovery and succeeds once the stream is up. Bounded by
        ``camera_grab_retries`` (total attempts = retries + 1); the steady-state path hits on the
        first attempt with zero retry overhead.
        """
        attempts = max(1, int(getattr(self.cfg, "camera_grab_retries", 2)) + 1)
        settle = float(getattr(self.cfg, "camera_grab_retry_settle_s", 0.5))
        for attempt in range(1, attempts + 1):
            result = self._run_and_read_once(build_cmd, reader)
            if result is not None:
                return result
            if attempt < attempts:
                logger.info("[CruzrCamera] grab attempt %d/%d got no frame; "
                            "retrying with a fresh worker", attempt, attempts)
                time.sleep(settle)
        return None

    def _run_and_read_once(self, build_cmd, reader):
        """One frame-grab attempt: run the worker in a temp dir and, on a valid (ok) result, call
        ``reader(out, meta)`` INSIDE the temp dir (its files vanish on exit) and return that value.
        Returns None on any subprocess / meta failure. Shared by ``_grab`` (waist/head RGB[D]) and
        ``grab_head_frame`` (head RGB + point cloud).
        """
        if getattr(self.cfg, "resident_workers", False):
            return self._resident_grab(build_cmd, reader)
        with tempfile.TemporaryDirectory(prefix="cruzr_cam_") as tmp:
            out = Path(tmp)
            cmd = build_cmd(out)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=float(self.cfg.camera_grab_timeout_s) + 15.0,
                    env=self._worker_env(cmd),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CruzrCamera] worker run failed: %s", exc)
                return None
            if proc.returncode != 0:
                logger.warning("[CruzrCamera] worker rc=%d stderr=%s", proc.returncode, (proc.stderr or "").strip())
                return None
            meta_path = out / "meta.json"
            color_path = out / "color.npy"
            if not meta_path.exists():
                logger.warning(
                    "[CruzrCamera] worker produced no meta; stdout=%s stderr=%s",
                    (proc.stdout or "").strip()[-500:],
                    (proc.stderr or "").strip()[-500:],
                )
                return None
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                logger.warning("[CruzrCamera] worker produced invalid meta")
                return None
            if not meta.get("ok"):
                logger.warning("[CruzrCamera] worker meta not ok: %s", meta.get("reason"))
                return None
            if not color_path.exists():
                logger.warning("[CruzrCamera] worker meta ok but color.npy is missing")
                return None
            return reader(out, meta)

    def _resident_grab(self, build_cmd, reader):
        """Resident-worker grab: reuse a persistent --serve camera worker (rclpy/TF kept warm) by
        writing a 'grab <dir>' request and reading back one meta line. Returns None on any failure
        (dropping the handle so ``_run_and_read``'s retry restarts a fresh worker) — same contract as
        ``_run_and_read_once``.
        """
        worker = self._ensure_resident(build_cmd)
        with tempfile.TemporaryDirectory(prefix="cruzr_cam_") as tmp:
            out = Path(tmp)
            meta = worker.request_json(
                f"grab {out}",
                float(self.cfg.camera_grab_timeout_s) + 15.0,
                bad_output_reason="bad_meta",
            )
            if meta is None:  # worker unavailable / died → the outer retry restarts a fresh one
                return None
            if not meta.get("ok"):
                logger.warning("[CruzrCamera] resident worker meta not ok: %s", meta.get("reason"))
                return None
            if not (out / "color.npy").exists():
                logger.warning("[CruzrCamera] resident worker meta ok but color.npy is missing")
                return None
            return reader(out, meta)

    def _ensure_resident(self, build_cmd) -> ros2_worker.ResidentWorker:
        """The persistent --serve worker for ``build_cmd`` (one per camera command).

        ``--serve`` takes the output dir from each request, so the dir baked into the startup
        argv is irrelevant; a throwaway temp path keeps ``build_cmd``'s signature unchanged.
        """
        worker = self._resident.get(build_cmd)
        if worker is None:
            worker = ros2_worker.ResidentWorker(
                lambda: build_cmd(Path(tempfile.gettempdir())) + ["--serve"],
                label="[CruzrCamera]",
                env_fn=self._worker_env,
            )
            self._resident[build_cmd] = worker
        return worker

    def close(self) -> None:
        """Stop all persistent camera workers. Idempotent."""
        for worker in list(self._resident.values()):
            worker.stop()
        self._resident.clear()

    def _worker_env(self, cmd: list) -> dict[str, str]:
        """Use the configured RMW and default Fast DDS transports by default."""
        env = os.environ.copy()
        env["RMW_IMPLEMENTATION"] = str(
            getattr(self.cfg, "head_camera_rmw_implementation", "rmw_fastrtps_cpp")
            or "rmw_fastrtps_cpp"
        )
        profile = str(
            getattr(self.cfg, "camera_fastdds_profiles_file", "") or ""
        )
        if profile:
            env["FASTRTPS_DEFAULT_PROFILES_FILE"] = profile
            env["FASTDDS_DEFAULT_PROFILES_FILE"] = profile
        else:
            env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
            env.pop("FASTDDS_DEFAULT_PROFILES_FILE", None)
        env.pop("RMW_FASTRTPS_USE_QOS_FROM_XML", None)
        env.pop("CYCLONEDDS_URI", None)
        env["ROS_DOMAIN_ID"] = str(getattr(self.cfg, "head_camera_ros_domain_id", 0))
        env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = str(
            getattr(self.cfg, "head_camera_discovery_range", "SUBNET")
        )
        env.pop("ROS_LOCALHOST_ONLY", None)
        return env

    def _grab(self, build_cmd, *, want_depth: bool) -> Optional[CameraFrame]:
        def _read(out: Path, meta: dict) -> CameraFrame:
            rgb = np.load(out / "color.npy")
            depth = None
            if want_depth and meta.get("has_depth") and (out / "depth.npy").exists():
                depth = np.load(out / "depth.npy")
            k = meta.get("K")
            intrinsics = np.asarray(k, dtype=np.float64) if k else None
            tf = meta.get("tf_base_cam")
            tf_base_cam = np.asarray(tf, dtype=np.float64) if tf else None
            return CameraFrame(rgb=rgb, depth_m=depth, intrinsics=intrinsics, tf_base_cam=tf_base_cam)

        return self._run_and_read(build_cmd, _read)


class CruzrNav:
    """用 ``/usr/bin/python3`` 子进程发差速轮速指令,把底盘相对移动 dx(前进)/dyaw(左转)。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._worker = ros2_worker.worker_path("jiuwensymbiosis.adapters.cruzr.ros2.wheel_worker")
        self._move = ros2_worker.ResidentWorker(self._serve_cmd, label="[CruzrNav] move")

    def navigate_relative(self, dx: float, dy: float = 0.0, dyaw: float = 0.0, *,
                          k_rot: Optional[float] = None,
                          k_rot_slow_rad: Optional[float] = None,
                          k_fwd: Optional[float] = None) -> dict:
        """相对移动(dy 因差速无横移被忽略)。返回 {ok, reason[, yaw_reached, dist_traveled]}。

        k_rot/k_rot_slow_rad/k_fwd 非 None 时覆盖全局 base_k_*(approach 精定位用温和值,搜索大转身用全局快值)。

        失败 reason ∈ {no_odom, rotate_timeout, forward_timeout, lidar_blocked,
        wheel_worker_error, wheel_worker_failed, wheel_no_output, wheel_bad_output}。
        """
        if getattr(self.cfg, "resident_workers", False):
            res = self._resident_move_request(dx, dyaw, k_rot, k_rot_slow_rad, k_fwd)
            if res is not None:
                return res   # None → resident worker unavailable; fall through to the one-shot path
        c = self.cfg
        timeout_s = float(getattr(c, "base_move_timeout_s", 20.0))
        cmd = [
            getattr(c, "nav_python", "/usr/bin/python3"), str(self._worker),
            "--dx", str(dx), "--dyaw", str(dyaw),
            "--command-topic", getattr(c, "command_topic", "/mc/sdk/robot_command"),
            "--odom-topic", getattr(c, "odom_topic", "/mc/odom"),
            "--front-lidar-topic", getattr(c, "front_lidar_topic", "/sensor/lidar/front"),
            "--back-lidar-topic", getattr(c, "back_lidar_topic", "/sensor/lidar/back"),
            "--left-wheel-joint", getattr(c, "left_wheel_joint", "driving_wheel_left_joint"),
            "--right-wheel-joint", getattr(c, "right_wheel_joint", "driving_wheel_right_joint"),
            "--k-rot", str(k_rot if k_rot is not None else getattr(c, "base_k_rot", 0.6)),
            "--k-rot-min", str(getattr(c, "base_k_rot_min", 0.25)),
            "--k-rot-slow-rad",
            str(k_rot_slow_rad if k_rot_slow_rad is not None else getattr(c, "base_k_rot_slow_rad", 0.5)),
            "--k-fwd", str(k_fwd if k_fwd is not None else getattr(c, "base_k_fwd", 0.8)),
            "--k-fwd-min", str(getattr(c, "base_k_fwd_min", 0.25)),
            "--k-fwd-slow-m", str(getattr(c, "base_k_fwd_slow_m", 0.25)),
            "--yaw-tol", str(getattr(c, "base_yaw_tol_rad", 0.05)),
            "--pos-tol", str(getattr(c, "base_pos_tol_m", 0.05)),
            "--safe-dist", str(getattr(c, "base_safe_dist_m", 0.45)),
            "--self-floor", str(getattr(c, "base_lidar_self_floor_m", 0.25)),
            "--sector-deg", str(getattr(c, "base_lidar_sector_deg", 25.0)),
            "--timeout", str(timeout_s),
        ]
        return ros2_worker.run_once(
            cmd, timeout_s=timeout_s * 2 + 20.0, label="[CruzrNav] wheel", reason_prefix="wheel_")

    def navigate_arc(self, radius: float, dyaw: float, *, k_fwd: Optional[float] = None) -> dict:
        """沿【定曲率弧】前进(边转边走):半径 radius(m)、总转角 dyaw(rad,有符号,+=左/CCW),
        odom 测瞬时曲率伺服(不需轮径/轮距标定)、按累计转角停。用于抓取精定位把底盘"拐上"面法线线。
        返回 {ok, reason, yaw_turned, dist_traveled}。落点位置误差由随后视觉纠正的直入吸收。

        失败 reason ∈ {no_odom, arc_timeout, arc_len_cap, lidar_blocked, wheel_worker_*}。
        """
        c = self.cfg
        timeout_s = float(getattr(c, "base_move_timeout_s", 20.0))
        cmd = [
            getattr(c, "nav_python", "/usr/bin/python3"), str(self._worker),
            "--arc", "--radius", str(radius), "--arc-dyaw", str(dyaw),
            "--arc-k-curv", str(getattr(c, "arc_curv_gain", 0.5)),
            "--arc-len-safety", str(getattr(c, "arc_len_safety", 1.5)),
            "--k-fwd", str(k_fwd if k_fwd is not None else getattr(c, "arc_k_fwd", 0.6)),
            "--k-fwd-min", str(getattr(c, "arc_k_fwd_min", 0.25)),
            "--k-rot-slow-rad", str(getattr(c, "base_k_rot_slow_rad", 0.5)),
            "--command-topic", getattr(c, "command_topic", "/mc/sdk/robot_command"),
            "--odom-topic", getattr(c, "odom_topic", "/mc/odom"),
            "--front-lidar-topic", getattr(c, "front_lidar_topic", "/sensor/lidar/front"),
            "--back-lidar-topic", getattr(c, "back_lidar_topic", "/sensor/lidar/back"),
            "--left-wheel-joint", getattr(c, "left_wheel_joint", "driving_wheel_left_joint"),
            "--right-wheel-joint", getattr(c, "right_wheel_joint", "driving_wheel_right_joint"),
            "--safe-dist", str(getattr(c, "base_safe_dist_m", 0.45)),
            "--self-floor", str(getattr(c, "base_lidar_self_floor_m", 0.25)),
            "--sector-deg", str(getattr(c, "base_lidar_sector_deg", 25.0)),
            "--timeout", str(timeout_s),
        ]
        return ros2_worker.run_once(
            cmd, timeout_s=timeout_s * 2 + 20.0, label="[CruzrNav] arc", reason_prefix="wheel_")

    def _serve_cmd(self) -> list[str]:
        """CLI for a persistent --serve wheel worker. Global base_k_* are the startup defaults; each
        request overrides k_rot/k_fwd/k_rot_slow inline (navigate_relative's gentle-approach values).
        """
        c = self.cfg
        return [
            getattr(c, "nav_python", "/usr/bin/python3"), str(self._worker), "--serve",
            "--command-topic", getattr(c, "command_topic", "/mc/sdk/robot_command"),
            "--odom-topic", getattr(c, "odom_topic", "/mc/odom"),
            "--front-lidar-topic", getattr(c, "front_lidar_topic", "/sensor/lidar/front"),
            "--back-lidar-topic", getattr(c, "back_lidar_topic", "/sensor/lidar/back"),
            "--left-wheel-joint", getattr(c, "left_wheel_joint", "driving_wheel_left_joint"),
            "--right-wheel-joint", getattr(c, "right_wheel_joint", "driving_wheel_right_joint"),
            "--k-rot", str(getattr(c, "base_k_rot", 0.6)),
            "--k-rot-min", str(getattr(c, "base_k_rot_min", 0.25)),
            "--k-rot-slow-rad", str(getattr(c, "base_k_rot_slow_rad", 0.5)),
            "--k-fwd", str(getattr(c, "base_k_fwd", 0.8)),
            "--k-fwd-min", str(getattr(c, "base_k_fwd_min", 0.25)),
            "--k-fwd-slow-m", str(getattr(c, "base_k_fwd_slow_m", 0.25)),
            "--yaw-tol", str(getattr(c, "base_yaw_tol_rad", 0.05)),
            "--pos-tol", str(getattr(c, "base_pos_tol_m", 0.05)),
            "--safe-dist", str(getattr(c, "base_safe_dist_m", 0.45)),
            "--self-floor", str(getattr(c, "base_lidar_self_floor_m", 0.25)),
            "--sector-deg", str(getattr(c, "base_lidar_sector_deg", 25.0)),
            "--timeout", str(getattr(c, "base_move_timeout_s", 20.0)),
        ]

    def _resident_move_request(self, dx, dyaw, k_rot, k_rot_slow, k_fwd) -> Optional[dict]:
        """Send one 'dx dyaw k_rot k_fwd k_rot_slow' request to the persistent --serve wheel worker and
        read back its JSON result. Returns None (→ caller falls back to the one-shot path) when the
        worker can't be started or dies mid-request.
        """
        c = self.cfg
        krot = k_rot if k_rot is not None else getattr(c, "base_k_rot", 0.6)
        kfwd = k_fwd if k_fwd is not None else getattr(c, "base_k_fwd", 0.8)
        krot_slow = k_rot_slow if k_rot_slow is not None else getattr(c, "base_k_rot_slow_rad", 0.5)
        timeout_s = float(getattr(c, "base_move_timeout_s", 20.0))
        return self._move.request_json(
            f"{dx} {dyaw} {krot} {kfwd} {krot_slow}",
            timeout_s * 2 + 20.0,
            bad_output_reason="wheel_bad_output",
        )

    def close(self) -> None:
        """Stop the persistent move worker. Idempotent."""
        self._move.stop()

    def _spin_cmd(self, direction: float) -> list[str]:
        c = self.cfg
        return [
            getattr(c, "nav_python", "/usr/bin/python3"), str(self._worker),
            "--spin", "--dyaw", str(1.0 if direction >= 0 else -1.0),
            "--command-topic", getattr(c, "command_topic", "/mc/sdk/robot_command"),
            "--odom-topic", getattr(c, "odom_topic", "/mc/odom"),
            "--front-lidar-topic", getattr(c, "front_lidar_topic", "/sensor/lidar/front"),
            "--back-lidar-topic", getattr(c, "back_lidar_topic", "/sensor/lidar/back"),
            "--left-wheel-joint", getattr(c, "left_wheel_joint", "driving_wheel_left_joint"),
            "--right-wheel-joint", getattr(c, "right_wheel_joint", "driving_wheel_right_joint"),
            "--k-rot", str(getattr(c, "search_spin_wheel_rs", getattr(c, "base_k_rot", 0.6))),
            "--yaw-tol", str(getattr(c, "base_yaw_tol_rad", 0.05)),
            "--spin-max-rad", str(getattr(c, "search_spin_max_rad", 6.6)),
            "--timeout", str(getattr(c, "search_spin_timeout_s", 40.0)),
        ]

    def start_spin(self, direction: float = 1.0) -> subprocess.Popen:
        """Launch a NON-blocking continuous-search spin worker (constant speed, interruptible).

        Returns the ``Popen`` handle; poll it with ``spin_running`` and halt it with ``stop_spin``.
        The worker self-bounds (``search_spin_max_rad`` / ``search_spin_timeout_s``) and stops on
        stdin EOF, so a crashed parent can't leave the base spinning.
        """
        return subprocess.Popen(
            self._spin_cmd(direction), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    @staticmethod
    def spin_running(proc: subprocess.Popen) -> bool:
        """True while the spin worker is still rotating (has not exited on its own)."""
        return proc.poll() is None

    @staticmethod
    def stop_spin(proc: subprocess.Popen) -> dict:
        """Halt the spin worker and collect its result. Sends the stdin 'stop' sentinel (clean
        wheel stop) when still running, else just drains the completed run. Falls back to
        SIGTERM (handled → clean stop) then kill, so the base is never left commanded.
        """
        return ros2_worker.stop_and_collect(
            proc, label="[CruzrNav] spin", kind="spin", empty_result={"ok": True, "yaw_turned": 0.0})

    def _forward_cmd(self, *, k_fwd: Optional[float] = None, fwd_max_m: Optional[float] = None) -> list[str]:
        c = self.cfg
        kf = k_fwd if k_fwd is not None else getattr(c, "approach_drive_wheel_rs", 0.5)
        fm = fwd_max_m if fwd_max_m is not None else getattr(c, "approach_drive_max_m", 2.5)
        return [
            getattr(c, "nav_python", "/usr/bin/python3"), str(self._worker),
            "--forward",
            "--command-topic", getattr(c, "command_topic", "/mc/sdk/robot_command"),
            "--odom-topic", getattr(c, "odom_topic", "/mc/odom"),
            "--front-lidar-topic", getattr(c, "front_lidar_topic", "/sensor/lidar/front"),
            "--back-lidar-topic", getattr(c, "back_lidar_topic", "/sensor/lidar/back"),
            "--left-wheel-joint", getattr(c, "left_wheel_joint", "driving_wheel_left_joint"),
            "--right-wheel-joint", getattr(c, "right_wheel_joint", "driving_wheel_right_joint"),
            "--k-fwd", str(kf),
            "--k-steer", str(getattr(c, "approach_drive_steer_gain", 0.6)),
            "--steer-max", str(getattr(c, "approach_drive_steer_max_rs", 0.4)),
            "--fwd-max-m", str(fm),
            "--safe-dist", str(getattr(c, "base_safe_dist_m", 0.45)),
            "--self-floor", str(getattr(c, "base_lidar_self_floor_m", 0.25)),
            "--sector-deg", str(getattr(c, "base_lidar_sector_deg", 25.0)),
            "--timeout", str(getattr(c, "approach_drive_timeout_s", 30.0)),
        ]

    def start_drive(self, *, k_fwd: Optional[float] = None, fwd_max_m: Optional[float] = None) -> subprocess.Popen:
        """Launch a NON-blocking continuous forward-creep worker (constant speed, interruptible).

        Returns the ``Popen`` handle; poll it with ``drive_running`` and halt it with ``stop_drive``.
        The worker self-bounds (``approach_drive_max_m`` / ``approach_drive_timeout_s``) and stops on
        stdin EOF, so a crashed parent can't leave the base driving. ``k_fwd`` / ``fwd_max_m`` override the
        creep speed / self-bound for the fine grasp servo (slower + a tighter cap); ``None`` = config default.
        """
        return subprocess.Popen(
            self._forward_cmd(k_fwd=k_fwd, fwd_max_m=fwd_max_m), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    @staticmethod
    def drive_running(proc: subprocess.Popen) -> bool:
        """True while the forward worker is still driving (has not exited on its own)."""
        return proc.poll() is None

    @staticmethod
    def steer_drive(proc: subprocess.Popen, bearing_rad: float) -> None:
        """Feed a live head bearing (rad, + = target left) to the running forward worker so it curves
        toward the target via differential wheels. Best-effort: a closed stdin / dead worker is
        ignored — the drive self-bounds and the next ``stop_drive`` still collects its result.
        """
        try:
            if proc.stdin is not None and proc.poll() is None:
                proc.stdin.write(f"{float(bearing_rad):.4f}\n")
                proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001 — broken pipe / closed stdin → ignore, drive self-bounds
            logger.debug("[CruzrNav] steer_drive ignored (%s)", exc)

    @staticmethod
    def hold_drive(proc: subprocess.Popen) -> None:
        """Tell the running forward worker to PAUSE the wheels (hold position) because the head lost
        the anchor — so it stops creeping blind on the last bearing. The next ``steer_drive`` resumes
        it. Best-effort like ``steer_drive``: a closed stdin / dead worker is ignored.
        """
        try:
            if proc.stdin is not None and proc.poll() is None:
                proc.stdin.write("hold\n")
                proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001 — broken pipe / closed stdin → ignore, drive self-bounds
            logger.debug("[CruzrNav] hold_drive ignored (%s)", exc)

    @staticmethod
    def stop_drive(proc: subprocess.Popen) -> dict:
        """Halt the forward worker and collect its result. Sends the stdin 'stop' sentinel (clean
        wheel stop) when still running, else just drains the completed run. Falls back to SIGTERM
        (handled → clean stop) then kill, so the base is never left commanded.
        """
        return ros2_worker.stop_and_collect(
            proc, label="[CruzrNav] drive", kind="drive", empty_result={"ok": True, "dist_traveled": 0.0})
