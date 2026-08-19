# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr 腰部相机外参标定：从机器人 TF 树读取 base->camera_optical 并写入标定 JSON。

``tf_base_cam`` 是 ``base_link -> waist_front_rgbd_color_optical_frame`` 的 4x4 变换
（平移换算为 mm，与 ``camera.project_to_base`` 一致）。该 TF 链路经过 lifter / waist_yaw
/ torso 等可动关节，所以外参对应**当前本体姿态**——抓取姿态改变后需要重新运行本脚本。

可顺带从 ``camera_info`` 读取内参写入 JSON（仅作 camera_info 不可用时的兜底）。

需在已 source ROS + Cruzr_ws 的 shell 里用 ROS 解释器运行（含 rclpy / tf2_ros）：

    source /opt/ros/<distro>/setup.bash
    source <Cruzr_ws>/install/setup.bash
    /usr/bin/python3 scripts/cruzr/calibrate_extrinsic.py \
        --output configs/cruzr/cruzr_camera_calib.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _quat_to_rot(x: float, y: float, z: float, w: float) -> list:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix (row-major lists)."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def main(argv=None) -> int:
    """Look up base->camera via tf2, optionally grab intrinsics, write calib JSON."""
    parser = argparse.ArgumentParser(description="Cruzr waist-camera extrinsic calibration from TF.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="waist_front_rgbd_color_optical_frame")
    parser.add_argument(
        "--camera-info-topic",
        default="/sensor/camera/waist_front_rgbd/color/info",
        help="CameraInfo topic for intrinsics; empty string to skip.",
    )
    parser.add_argument("--output", default="configs/cruzr/cruzr_camera_calib.json")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=None)
    node = rclpy.create_node("cruzr_calibrate_extrinsic")
    buf = Buffer()
    TransformListener(buf, node)

    # --- intrinsics from CameraInfo (latched -> TRANSIENT_LOCAL) ------------
    intrinsics = None
    if args.camera_info_topic:
        from sensor_msgs.msg import CameraInfo

        info_qos = QoSProfile(depth=1)
        info_qos.reliability = ReliabilityPolicy.RELIABLE
        info_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        info_qos.history = HistoryPolicy.KEEP_LAST
        state = {"k": None}
        node.create_subscription(
            CameraInfo, args.camera_info_topic, lambda m: state.__setitem__("k", list(m.k)), info_qos
        )

    # --- spin until the TF lookup succeeds AND intrinsics arrive -----------
    deadline = time.time() + float(args.timeout_s)
    tf = None
    last_exc = None
    want_info = bool(args.camera_info_topic)
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if tf is None:
            try:
                tf = buf.lookup_transform(args.base_frame, args.camera_frame, rclpy.time.Time())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        have_info = (not want_info) or (state.get("k") is not None)
        if tf is not None and have_info:
            break

    if tf is None:
        logger.error("TF lookup %s -> %s failed: %s", args.base_frame, args.camera_frame, last_exc)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    t = tf.transform.translation
    q = tf.transform.rotation
    rot = _quat_to_rot(q.x, q.y, q.z, q.w)
    # Translation metres -> mm (project_to_base composes mm-valued transforms).
    tf_base_cam = [
        [rot[0][0], rot[0][1], rot[0][2], t.x * 1000.0],
        [rot[1][0], rot[1][1], rot[1][2], t.y * 1000.0],
        [rot[2][0], rot[2][1], rot[2][2], t.z * 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    if args.camera_info_topic and state.get("k") and len(state["k"]) == 9:
        k = state["k"]
        intrinsics = [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]]

    payload = {
        "_note": (
            f"tf_base_cam = {args.base_frame} -> {args.camera_frame}, from robot TF (tf2), "
            "translation in mm. Chain crosses lifter/waist_yaw/torso joints -> valid for the "
            "current body pose; re-run after the pose changes. intrinsics from camera_info."
        ),
        "tf_base_cam": tf_base_cam,
    }
    if intrinsics is not None:
        payload["intrinsics"] = intrinsics

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out)
    logger.info("  translation (m): %.4f %.4f %.4f", t.x, t.y, t.z)
    logger.info("  intrinsics: %s", "from camera_info" if intrinsics else "skipped")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
