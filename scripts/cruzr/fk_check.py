# coding: utf-8
"""M1 离线验证：自实现 FK 对齐真机 TF + 查证掌法向轴 / lifter 接口。只读不动机器人。

需 source ROS + Cruzr_ws，用 /usr/bin/python3 运行：
    /usr/bin/python3 scripts/cruzr/fk_check.py
"""
from __future__ import annotations

import logging
import time

import numpy as np

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.kinematics.fk import fk_chain
from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

logger = logging.getLogger(__name__)


def main() -> int:
    import rclpy
    import tf2_ros
    from mc_state_msgs.msg import RobotState
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import Buffer, TransformListener

    cfg = CruzrConfig()
    rclpy.init()
    node = rclpy.create_node("cruzr_fk_check")
    buf = Buffer()
    TransformListener(buf, node)
    state = {"q": {}}
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    qos.history = HistoryPolicy.KEEP_LAST

    def _on_state(msg):
        state["q"] = {str(n): float(p) for n, p in zip(msg.joint_states.name, msg.joint_states.position)}

    node.create_subscription(RobotState, "/mc/sdk/robot_state", _on_state, qos)

    chain = parse_chain(cfg.urdf_path, "base_link", cfg.left_arm_leaf)
    deadline = time.time() + 10.0
    tf = None
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buf.lookup_transform("base_link", cfg.left_arm_leaf, rclpy.time.Time())
        except tf2_ros.TransformException:
            tf = None
        if state["q"] and tf is not None:
            break
    if not state["q"] or tf is None:
        logger.error("missing joint_states or TF")
        return 1

    tf = fk_chain(chain, state["q"])
    p_fk = tf[:3, 3]
    t = tf.transform.translation
    p_tf = np.array([t.x, t.y, t.z])
    err = float(np.linalg.norm(p_fk - p_tf))
    logger.info("FK  base->%s (m): %s", cfg.left_arm_leaf, p_fk)
    logger.info("TF  base->%s (m): %s", cfg.left_arm_leaf, p_tf)
    logger.info("position error (m): %.4f  (%s)", err, "OK" if err < 0.01 else "CHECK")
    for ax, name in zip(np.eye(3), ["x", "y", "z"]):
        logger.info("tool local %s-axis in base: %s", name, tf[:3, :3] @ ax)
    lifter_names = ("lifter_pitch_1_joint", "lifter_pitch_2_joint", "lifter_pitch_3_joint")
    logger.info("lifter joints: %s", {k: round(state['q'].get(k, 0.0), 4) for k in lifter_names})
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
