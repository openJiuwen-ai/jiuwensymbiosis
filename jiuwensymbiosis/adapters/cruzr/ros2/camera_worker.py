# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr 腰部 RGBD 相机抓帧 worker（用 /usr/bin/python3 执行）。

订阅 color/depth/camera_info 各一帧，解码后落盘 color.npy / depth.npy / meta.json。
rclpy 仅在 ``main`` 内、解析参数之后 import，因此 ``--help`` 在 conda 下也可用。

``--serve`` 常驻模式：init rclpy + 建订阅一次，然后循环读 stdin 的 "grab <outdir>" 请求，
每条抓一帧落盘到该 outdir、打一行 meta JSON，跨请求保活 node/订阅/TF 缓冲，免去每次抓帧的
新进程 + DDS 发现 + TF 重填冷启动（``resident_workers`` 开启时用；默认走一次性路径）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import select
import sys
from pathlib import Path
from typing import Any, Optional


def _import_msg(type_str: str):
    """``"shm_msgs/msg/Image1m"`` -> 消息类；失败回退 ``sensor_msgs/msg/Image``。"""
    for candidate in (type_str, "sensor_msgs/msg/Image"):
        try:
            pkg, _msg, name = candidate.split("/")
            mod = importlib.import_module(f"{pkg}.msg")
            return getattr(mod, name)
        except Exception:  # noqa: BLE001
            continue
    raise ImportError(f"cannot import image message type {type_str!r}")


def _quat_to_rot(x: float, y: float, z: float, w: float) -> list:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix (row-major lists)."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _stamp_ns(msg: Any) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _rectify_cruzr_stereo_left(rgb):
    """Map the vendor head RGB into the rectified left-camera projection used by the cloud."""
    import cv2
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[:2] != (720, 1280) or rgb.shape[2] != 3:
        raise ValueError(f"expected Cruzr stereo RGB 720x1280x3, got {rgb.shape}")

    half = np.ascontiguousarray(rgb[::2, ::2])
    k = np.array([
        [218.46993685643483, 0.0, 320.2900599103507],
        [0.0, 217.28944658903572, 182.1195864900839],
        [0.0, 0.0, 1.0],
    ])
    d = np.array([
        -0.06877947211660146,
        0.0007721139992952977,
        -0.001185811100523145,
        -0.0016183623674533986,
        0.0,
    ])
    r = np.array([
        [0.9974925941474858, 0.012441576760204938, -0.06966872891505414],
        [-0.012494853600762267, 0.9999218821541679, -0.00032897228131833724],
        [0.06965919361013943, 0.0011986479826396406, 0.9975701278549789],
    ])
    p = np.array([
        [246.05646885806817, 0.0, 380.4923095703125],
        [0.0, 246.05646885806817, 175.3212432861328],
        [0.0, 0.0, 1.0],
    ])
    map_x, map_y = cv2.initUndistortRectifyMap(k, d, r, p, (640, 360), cv2.CV_32FC1)
    return cv2.remap(half, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _emit(meta: dict) -> None:
    # stdout IS this worker's protocol (parsed by ros2/worker.py), not logging: a logger
    # would prefix/redirect the line and the parent would find no result.
    sys.stdout.write(json.dumps(meta) + "\n")
    sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cruzr waist RGBD frame-grab worker.")
    p.add_argument("--color-topic", required=True)
    p.add_argument("--depth-topic", default="")
    p.add_argument("--camera-info-topic", default="")
    p.add_argument("--color-msg-type", default="shm_msgs/msg/Image1m")
    p.add_argument("--depth-msg-type", default="shm_msgs/msg/Image1m")
    # Optional organized point cloud (head stereo). Empty = don't subscribe (waist path unchanged).
    p.add_argument("--pointcloud-topic", default="")
    p.add_argument("--pointcloud-msg-type", default="sensor_msgs/msg/PointCloud2")
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--timeout-s", type=float, default=8.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--camera-optical-frame", default="waist_front_rgbd_color_optical_frame")
    p.add_argument("--tf-timeout-s", type=float, default=2.0,
                   help="Poll this long for the live base->camera TF before giving up.")
    p.add_argument("--ensure-rgb", action="store_true",
                   help="If the decoded frame is 2-D (grayscale), stack it to HxWx3.")
    p.add_argument("--rectify-cruzr-stereo-left", action="store_true",
                   help="Rectify the vendor head RGB into the organized stereo-cloud projection.")
    p.add_argument("--sync-tolerance-s", type=float, default=0.12,
                   help="Maximum color/point-cloud timestamp delta when a cloud is requested.")
    p.add_argument("--serve", action="store_true",
                   help="Resident mode: init rclpy + subscriptions once, then loop reading 'grab <outdir>' "
                        "request lines from stdin, grab one frame into <outdir> per line and emit one meta JSON "
                        "each. 'stop'/EOF exits. Keeps node/TF warm across grabs (avoids per-grab cold start).")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.output_dir)

    # rclpy + camera.py 仅在此处 import（需 ROS 环境 / numpy）。
    import numpy as np
    import rclpy
    import rclpy.time
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    # ``decode_image_msg`` 以“文件路径”方式从通用的 ``jiuwensymbiosis/ros2/image_decode.py``
    # 加载，绕过 ``jiuwensymbiosis`` 包的 ``__init__``（其 eager 导入 openjiuwen，而本 worker
    # 运行于 ``/usr/bin/python3``，未安装 openjiuwen）。``image_decode`` 仅依赖
    # numpy+stdlib，可独立加载，故无需把仓库根加入 sys.path 或 import 整个包。
    # 路径按本文件位置（adapters/cruzr/ros2/）上溯到包根，随目录搬迁必须同步改。
    _decode_path = Path(__file__).resolve().parents[3] / "ros2" / "image_decode.py"
    _spec = importlib.util.spec_from_file_location("cruzr_image_decode", _decode_path)
    _decode_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_decode_mod)
    decode_image_msg = _decode_mod.decode_image_msg

    # Sensor streams (color/depth/cloud) are high-rate BEST_EFFORT/VOLATILE.
    # The head compressed color topic is published RELIABLE, but a BEST_EFFORT
    # subscriber is still compatible with a RELIABLE publisher and matches faster
    # on the fresh short-lived per-grab worker (no reliable handshake) — the
    # RELIABLE path stalled cross-machine while the BEST_EFFORT waist path did not.
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    qos.history = HistoryPolicy.KEEP_LAST

    # CameraInfo is latched config data published RELIABLE/TRANSIENT_LOCAL and
    # rarely republished. A VOLATILE subscriber would never see the latched
    # sample, so intrinsics must use a matching TRANSIENT_LOCAL profile.
    info_qos = QoSProfile(depth=1)
    info_qos.reliability = ReliabilityPolicy.RELIABLE
    info_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    info_qos.history = HistoryPolicy.KEEP_LAST

    rclpy.init(args=None)
    node = rclpy.create_node("jiuwensymbiosis_cruzr_camera_worker")
    state: dict[str, Any] = {"color": None, "depth": None, "K": None, "cloud": None}

    # Set up TF buffer for live base->camera transform lookup.
    from tf2_ros import Buffer, TransformListener
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    # Message types imported once; the sensor SUBSCRIPTIONS are created per-grab in _grab_once and
    # destroyed right after (a persistent sensor subscription degrades the SDK camera stream over a
    # long-lived worker — cruzr-inprocess-dds-shm). node / rclpy / TF stay resident for the savings.
    color_msg_cls = _import_msg(args.color_msg_type)
    depth_msg_cls = _import_msg(args.depth_msg_type) if args.depth_topic else None
    want_cloud = bool(args.pointcloud_topic)
    from sensor_msgs.msg import CameraInfo, PointCloud2

    def _grab_once(dst: Path) -> dict:
        """Create the sensor subscriptions, grab one frame, then DESTROY them (never persistent — that
        is what degrades the SDK camera stream over a long-lived worker). node/rclpy/TF stay resident.
        """
        state["color"] = None
        state["depth"] = None
        state["K"] = None
        state["cloud"] = None
        subs = [node.create_subscription(color_msg_cls, args.color_topic, lambda m: state.__setitem__("color", m), qos)]
        if depth_msg_cls is not None:
            subs.append(node.create_subscription(depth_msg_cls, args.depth_topic,
                                                 lambda m: state.__setitem__("depth", m), qos))
        if args.camera_info_topic:
            subs.append(node.create_subscription(CameraInfo, args.camera_info_topic,
                                                 lambda m: state.__setitem__("K", list(m.k)), info_qos))
        if want_cloud:
            subs.append(node.create_subscription(PointCloud2, args.pointcloud_topic,
                                                 lambda m: state.__setitem__("cloud", m), qos))
        try:
            return _grab_body(dst)
        finally:
            for s in subs:
                node.destroy_subscription(s)

    def _grab_body(dst: Path) -> dict:
        """Wait for the frame(s) on the freshly-created subscriptions, decode + persist, return meta."""
        dst.mkdir(parents=True, exist_ok=True)

        def _persist(meta: dict) -> dict:
            (dst / "meta.json").write_text(json.dumps(meta))
            return meta

        deadline = node.get_clock().now().nanoseconds + int(args.timeout_s * 1e9)
        while state["color"] is None and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        # color 拿到后，再给 depth/camera_info/pointcloud 一个短窗口（尽力而为）。
        grace = node.get_clock().now().nanoseconds + int(0.5 * 1e9)
        while ((state["depth"] is None or state["K"] is None or (want_cloud and state["cloud"] is None))
               and node.get_clock().now().nanoseconds < grace):
            rclpy.spin_once(node, timeout_sec=0.05)

        if want_cloud:
            tolerance_ns = max(0, int(args.sync_tolerance_s * 1e9))
            while node.get_clock().now().nanoseconds < deadline:
                color_stamp_ns = _stamp_ns(state["color"])
                cloud_stamp_ns = _stamp_ns(state["cloud"])
                if color_stamp_ns and cloud_stamp_ns and abs(color_stamp_ns - cloud_stamp_ns) <= tolerance_ns:
                    break
                rclpy.spin_once(node, timeout_sec=0.05)

            color_stamp_ns = _stamp_ns(state["color"])
            cloud_stamp_ns = _stamp_ns(state["cloud"])
            sync_delta_ns = abs(color_stamp_ns - cloud_stamp_ns) if color_stamp_ns and cloud_stamp_ns else None
            if sync_delta_ns is None or sync_delta_ns > tolerance_ns:
                state["cloud"] = None
        else:
            color_stamp_ns = _stamp_ns(state["color"])
            cloud_stamp_ns = 0
            sync_delta_ns = None

        if state["color"] is None:
            return _persist({"ok": False, "reason": "no_color_frame", "has_depth": False, "K": None})

        if args.color_msg_type.endswith("/CompressedImage"):
            import cv2
            encoded = np.frombuffer(bytes(state["color"].data), dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                return _persist({"ok": False, "reason": "compressed_decode_failed", "has_depth": False, "K": None})
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            try:
                rgb = decode_image_msg(state["color"], depth_scale=args.depth_scale)
            except ValueError as exc:
                return _persist({"ok": False, "reason": f"unsupported_encoding:{exc}",
                                 "has_depth": False, "K": None})
        if args.rectify_cruzr_stereo_left:
            try:
                rgb = _rectify_cruzr_stereo_left(rgb)
            except ValueError as exc:
                return _persist({"ok": False, "reason": f"rectify_failed:{exc}", "has_depth": False, "K": None})
        if args.ensure_rgb and rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        np.save(dst / "color.npy", rgb)

        # Look up the live base->camera TF at grab time. The TransformListener
        # buffer needs time to accumulate /tf + /tf_static; a single attempt right
        # after the color frame races and often fails, which would silently fall
        # back to the (pose-dependent) static calib. So POLL until the transform
        # is available, spinning to fill the buffer, up to tf_timeout_s.
        tf_base_cam = None
        tf_used_latest = False
        if args.camera_optical_frame:
            tf_time = rclpy.time.Time()
            if state["cloud"] is not None and _stamp_ns(state["cloud"]):
                tf_time = rclpy.time.Time.from_msg(state["cloud"].header.stamp)
            tf_deadline = node.get_clock().now().nanoseconds + int(args.tf_timeout_s * 1e9)
            while node.get_clock().now().nanoseconds < tf_deadline:
                tr = None
                # A fresh worker cannot receive dynamic TF samples older than its
                # subscription.  Prefer the cloud timestamp, then use the newest
                # transform once available; the head is stationary during a grab.
                for query_time, used_latest in (
                    (tf_time, False),
                    (rclpy.time.Time(), True),
                ):
                    try:
                        tr = tf_buffer.lookup_transform(
                            args.base_frame, args.camera_optical_frame, query_time
                        )
                        tf_used_latest = used_latest
                        break
                    except Exception:  # noqa: BLE001 — try latest / keep spinning
                        continue
                if tr is None:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    continue
                t = tr.transform.translation
                q = tr.transform.rotation
                rot = _quat_to_rot(q.x, q.y, q.z, q.w)
                tf_base_cam = [
                    [rot[0][0], rot[0][1], rot[0][2], t.x * 1000.0],
                    [rot[1][0], rot[1][1], rot[1][2], t.y * 1000.0],
                    [rot[2][0], rot[2][1], rot[2][2], t.z * 1000.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                break

        has_depth = False
        if state["depth"] is not None:
            try:
                depth = decode_image_msg(state["depth"], depth_scale=args.depth_scale)
                np.save(dst / "depth.npy", depth)
                has_depth = True
            except ValueError:
                has_depth = False

        # Organized point cloud (H×W×3 metres, same optical frame as the RGB). Parse the raw
        # PointCloud2 buffer directly (x,y,z float32 at offsets 0/4/8; 4th float = padding). NaN
        # holes on untextured faces are kept as-is; any parse failure just drops the cloud (the
        # consumer degrades to bearing-only). is_bigendian honoured; assumes float32 x/y/z.
        has_cloud = False
        cloud_h = cloud_w = None
        if state["cloud"] is not None:
            try:
                msg = state["cloud"]
                cloud_h, cloud_w = int(msg.height), int(msg.width)
                step_floats = int(msg.point_step) // 4
                dt = ">f4" if getattr(msg, "is_bigendian", False) else "<f4"
                buf = np.frombuffer(bytes(msg.data), dtype=dt).reshape(cloud_h, cloud_w, step_floats)
                np.save(dst / "cloud.npy", np.ascontiguousarray(buf[:, :, :3]))
                has_cloud = True
            except Exception:  # noqa: BLE001 — malformed cloud → drop it, caller degrades
                has_cloud = False
                cloud_h = cloud_w = None

        k_mat = None
        if state["K"] is not None and len(state["K"]) == 9:
            k = state["K"]
            k_mat = [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]]

        return _persist({"ok": True, "has_depth": has_depth, "K": k_mat,
                         "tf_base_cam": tf_base_cam,
                         "tf_used_latest": tf_used_latest,
                         "has_cloud": has_cloud, "cloud_frame": args.camera_optical_frame or None,
                         "cloud_h": cloud_h, "cloud_w": cloud_w,
                         "color_stamp_ns": color_stamp_ns, "cloud_stamp_ns": cloud_stamp_ns or None,
                         "sync_delta_ms": None if sync_delta_ns is None else sync_delta_ns / 1e6,
                         "width": int(rgb.shape[1]), "height": int(rgb.shape[0])})

    try:
        # ---- resident serve mode: idle until a "grab <outdir>" line, grab one frame, emit meta, loop.
        # "stop"/EOF exits. node/rclpy/TF stay warm across grabs; sensor subscriptions are created and
        # destroyed per-grab inside _grab_once (no persistent stream that degrades the SDK).
        if args.serve:
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                rclpy.spin_once(node, timeout_sec=0.0)   # keep TF warm while idle
                if not r:
                    continue
                line = sys.stdin.readline()
                if line == "" or line.strip() == "stop":   # EOF (parent gone) or sentinel
                    break
                toks = line.split()
                if not toks:
                    continue
                _emit(_grab_once(Path(toks[-1])))   # last token = requested output dir
            return 0

        _emit(_grab_once(out))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
