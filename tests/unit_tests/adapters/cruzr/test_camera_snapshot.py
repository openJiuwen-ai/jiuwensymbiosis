"""Exercise the real worker loop with DDS delivery interleaved with TF polling."""

import json
import sys
import types
from types import SimpleNamespace as NS

import numpy as np

from jiuwensymbiosis.adapters.cruzr.ros2 import camera_worker


def test_waiting_for_tf_cannot_replace_already_decoded_rgb_or_depth(monkeypatch, tmp_path):
    callbacks = {}
    delivered = [False]
    destroyed = []

    def message(depth=False, newer=False):
        data = (
            np.full((2, 2), 2000 if newer else 1000, np.uint16)
            if depth
            else np.full((2, 2, 3), 200 if newer else 30, np.uint8)
        )
        return NS(
            header=NS(stamp=NS(sec=10, nanosec=100_000_000 if newer else 0)),
            height=2,
            width=2,
            encoding="16UC1" if depth else "rgb8",
            step=4 if depth else 6,
            data=data.tobytes(),
            is_bigendian=False,
        )

    class Node:
        def create_subscription(self, cls, topic, callback, qos):
            callbacks[topic] = callback
            callback(
                NS(k=[100.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0])
                if topic == "info"
                else message(topic == "depth")
            )
            return topic

        def destroy_subscription(self, token):
            destroyed.append(token)

        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=10_200_000_000))

        def destroy_node(self):
            destroyed.append("node")

    class Buffer:
        def lookup_transform(self, base, camera, query_time):
            if not delivered[0]:
                raise LookupError("TF not received yet")
            return NS(transform=NS(translation=NS(x=0.0, y=0.0, z=0.0), rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))

    def spin_once(*args, **kwargs):
        callbacks["color"](message(newer=True))
        callbacks["depth"](message(depth=True, newer=True))
        delivered[0] = True

    ros = types.ModuleType("rclpy")
    ros.init = lambda **kw: None
    ros.shutdown = lambda: destroyed.append("context")
    ros.create_node = lambda name: Node()
    ros.spin_once = spin_once
    ros.time = types.ModuleType("rclpy.time")
    ros.time.Time = type("Time", (), {"from_msg": staticmethod(lambda stamp: stamp.sec * 10**9 + stamp.nanosec)})
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = lambda **kw: NS(**kw)
    qos.DurabilityPolicy = NS(VOLATILE=0, TRANSIENT_LOCAL=1)
    qos.ReliabilityPolicy = NS(BEST_EFFORT=0, RELIABLE=1)
    qos.HistoryPolicy = NS(KEEP_LAST=0)
    sensor = types.ModuleType("sensor_msgs")
    sensor.msg = types.ModuleType("sensor_msgs.msg")
    sensor.msg.CameraInfo = object
    sensor.msg.PointCloud2 = object
    tf = types.ModuleType("tf2_ros")
    tf.Buffer = Buffer
    tf.TransformListener = lambda *args: None
    for name, module in {
        "rclpy": ros,
        "rclpy.time": ros.time,
        "rclpy.qos": qos,
        "sensor_msgs": sensor,
        "sensor_msgs.msg": sensor.msg,
        "tf2_ros": tf,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(camera_worker, "_import_msg", lambda name: object)
    assert (
        camera_worker.main(
            [
                "--color-topic",
                "color",
                "--depth-topic",
                "depth",
                "--camera-info-topic",
                "info",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert delivered[0]
    assert np.all(np.load(tmp_path / "color.npy") == 30)
    assert np.all(np.load(tmp_path / "depth.npy") == 1.0)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["color_stamp_ns"] == meta["depth_stamp_ns"] == 10_000_000_000
    assert meta["depth_sync_delta_ms"] == 0 and meta["tf_used_latest"] is False
    assert set(destroyed) == {"color", "depth", "info", "node", "context"}
