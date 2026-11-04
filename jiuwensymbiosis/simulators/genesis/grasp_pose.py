"""Genesis candidate generation. All spatial inputs and outputs use metres.

The caller declares the output frame by providing its rigid transformation.
Only the configured point source is evaluated, and inference never moves a robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rigid(matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape[-2:] != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[..., 3, :], (0, 0, 0, 1), atol=1e-5):
        raise ValueError("invalid rigid transform")
    rotation = m[..., :3, :3]
    if not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=2e-3) or not np.allclose(
        np.linalg.det(rotation), 1, atol=2e-3
    ):
        raise ValueError("invalid rotation")
    return m


def rgbd_points(depth_m, target_mask, intrinsics, frame_from_camera):
    depth = np.asarray(depth_m)
    mask = np.asarray(target_mask)
    k = np.asarray(intrinsics)
    frame = rigid(frame_from_camera)
    if depth.ndim != 2 or mask.dtype != bool or mask.shape != depth.shape or frame.shape != (4, 4):
        raise ValueError("require aligned depth and boolean target mask from the same capture")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("invalid rectified camera intrinsics")
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    xyz = (np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(k).T) * z[:, None]
    return xyz @ frame[:3, :3].T + frame[:3, 3]


def mesh_points(vertices, faces, frame_from_object, *, count=4096):
    """Area-weighted samples from the complete mesh, including occluded faces."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("invalid target mesh")
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=42)
    t = rigid(frame_from_object)
    if t.shape != (4, 4):
        raise ValueError("expected one object pose")
    return points @ t[:3, :3].T + t[:3, 3]


def infer(endpoint, request):
    import msgpack
    import zmq

    def encode(value):
        if not isinstance(value, np.ndarray):
            raise TypeError("expected numeric array")
        return {b"nd": True, b"type": value.dtype.str, b"kind": b"", b"shape": value.shape, b"data": value.tobytes()}

    def decode(value):
        if value.get(b"nd") is True:
            dtype = np.dtype(value[b"type"])
            if dtype.kind not in "iufb":
                raise ValueError("non-numeric service array")
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return value

    with zmq.Context() as context, context.socket(zmq.REQ) as sock:
        sock.linger = 0
        sock.sndtimeo = sock.rcvtimeo = 120000
        sock.setsockopt(zmq.MAXMSGSIZE, 8 * 1024 * 1024)
        sock.connect(endpoint)
        sock.send(msgpack.packb(request, default=encode, use_bin_type=True))
        return msgpack.unpackb(sock.recv(), object_hook=decode, raw=False)


@dataclass(frozen=True)
class Candidates:
    grippers: np.ndarray
    tools: np.ndarray
    scores: np.ndarray


class GraspPoseBackend:
    def __init__(self, config):
        self.source = config["point_cloud_source"]
        if config["backend"] != "graspgenx" or self.source not in {"rgbd", "ground_truth"}:
            raise ValueError("unknown grasp backend or point cloud source")
        self.endpoint = config["endpoint"]
        self.group = config["group"]
        self.effector = config["effector"]
        self.tool = config["control_link"]
        self.tool_offset = rigid(config["T_gripper_control"])
        self.count = config["num_grasps"]
        self.sweep = config["sweep_volume_params"]
        if (
            self.tool_offset.shape != (4, 4)
            or type(self.count) is not int
            or not 1 <= self.count <= 256
            or not self.tool
        ):
            raise ValueError("invalid gripper configuration")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def point_cloud(self, *, rgbd, ground_truth):
        points = np.asarray(rgbd() if self.source == "rgbd" else ground_truth(), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
            raise ValueError("insufficient or invalid segmented object point cloud")
        if len(points) > 4096:
            points = points[np.random.default_rng(42).choice(len(points), 4096, replace=False)]
        return points

    def propose(self, points):
        reply = infer(
            self.endpoint,
            {
                "action": "infer_object",
                "point_cloud": points,
                "sweep_volume_params": self.sweep,
                "planner": "diffusion",
                "num_grasps": self.count,
            },
        )
        if "error" in reply:
            raise RuntimeError(str(reply["error"]))
        g = rigid(reply["grasps"])
        scores = np.asarray(reply["confidences"], dtype=float)
        if (
            g.ndim != 3
            or not len(g)
            or scores.shape != (len(g),)
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("missing or invalid grasp candidates")
        return Candidates(g, g @ self.tool_offset, scores)
