"""Mobile exploration must use downward-looking observations without inventing depth."""

import math

import numpy as np

from jiuwensymbiosis.spatial_memory.exploration import OnlineMap


def tilted_floor():
    width, height = 160, 120
    k = np.array([[120, 0, width / 2], [0, 120, height / 2], [0, 0, 1.0]])
    angle = math.radians(50)
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, -math.sin(angle), math.cos(angle)], [0, -math.cos(angle), -math.sin(angle)]]
    pose[2, 3] = 1.8
    v, u = np.mgrid[:height, :width]
    rays = np.stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)), axis=-1)
    world = rays @ pose[:3, :3].T
    depth = -pose[2, 3] / world[:, :, 2]
    return np.zeros((height, width, 3), np.uint8), depth, k, pose


def test_downward_camera_proposes_observation_backed_translation():
    online = OnlineMap()
    rgb, depth, k, pose = tilted_floor()
    online.integrate(rgb, depth, k, pose)
    candidates, _ = online.candidates(pose)
    assert any(c["kind"] == "advance" and c["path_length_m"] >= 0.1 for c in candidates)


def test_absent_depth_cannot_create_translation_candidates():
    online = OnlineMap()
    rgb, depth, k, pose = tilted_floor()
    online.integrate(rgb, np.full_like(depth, np.nan), k, pose)
    candidates, _ = online.candidates(pose)
    assert not any(c["path_length_m"] > 0 for c in candidates)
