"""Calibrated Nyx sampling; called only by the scene-owning worker thread.

No scene identities or object labels cross this interface. Missing depth stays
NaN: this is a sampled RGB-D observation, not a reconstructed dense surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product

import numpy as np


def effective_focal(size: tuple[int, int], fov_degrees: float, nyx_version: str, plugin_version: str) -> float:
    """Pinned projection behavior established by independent native-ray/RGB probes."""
    if (nyx_version, plugin_version) != ("0.1.7", "0.1.9"):
        raise ValueError("Nyx camera projection requires validation for this SDK/plugin version")
    radians = float(np.float16(math.radians(fov_degrees)))
    return size[1] / (2 * math.tan(radians / 2))


def rectified_grid(camera, native_size: tuple[int, int], focal: float) -> np.ndarray:
    """Map rectified calibrated pixels directly to Nyx, avoiding a second RGB warp."""
    width, height = camera.resolution
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    k = camera.intrinsic
    rays = np.stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]), axis=-1)
    grid = (rays * (2 * focal / np.asarray(native_size))).astype(np.float32)
    if not np.isfinite(grid).all() or np.max(np.abs(grid)) >= 1:
        raise ValueError("Rectified camera exceeds the native render coverage")
    return grid


def sample_depth(
    native, camera, world_from_camera: np.ndarray, *, stride: int, check: Callable[[], None]
) -> np.ndarray:
    """Ray-cast a regular rectified grid at the already-rendered physical state.

    The caller's check enforces the deadline and cancellation between short
    blocks. All native calls remain on the worker's main thread.
    """
    import gs_nyx.nyx_py_renderer as npr
    import gs_nyx.nyx_py_sdk as nps

    if type(stride) is not int or not 4 <= stride <= 32:
        raise ValueError("RGB-D sampling stride must be an integer in [4, 32]")
    width, height = camera.resolution
    xs = np.unique(np.r_[np.arange(0, width, stride), width - 1])
    ys = np.unique(np.r_[np.arange(0, height, stride), height - 1])
    u, v = np.meshgrid(xs, ys)
    k = camera.intrinsic
    directions = np.stack(((u.ravel() - k[0, 2]) / k[0, 0], (v.ravel() - k[1, 2]) / k[1, 1], np.ones(u.size)), axis=1)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    rotation, origin = world_from_camera[:3, :3], world_from_camera[:3, 3]
    world_directions = directions @ rotation.T
    # Derive Nyx's axis permutation from its own helpers, then apply it in bulk.
    # Keep float64 geometry until the SDK's float32 ray/hit boundary.
    native_axes = [nps.float3_z_up_to_y_up_a(nps.float3(*axis)) for axis in np.eye(3)]
    native_basis = np.array([(axis.x, axis.y, axis.z) for axis in native_axes])
    native_directions = (world_directions @ native_basis).tolist()
    ray_limits = (np.asarray(camera.clipping) / directions[:, 2, None]).tolist()
    origin_native = nps.float3_z_up_to_y_up_a(nps.float3(*map(float, origin)))
    hit, reference = nps.float3(0, 0, 0), npr.SceneReference()
    points = np.full((u.size, 3), np.nan)
    for i, (direction, limits) in enumerate(zip(native_directions, ray_limits, strict=True)):
        if i % 32 == 0:
            check()
        if native.trace_ray(origin_native, nps.float3(*direction), nps.float2(*limits), hit, reference):
            points[i] = hit.x, hit.y, hit.z
    check()
    z = (points @ native_basis.T - origin) @ rotation[:, 2]
    valid = np.isfinite(z) & (z >= camera.clipping[0]) & (z <= camera.clipping[1])
    depth = np.full((height, width), np.nan, dtype=np.float32)
    depth[v.ravel()[valid], u.ravel()[valid]] = z[valid]
    check()
    return depth


def base_from_camera(world_from_camera: np.ndarray, base_state: dict) -> np.ndarray:
    """Gravity-aligned MOZ1 forward/left/up frame; translations in millimetres."""
    yaw = base_state["body_yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    world_from_base[:3, 3] = base_state["position_m"]
    transform = np.asarray(np.linalg.inv(world_from_base) @ world_from_camera, dtype=float)
    transform[:3, 3] *= 1000
    return transform


def base_body_bounds(world_bounds: np.ndarray, base_state: dict) -> np.ndarray:
    """Enclose this robot's live physical AABB in its forward/left/up frame.

    Rotating an already axis-aligned world box can enlarge it. That is deliberate:
    this observation guard requires a conservative body envelope, not scene truth.
    """
    bounds = np.asarray(world_bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Robot body bounds unavailable")
    corners = np.array(list(product(*bounds.T)))
    transform = base_from_camera(np.eye(4), base_state)
    points = corners @ transform[:3, :3].T + transform[:3, 3] / 1000
    return np.stack([points.min(axis=0), points.max(axis=0)])
