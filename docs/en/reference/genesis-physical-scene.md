# Genesis physical scene backend

This page describes host/worker interfaces in `jiuwensymbiosis.simulators.genesis`.
The [Chinese version](../../zh/reference/genesis-physical-scene.md) is authoritative.
These are backend capabilities; new Agent bindings and workbench controls follow separately.

## Scene cache

`RoomSceneCache.prepare()` caches fixed room collision entities and original meshes, excluding
robots, dynamic objects and live world state. Robots must still be added and the scene built.
Static USD backgrounds also load collision geometry.

The default is `$XDG_CACHE_HOME/jiuwensymbiosis/genesis/scenes`, falling back to
`~/.cache/jiuwensymbiosis/genesis/scenes`. Override with an absolute
`JIUWENSYMBIOSIS_GENESIS_SCENE_CACHE`; relative paths are rejected.
Keys include source/USD dependency content, physics options, relevant SDK versions and
cache/collision processor code. Hits validate the manifest and payload checksums; corrupt
entries rebuild under a per-key process lock. Publication is atomic, and source changes
during preparation prevent publication. `scene.scene_cache` reports `hit`/`built` and rebuild reasons.
Cache reuse does not establish collision equivalence or physical acceptance.

## Reset within the current world

`GenesisPreview.reset_scene()` retains the physical scene and renderer while restoring the
initial settled checkpoint. Motion ownership, capture or scene query activity prevents reset.
Mobile robots require measured low speed for 0.6 simulated seconds (linear below 0.008 m/s,
angular below 0.02 rad/s). Unconfirmed stopping or cancellation prevents checkpoint restoration.

Restoration changes `scene_instance_id` and `reset_generation`, clearing selection and truth
snapshots. The host confirms completion only after a fresh restored-world frame. A 15-second
host timeout means completion is unconfirmed, without automatic retry or guaranteed worker
cancellation. Old frames, queries and control sessions cannot cross a reset. Changing only
observer resolution retains world state and pose.

## RGB-D capture

`GenesisPreview.capture_image(camera_name, include_depth=True, depth_stride=8)` accepts only
robot cameras on a supported mobile robot without active motion. It needs no video connection;
callers remain responsible for establishing measured stopping. The scene-owning worker samples
without advancing physics, checking cancellation, deadlines and configuration changes in blocks.

Results include JPEG, aligned `rgb`/`depth_m`, calibration, `base_sample`, `body_bounds_m` and
scene/frame binding metadata. Depth is optical Z in metres; `T_base_camera_mm` uses millimetre
translation; body bounds use forward/left/up base metres. Integer sampling strides are 4–32.
Unsampled pixels and missing hits remain NaN and cannot imply free space. The projection guard
accepts the validated Nyx SDK 0.1.7 / plugin 0.1.9 pair; other versions require new validation.
The host rejects stale/mismatched results, invalid array shapes/dtypes and RGB-D payloads over 32 MiB.

## Explicit truth and navigation plans

`GenesisPreview.query_scene(operation, params=None, cancelled=None)` executes between physics steps:

| operation | Parameters and behavior |
|---|---|
| `list_entities` | Filter by `query`/`kind`; integer `limit` 1–50, nonnegative `offset`; returns next offset. |
| `robot_state` | Measured mobile-robot state; unsupported scenes fail explicitly. |
| `entity_state` | `entity_id` with `frame="world"` or `"base"`; transforms origin, orientation and bounds together. |
| `plan_navigation` | Exact `x_m/y_m/yaw_rad`, or a static bounded `entity_id` plus `stand_off_m`; plans without executing. |

Every result states `source="simulation_ground_truth"`, units, scene/reset identity, step and time.
Entity names are authored asset names, not model recognition. Reuse `snapshot_id` for data from
one step across queries. At most 16 snapshots are retained; they expire after 60 seconds or reset.
Host query parameters are capped at 2048 bytes, with a 60-second deadline. Truth snapshots and
mesh-derived maps must not substitute for RGB-D, exploration or measured motion completion.

Navigation separates support from obstacles and erodes free space by the body envelope plus
grid margin. Unsupported/blocked positions, oversized maps and dynamic targets are rejected.
A path does not prove execution, stopping or arrival. MOZ1 holds initial upper-body joints
through physics control; payload/contact behavior still requires separate physical acceptance.

## Robot model admission and materials

Mobile mode prepares and audits collision geometry before initializing motors; failure prevents motion.
No MOZ1 example policy is loaded by default. `info.model_admission` exposes the report path;
`robot_model_instance_id` survives reset. Evidence persists under the state directory’s `robot-models/`,
with a `closed` marker after shutdown for operator-managed archival. See the
[model admission guide](../how-to/audit-robot-models.md) for import settings and retention requirements.
Source asset manifests pair physical and visual URDFs; derived model manifests identify loaded collisions.
Visual materials are handled separately by `material_surface.py`; visual appearance is not collision evidence.
