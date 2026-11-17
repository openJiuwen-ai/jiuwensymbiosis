"""Exercise shared RGB-D sensing and bounded approach without robot or model services."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.motion.target_approach import TargetApproachTuning, approach_target
from jiuwensymbiosis.perception.frame import CameraFrame
from jiuwensymbiosis.perception.scene3d import locate_target


class Scene:
    cameras = ("head",)
    capabilities = frozenset({"vision.detection", "motion.base", "motion.goal"})

    def __init__(self, distance=2.0, bearing=0.2):
        self.cfg = SimpleNamespace(viz_detections=False)
        self.x, self.y = distance * math.cos(bearing), distance * math.sin(bearing)
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.commands = []
        self.frames = 0
        self.stopped = True
        self.depth_valid = True
        self.found = True
        self.move = True
        self.refusal = None

    def grab_calibrated_frame(self, camera=None):
        self.frames += 1
        depth = np.full((96, 128), self.x if self.depth_valid else np.nan, np.float32)
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        return CameraFrame(
            rgb=np.zeros((96, 128, 3), np.uint8),
            depth_m=depth,
            intrinsics=np.array([[100, 0, 64], [0, 100, 48], [0, 0, 1]]),
            tf_base_cam=transform,
            frame_id=f"frame-{self.frames}",
            captured_monotonic_s=time.monotonic(),
            metadata={
                "base_sample": {
                    "position_m": self.position.tolist(),
                    "body_yaw_rad": self.yaw,
                    "odom_frame": "test-odom",
                }
            },
        )

    def segment(self, rgb, text_prompt):
        if not self.found:
            return []
        u = round(64 - self.y / self.x * 100)
        mask = np.zeros((96, 128), bool)
        mask[30:67, max(0, u - 15) : min(128, u + 16)] = True
        return [{"mask": mask, "score": 0.95, "box": [u - 15, 30, u + 16, 67]}]

    def navigate_relative(self, dx_m, dy_m, dyaw_rad):
        self.commands.append((dx_m, dy_m, dyaw_rad))
        if self.move:
            c, s = math.cos(dyaw_rad), math.sin(dyaw_rad)
            self.x, self.y = c * self.x + s * self.y - dx_m, -s * self.x + c * self.y - dy_m
            self.yaw += dyaw_rad
            self.position += [
                math.cos(self.yaw) * dx_m - math.sin(self.yaw) * dy_m,
                math.sin(self.yaw) * dx_m + math.cos(self.yaw) * dy_m,
                0.0,
            ]
        return {"ok": True, "settled": self.stopped, "stop_confirmation": "measured" if self.stopped else "unknown"}

    def check_target_step(self, forward, turn, frame_id, tuning):
        assert frame_id == f"frame-{self.frames}"
        return self.refusal


def make_api(**kwargs):
    scene = Scene(**kwargs)
    api = BaseRobotApi(scene)
    api._seg_fn = scene.segment
    return api, scene


def test_real_geometry_and_closed_loop_reread_after_every_bounded_move():
    api, scene = make_api()
    result = approach_target(api, "box", 1.2)
    assert result["ok"] and result["goal_reached"] and result["settled"]
    assert abs(result["distance_error_m"]) <= 0.05
    assert abs(result["bearing_error_rad"]) <= math.radians(3)
    assert scene.frames == 2 * (len(result["steps"]) + 1)
    assert result["observations"] == scene.frames
    assert all(len(set(step["support_frame_ids"])) == 2 for step in result["steps"])
    assert len(set(result["confirmation_frame_ids"])) == 2
    assert len({s["frame_id"] for s in result["steps"]}) == len(result["steps"])
    assert all(abs(dx) <= 0.25 and dy == 0 and abs(yaw) <= 0.35 for dx, dy, yaw in scene.commands)
    assert all(s["visible_variation_m"] < 0.025 for s in result["steps"])
    assert result["measurement"]["point_kind"] == "visible_point_median"


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("stopped", "stop_unconfirmed"),
        ("depth_valid", "too_few_points"),
        ("found", "no_detection"),
    ],
)
def test_missing_evidence_never_moves_or_claims_arrival(condition, reason):
    api, scene = make_api()
    setattr(scene, condition, False)
    result = approach_target(api, "box")
    assert not result["ok"] and not result["goal_reached"]
    assert result["reason"] == reason
    assert scene.commands == [(0.0, 0.0, 0.0)]
    assert api.last_detection is None


@pytest.mark.parametrize("changed_reference", ["odom_frame", "scene_instance_id"])
@pytest.mark.parametrize(
    "initial_reference,new_reference", [("original", "reset-reference"), ("original", None), (None, "new-reference")]
)
def test_reference_reset_between_confirmation_frames_prevents_motion(
    changed_reference, initial_reference, new_reference
):
    api, scene = make_api(bearing=0.0)
    original = scene.grab_calibrated_frame

    def grab(camera=None):
        frame = original(camera)
        frame.metadata["scene_instance_id"] = "scene-1"
        metadata = frame.metadata["base_sample"] if changed_reference == "odom_frame" else frame.metadata
        metadata[changed_reference] = initial_reference
        if scene.frames == 2:
            metadata[changed_reference] = new_reference
        return frame

    scene.grab_calibrated_frame = grab
    result = approach_target(api, "box")
    assert not result["ok"] and not result["goal_reached"]
    assert result["reason"] == "target_reference_changed"
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_stable_scene_reference_can_supply_capture_aligned_base_coordinates():
    api, scene = make_api(bearing=0.0)
    original = scene.grab_calibrated_frame

    def grab(camera=None):
        frame = original(camera)
        frame.metadata["scene_instance_id"] = "stable-scene"
        del frame.metadata["base_sample"]["odom_frame"]
        return frame

    scene.grab_calibrated_frame = grab
    result = approach_target(api, "box")
    assert result["ok"] and result["goal_reached"]


@pytest.mark.parametrize(
    "motion",
    [{"ok": True}, {"ok": True, "settled": True}, {"ok": True, "settled": True, "stop_confirmation": "unknown"}],
)
def test_plain_success_or_unmeasured_stop_cannot_start_target_approach(motion):
    api, scene = make_api()

    def navigate(dx, dy, turn):
        scene.commands.append((dx, dy, turn))
        return motion

    scene.navigate_relative = navigate
    result = approach_target(api, "box")
    assert not result["ok"] and result["reason"] == "stop_unconfirmed"
    assert scene.frames == 0
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_default_minimum_target_distance_remains_reachable_with_clearance():
    api, scene = make_api(distance=1.0, bearing=0.0)
    result = approach_target(api, "box", TargetApproachTuning().min_target_distance_m)
    assert result["ok"] and result["goal_reached"]
    assert any(dx > 0 for dx, _, _ in scene.commands)


def test_valid_goal_does_not_override_increased_body_clearance():
    api, scene = make_api(distance=1.0, bearing=0.0)
    api.target_approach_tuning = TargetApproachTuning(body_radius_m=0.84)
    result = approach_target(api, "box", 0.8)
    assert result["reason"] == "insufficient_target_clearance"
    assert not result["goal_reached"]
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_confirmation_failure_preserves_service_error_code_without_claiming_arrival():
    from jiuwensymbiosis.errors import InferenceServiceError

    api, scene = make_api(distance=1.2, bearing=0.0)

    def segment(rgb, text_prompt):
        if scene.frames == 2:
            raise InferenceServiceError("test service failure", code="inference_failed")
        return scene.segment(rgb, text_prompt)

    api._seg_fn = segment
    result = approach_target(api, "box", 1.2)
    assert not result["ok"] and not result["goal_reached"]
    assert result["confirmation_failure"]["error_code"] == "inference_failed"
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_exhaustion_is_failure_even_if_each_command_reports_success():
    api, scene = make_api(bearing=0.0)
    scene.move = False
    api.target_approach_tuning = TargetApproachTuning(max_iterations=2)
    result = approach_target(api, "box")
    assert result["reason"] == "max_iterations"
    assert not result["ok"] and not result["goal_reached"]
    assert len(result["steps"]) == 2


def test_obstacle_guard_prevents_command_and_too_close_never_reverses():
    api, scene = make_api(bearing=0.0)
    scene.refusal = "depth_obstacle_blocked"
    assert approach_target(api, "box")["reason"] == "depth_obstacle_blocked"
    assert scene.commands == [(0.0, 0.0, 0.0)]
    api, scene = make_api(distance=1.0, bearing=0.0)
    assert approach_target(api, "box", 1.5)["reason"] == "too_close"
    assert scene.commands == [(0.0, 0.0, 0.0)]


@pytest.mark.parametrize("distance", [True, float("nan"), float("inf"), -0.1, 0.1, 10.0])
def test_invalid_goal_rejected_before_any_command(distance):
    api, scene = make_api()
    with pytest.raises(ValueError):
        approach_target(api, "box", distance)
    assert scene.commands == []


def test_genesis_binding_uses_the_rgbd_geometry():
    from jiuwensymbiosis.simulators.common.agent import SimulationPerceptionApi

    env = Scene(bearing=0.0)
    api = BaseRobotApi(env)
    api._seg_fn = env.segment
    genesis = SimulationPerceptionApi(env, env.segment)
    first = locate_target(api, "box")
    second = genesis.locate_target("box")
    for key in ("point_base_mm", "distance_m", "bearing_rad", "near_surface_x_m", "n_points"):
        assert first[key] == second[key]


def test_target_tools_are_exposed_for_genesis_but_not_cruzr():
    from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
    from jiuwensymbiosis.simulators.common.agent import SimulationPerceptionMotionApi
    from jiuwensymbiosis.tools.builder import build_robot_tools

    env = Scene(bearing=0.0)
    genesis = SimulationPerceptionMotionApi(env, env.segment)
    cruzr = CruzrApi(env, detector_client=env.segment)
    target_tools = {"locate_target", "approach_target"}
    assert target_tools <= {tool.card.name for tool in build_robot_tools(genesis, env=env)}
    assert target_tools.isdisjoint(tool.card.name for tool in build_robot_tools(cruzr, env=env))


def test_existing_grasp_location_keeps_its_result_shape():
    from jiuwensymbiosis.perception.scene3d import locate_for_grasp

    api, _ = make_api(bearing=0.0)
    grasp = locate_for_grasp(api, "box")
    assert grasp["ok"]
    assert "frame_id" not in grasp
    assert "captured_monotonic_s" not in grasp
    target = locate_target(api, "box")
    assert target["ok"]
    assert target["point_base_mm"] == grasp["center_mm"]
    assert target["frame_id"]
    assert target["captured_monotonic_s"] > 0


def test_target_location_reports_model_label_instead_of_turning_query_into_identity():
    api, scene = make_api(bearing=0.0)
    detection = {**scene.segment(None, "box")[0], "label": "bed", "score": 0.36}
    api._seg_fn = lambda *args, **kwargs: [detection]
    result = locate_target(api, "wardrobe")
    assert result["ok"]  # geometry can be valid while the semantic prediction is wrong
    assert result["object"] == "wardrobe"
    assert result["detection_label"] == "bed"
    assert result["detection_score"] == 0.36
    assert result["box_px"] == detection["box"]


def test_unreported_detector_label_stays_unknown():
    api, _ = make_api(bearing=0.0)
    result = locate_target(api, "box")
    assert result["detection_label"] is None


def test_inference_completion_guard_rejects_reset_before_cache_publication():
    api, scene = make_api()

    def changed(frame):
        raise RuntimeError("scene changed")

    scene.validate_frame = changed
    with pytest.raises(RuntimeError, match="scene changed"):
        locate_target(api, "box")
    assert api.last_detection is None


def test_two_distinct_targets_fail_but_duplicate_masks_do_not():
    api, scene = make_api(bearing=0.0)
    original = scene.segment(None, "box")[0]
    other = {**original, "mask": np.roll(original["mask"], 40, axis=1)}
    api._seg_fn = lambda *a, **k: [original, other]
    result = approach_target(api, "box")
    assert result["reason"] == "ambiguous_target" and not result["goal_reached"]
    assert scene.commands == [(0.0, 0.0, 0.0)]
    api._seg_fn = lambda *a, **k: [original, {**original, "score": 0.9}]
    assert locate_target(api, "box")["ok"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_target_masks_are_aligned_before_geometry_and_identity(duplicate):
    api, scene = make_api(bearing=0.0)
    original = scene.segment(None, "box")[0]
    reduced = {**original, "mask": original["mask"][::2, ::2]}
    api._seg_fn = lambda *args, **kwargs: [reduced, original] if duplicate else [reduced]
    assert locate_target(api, "box")["ok"]
    previous = api.target_observation
    assert previous is not None
    assert locate_target(api, "box", previous=previous)["ok"]


@pytest.mark.parametrize("failure", ["not_found", "invalid_frame", "service", "unexpected", "identity"])
def test_failed_relocalization_discards_previous_target_anchor(failure):
    from jiuwensymbiosis.errors import InferenceServiceError
    from jiuwensymbiosis.perception.frame import FrameValidationError

    api, scene = make_api(bearing=0.0)
    assert locate_target(api, "box")["ok"]
    previous = api.target_observation
    assert previous is not None
    if failure == "not_found":
        scene.found = False
    elif failure == "identity":
        scene.x = 1.2
    else:
        error = {
            "invalid_frame": FrameValidationError("stale_target_frame"),
            "service": InferenceServiceError("unavailable", service="vision", code="inference_unavailable"),
            "unexpected": RuntimeError("capture interrupted"),
        }[failure]

        def failed_capture(camera=None):
            raise error

        scene.grab_calibrated_frame = failed_capture
    if failure == "unexpected":
        with pytest.raises(RuntimeError, match="capture interrupted"):
            locate_target(api, "box", previous=previous)
    else:
        assert not locate_target(api, "box", previous=previous)["ok"]
    assert api.target_observation is None
    assert api.last_detection is None


@pytest.mark.parametrize(
    "base",
    [
        SimpleNamespace(odom_frame="test-odom"),
        {"odom_frame": "test-odom", "position_m": [0, 0, 0]},
        {"odom_frame": "test-odom", "body_yaw_rad": 0.0},
        {"odom_frame": "test-odom", "body_yaw_rad": 0.0, "position_m": [0, 0]},
        {"odom_frame": "test-odom", "body_yaw_rad": 0.0, "position_m": 0},
        {"odom_frame": "test-odom", "body_yaw_rad": float("inf"), "position_m": [0, 0, 0]},
        {"odom_frame": "test-odom", "body_yaw_rad": "invalid", "position_m": [0, 0, 0]},
        {"odom_frame": "test-odom", "body_yaw_rad": 0.0, "position_m": [0, float("nan"), 0]},
    ],
)
def test_invalid_egomotion_keeps_localization_but_prevents_approach(base):
    api, scene = make_api(bearing=0.0)
    capture = scene.grab_calibrated_frame

    def invalid_base(camera=None):
        frame = capture(camera)
        frame.metadata["base_sample"] = base
        return frame

    scene.grab_calibrated_frame = invalid_base
    assert locate_target(api, "box")["ok"]
    assert api.target_observation is None
    assert approach_target(api, "box")["reason"] == "association_odometry_unavailable"
    assert scene.commands == [(0.0, 0.0, 0.0)]


@pytest.mark.parametrize("refusal", [None, "turn_clearance_unobserved"])
def test_pure_rotation_uses_rotation_guard_without_forward_clearance(refusal):
    api, scene = make_api(distance=0.8, bearing=0.55)
    api.target_approach_tuning = TargetApproachTuning(max_iterations=1)
    assert locate_target(api, "box")["near_surface_x_m"] < 0.7
    scene.refusal = refusal
    result = approach_target(api, "box", 0.8)
    assert result["reason"] != "insufficient_target_clearance"
    if refusal:
        assert result["reason"] == refusal
        assert scene.commands == [(0.0, 0.0, 0.0)]
    else:
        assert scene.commands[1][:2] == (0.0, 0.0)
        assert scene.commands[1][2] > 0


def test_many_depth_points_in_one_corner_do_not_establish_geometry():
    api, scene = make_api(bearing=0.0)
    frame = scene.grab_calibrated_frame()
    frame.depth_m[:] = np.nan
    frame.depth_m[30:39, 49:58] = 2.0
    scene.grab_calibrated_frame = lambda camera=None: frame
    result = locate_target(api, "box")
    assert result["reason"] == "insufficient_depth_coverage"
    assert api.last_detection is None


@pytest.mark.parametrize("failure", ["repeated_frame", "drifting_target", "lost_target"])
def test_initially_in_range_still_requires_fresh_stable_final_observation(failure):
    api, scene = make_api(distance=1.2, bearing=0.0)
    original = scene.grab_calibrated_frame

    def capture(camera=None):
        if scene.frames:
            if failure == "drifting_target":
                scene.x += 0.2
            elif failure == "lost_target":
                scene.found = False
        frame = original(camera)
        if failure == "repeated_frame":
            frame.frame_id = "cached"
        return frame

    scene.grab_calibrated_frame = capture
    result = approach_target(api, "box", 1.2)
    assert not result["ok"] and not result["goal_reached"]
    assert (
        result["reason"]
        == {
            "repeated_frame": "stale_target_observation",
            "drifting_target": "target_identity_changed",
            "lost_target": "no_detection",
        }[failure]
    )
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_other_instance_at_goal_distance_cannot_replace_target_after_motion():
    api, scene = make_api(distance=2.0, bearing=0.0)
    original = scene.grab_calibrated_frame

    def switched(camera=None):
        if len(scene.commands) == 2:
            scene.x = 1.2  # same class, but another physical surface at the goal distance
        return original(camera)

    scene.grab_calibrated_frame = switched
    result = approach_target(api, "box", 1.2)
    assert result["reason"] == "target_identity_changed"
    assert not result["ok"] and not result["goal_reached"]
    assert len(result["steps"]) == 1
    assert api.last_detection is None and api.target_observation is None


def test_standalone_location_anchors_the_following_approach():
    api, scene = make_api(distance=2.0, bearing=0.0)
    assert locate_target(api, "box")["ok"]
    scene.x = 1.2
    result = approach_target(api, "box", 1.2)
    assert result["reason"] == "target_identity_changed"
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_appearance_change_cannot_be_hidden_by_matching_geometry():
    api, scene = make_api(distance=1.2, bearing=0.0)
    original = scene.grab_calibrated_frame

    def changed(camera=None):
        frame = original(camera)
        if scene.frames > 1:
            frame.rgb[:] = 255
        return frame

    scene.grab_calibrated_frame = changed
    assert approach_target(api, "box", 1.2)["reason"] == "target_identity_changed"


def test_missing_egomotion_refuses_motion_but_preserves_static_localization():
    api, scene = make_api()
    original = scene.grab_calibrated_frame

    def no_odom(camera=None):
        frame = original(camera)
        frame.metadata = {}
        return frame

    scene.grab_calibrated_frame = no_odom
    assert locate_target(api, "box")["ok"]
    assert approach_target(api, "box")["reason"] == "association_odometry_unavailable"
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_near_goal_jitter_is_reobserved_without_averaging_into_success():
    api, scene = make_api(distance=1.2, bearing=0.0)
    capture = scene.grab_calibrated_frame
    readings = iter([1.23, 1.26, 1.23, 1.22])

    def jittered(camera=None):
        frame = capture(camera)
        frame.depth_m[:] = next(readings)
        return frame

    scene.grab_calibrated_frame = jittered
    result = approach_target(api, "box", 1.2)
    assert result["ok"] and result["goal_reached"]
    assert result["confirmation_frame_ids"] == ["frame-3", "frame-4"]
    assert len(result["observation_retries"]) == 2
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_far_range_variation_uses_the_nearer_observation_for_a_bounded_step():
    api, scene = make_api(distance=1.6, bearing=0.0)
    api.target_approach_tuning = TargetApproachTuning(max_iterations=1)
    capture = scene.grab_calibrated_frame

    def jittered(camera=None):
        frame = capture(camera)
        frame.depth_m += 0.08 if scene.frames % 2 == 0 else 0.0
        return frame

    scene.grab_calibrated_frame = jittered
    result = approach_target(api, "box", 1.2)
    assert not result["ok"] and result["reason"] == "max_iterations"
    assert len(result["steps"]) == 1
    assert 0 < scene.commands[1][0] <= 0.201
    assert scene.commands[1][1:] == (0.0, 0.0)
    assert not result["observation_retries"]


def test_shorter_final_step_leaves_room_for_viewpoint_dependent_mask_change():
    api, scene = make_api(distance=1.43, bearing=0.0)
    capture = scene.grab_calibrated_frame

    def changed_view(camera=None):
        frame = capture(camera)
        if len(scene.commands) > 1:
            frame.depth_m -= 0.09
        return frame

    scene.grab_calibrated_frame = changed_view
    result = approach_target(api, "box", 1.2)
    assert result["ok"] and result["goal_reached"]
    assert len(result["steps"]) == 1
    assert result["confirmation_frame_ids"] == ["frame-3", "frame-4"]
    assert abs(result["distance_error_m"]) <= 0.05
    assert all(dx >= 0 and dy == yaw == 0 for dx, dy, yaw in scene.commands)


def test_cross_view_median_shift_reduces_step_even_when_stationary_pair_agrees():
    api, scene = make_api(distance=1.75, bearing=0.0)
    api.target_approach_tuning = TargetApproachTuning(max_iterations=2)
    capture = scene.grab_calibrated_frame

    def changed_view(camera=None):
        frame = capture(camera)
        if len(scene.commands) > 1:
            frame.depth_m += 0.1
        return frame

    scene.grab_calibrated_frame = changed_view
    result = approach_target(api, "box", 1.2)
    assert result["reason"] == "max_iterations" and not result["ok"]
    assert scene.commands[1][0] == 0.25
    assert 0 < scene.commands[2][0] <= 0.201
    assert result["steps"][1]["visible_variation_m"] > 0.09


def test_permanently_inconsistent_measurements_exhaust_reobservations_without_motion():
    api, scene = make_api(distance=1.2, bearing=0.0)
    capture = scene.grab_calibrated_frame

    def jittered(camera=None):
        frame = capture(camera)
        frame.depth_m[:] = 1.16 if scene.frames % 2 else 1.24
        return frame

    scene.grab_calibrated_frame = jittered
    result = approach_target(api, "box", 1.2)
    assert not result["ok"] and not result["goal_reached"]
    assert result["reason"] == "arrival_not_stable"
    assert scene.frames == 4 and len(result["observation_retries"]) == 2
    assert scene.commands == [(0.0, 0.0, 0.0)]


def test_observation_retry_budget_is_shared_across_motion_steps():
    api, scene = make_api(distance=2.0, bearing=0.0)

    def intermittent(rgb, text_prompt):
        return [] if scene.frames in {2, 5, 8} else scene.segment(rgb, text_prompt)

    api._seg_fn = intermittent
    result = approach_target(api, "box", 1.2)
    assert result["reason"] == "no_detection" and not result["ok"]
    assert scene.frames == 8 and len(result["observation_retries"]) == 2
    assert len(result["steps"]) == 2


def test_scene_reset_during_stationary_retry_is_terminal():
    api, scene = make_api(distance=1.2, bearing=0.0)
    api._seg_fn = lambda rgb, text_prompt: [] if scene.frames == 2 else scene.segment(rgb, text_prompt)

    def validate(frame):
        if scene.frames >= 3:
            raise RuntimeError("scene changed")

    scene.validate_frame = validate
    with pytest.raises(RuntimeError, match="scene changed"):
        approach_target(api, "box", 1.2)
    assert scene.commands == [(0.0, 0.0, 0.0)]


@pytest.mark.parametrize("value", [-1, True, 1.5, 9])
def test_invalid_observation_retry_budget(value):
    with pytest.raises(ValueError, match="observation retry"):
        TargetApproachTuning(max_observation_retries=value)
