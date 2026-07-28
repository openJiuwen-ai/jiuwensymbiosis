# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic action-sequence runner for the C1 fast path (no per-step LLM).

Executes an ordered ``list[ActionStep]`` (produced once by the skill-selection
LLM, see ``fast_path_single_source_design.md``) against a live session. It is
**task-agnostic** — it knows nothing about pick/place/carry/push; it only knows:

  * how to call any ``@robot_tool`` action by name (the same ``_build_action_index``
    the agent's ``robot_control`` uses), so whatever a robot/skill exposes runs;
  * how to evaluate a step's symbolic params against a variable environment
    (config constants + detection bindings) via ``sequence.resolve_params``;
  * ``track_detect``: the legacy eye-in-hand relative tracker, which falls back
    to a home pre-scan when the wrist camera is occluded;
  * ``track_grasp``: the eye-to-hand absolute two-stage approach/descend tracker.

The whole pick/place *meaning* lives in the SKILL.md the LLM compiled — never
here. Adding a new task = adding a SKILL.md; this runner is unchanged.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.agent.fast.realtime.mask_tracking import MaskTargetFilter, MaskTrackingConfig
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController, ServoResult
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker
from jiuwensymbiosis.agent.fast.sequence import (
    TRACK_DETECT,
    TRACK_GRASP,
    ActionStep,
    normalize_detection,
    resolve_params,
)
from jiuwensymbiosis.rails.recovery import read_holding_payload, recover_session
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

logger = logging.getLogger(__name__)

Pose = dict[str, float]

# Ops that toggle the "holding something / wrist camera occluded" state. Generic
# eye-in-hand heuristic: once the end effector grips, a wrist camera is likely
# blocked, so detection should read the home pre-scan rather than track live.
_GRIP_CLOSE_OPS = frozenset({"close_gripper", "activate_suction"})
_GRIP_OPEN_OPS = frozenset({"open_gripper", "deactivate_suction"})
# Internal safety policy: tracking never drives from an image older than this.
# Independent of user-facing timeout tuning so it cannot be widened by accident.
_MAX_TRACKING_IMAGE_AGE_S = 8.0


@dataclass
class SkillExecConfig:
    """Tuning for the fast-path runner (servo / detection only).

    No motion-offset knobs (approach/lift): like the agent path, all working
    heights come from the detection's ``grasp_z`` / ``place_z`` (which already
    embed the calibration offsets ``grasp_z_offset`` / ``chip_thickness``). The
    workflow descends straight to those — no extra hover/lift offset, so there is
    nothing to tune here for motion geometry.
    """

    detect_hz: float = 5.0  # background detection rate cap
    first_target_timeout_s: float = 8.0  # wait this long for the first detection
    settle_grip_s: float = 0.5  # pause after a gripper command (let it actuate)
    # Cap on post-descend re-align passes before fail-closing (bounds re-servoing
    # when the object keeps moving between detections).
    max_re_align_iters: int = 1
    # A gripper adapter may expose a private ``is_grasp_confirmed`` hook.  When
    # it reports an empty close after ``track_grasp``, return home and run the
    # complete perception/approach/close attempt once more.
    max_grasp_retries: int = 1
    servo: ServoConfig = field(default_factory=ServoConfig)  # track-loop tuning
    # Opt-in at the adapter boundary: only an API exposing the private
    # get_grasp_tracking_sample() hook uses this filter. Other robots keep the
    # original get_grasp_info_simple() tracking path.
    mask_tracking: MaskTrackingConfig = field(default_factory=MaskTrackingConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("detect_hz", self.detect_hz),
            ("first_target_timeout_s", self.first_target_timeout_s),
        ):
            if not (isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"SkillExecConfig.{name} must be finite and > 0, got {value!r}.")
        if not (
            isinstance(self.settle_grip_s, (int, float))
            and math.isfinite(float(self.settle_grip_s))
            and float(self.settle_grip_s) >= 0
        ):
            raise ValueError(f"SkillExecConfig.settle_grip_s must be finite and >= 0, got {self.settle_grip_s!r}.")
        for name in ("max_re_align_iters", "max_grasp_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"SkillExecConfig.{name} must be int, got {value!r}.")
            if value < 0:
                raise ValueError(f"SkillExecConfig.{name} must be >= 0, got {value}.")
        if not isinstance(self.mask_tracking, MaskTrackingConfig):
            raise ValueError(
                f"SkillExecConfig.mask_tracking must be a MaskTrackingConfig, got {type(self.mask_tracking).__name__}."
            )


def _grasp_alignment_error_mm(cur: Mapping[str, Any], detection: Mapping[str, Any]) -> float:
    """Return absolute tip-to-grasp error in base-frame XYZ millimetres."""
    _validate_grasp_detection(detection)
    try:
        dx = float(cur["x"]) - float(detection["x"])
        dy = float(cur["y"]) - float(detection["y"])
        dz = float(cur["z"]) - float(detection["grasp_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("track_grasp alignment requires numeric tip x/y/z") from exc
    if not all(math.isfinite(value) for value in (dx, dy, dz)):
        raise ValueError("track_grasp alignment error must be finite")
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _validate_grasp_detection(gi: Mapping[str, Any]) -> None:
    """Validate raw detection fields required by absolute grasp servoing."""
    position = gi.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError("absolute grasp detection requires position[x,y]")
    try:
        values = (float(position[0]), float(position[1]), float(gi["grasp_z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("absolute grasp detection requires numeric position and grasp_z") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("absolute grasp detection position/grasp_z must be finite")


def _run_servo_phase(
    binding: ServoBinding,
    target_provider: Callable[[], Pose | None],
    *,
    config: ServoConfig,
    phase: str,
    target_is_live: Callable[[], bool],
) -> ServoResult:
    """Run one servo phase with the adapter's shared control policy."""
    return ServoController(
        binding.read_pose,
        binding.servo_to,
        target_provider,
        config=config,
        on_tick=binding.make_tick_logger(phase),
        target_is_live=target_is_live,
        slew_from_last_command=binding.slew_from_last_command,
        reached_angular_keys=binding.reached_angular_keys,
    ).run()


def _detect_once(api: Any, object_name: str, *, require_grasp: bool = False) -> dict[str, Any] | None:
    """One detection → normalized binding dict, or ``None`` if not detected."""
    try:
        gi = api.get_grasp_info_simple(object_name)
    except Exception as exc:  # noqa: BLE001 - detection may raise; treat as miss
        logger.debug("[runner] detection raised for %r: %s", object_name, exc)
        return None
    if not isinstance(gi, dict) or not gi.get("ok"):
        return None
    if require_grasp:
        try:
            _validate_grasp_detection(gi)
        except ValueError as exc:
            logger.warning("[runner] absolute grasp detection invalid for %r: %s", object_name, exc)
            return None
    return normalize_detection(gi)


def _detect_mask_tracking_once(
    provider: Callable[[str], dict[str, Any]],
    object_name: str,
    target_filter: MaskTargetFilter,
) -> dict[str, Any] | None:
    """One private adapter sample through the fixed-camera mask filter."""
    try:
        gi = provider(object_name)
    except Exception as exc:  # noqa: BLE001 - detection may raise; treat as miss
        logger.debug("[runner] mask tracking detection raised for %r: %s", object_name, exc)
        return target_filter.miss(type(exc).__name__)
    if not isinstance(gi, dict) or not gi.get("ok"):
        reason = gi.get("reason", "not_detected") if isinstance(gi, dict) else "invalid_result"
        return target_filter.miss(str(reason))
    try:
        sample = normalize_detection(gi)
    except (TypeError, ValueError, IndexError) as exc:
        # Let the filter inspect the mask first.  A partial mask can establish
        # occlusion even when its newly projected geometry is unusable.
        logger.debug("[runner] mask tracking geometry invalid for %r: %s", object_name, exc)
        sample = dict(gi)
    return target_filter.update(sample)


def _prescan(session: Any, steps: list[ActionStep]) -> dict[str, dict[str, Any]]:
    """At the home pose, detect every tracking target once and cache it.

    Eye-in-hand: a target grasped later occludes the wrist camera, so its
    position must be read now. Best-effort — a target not seen at home is simply
    absent (the live track will try again). Task-agnostic: just caches whatever
    objects the sequence names. Only ``track_detect`` reads this cache: the
    eye-to-hand ``track_grasp`` drives from absolute base-frame coordinates
    each tick and never consults a home-pose snapshot, so it is excluded here.
    """
    api = session.api
    names: list[str] = []
    for s in steps:
        if s.op == TRACK_DETECT:
            n = s.params.get("object_name")
            if isinstance(n, str) and n and n not in names:
                names.append(n)
    cache: dict[str, dict[str, Any]] = {}
    if not names:
        return cache
    try:
        api.home()
    except Exception as exc:  # noqa: BLE001 - pre-scan home() is best-effort
        logger.warning("[runner] pre-scan: home() failed: %s", exc)
    for n in names:
        det = _detect_once(api, n)
        if det is not None:
            cache[n] = det
            logger.info("[runner] pre-scan cached %r at home: pos=%s", n, det.get("position"))
        else:
            logger.warning("[runner] pre-scan: %r not detected at home (will retry live)", n)
    return cache


def _track_detect(
    session: Any,
    object_name: str,
    cfg: SkillExecConfig,
    cache: Mapping[str, dict[str, Any]],
    *,
    occluded: bool,
) -> dict[str, Any] | None:
    """Real-time-track ``object_name`` until it settles; return its normalized
    detection binding (full field set), or the home pre-scan when occluded /
    never seen.

    When ``occluded`` (gripper holding something → wrist camera blocked), skip
    the live loop and use the cache directly. Otherwise the tip mirrors the
    object's XY displacement at the observe height so it stays framed while it
    moves; the loop settles when the object stops and the tip has caught up.
    """
    if occluded:
        cached = cache.get(object_name)
        if cached is not None:
            logger.info("[runner] track_detect %r: using home pre-scan (occluded)", object_name)
            return dict(cached)
        logger.warning("[runner] track_detect %r: occluded and not cached; trying live anyway", object_name)

    binding = ServoBinding(session)
    pose0 = binding.read_pose()
    r0 = float(pose0.get("r", pose0.get("rz", 0.0)))
    obs_x, obs_y, obs_z = float(pose0["x"]), float(pose0["y"]), float(pose0["z"])

    api = session.api
    tracker = BackgroundTracker(
        lambda: _detect_once(api, object_name),
        max_hz=cfg.detect_hz,
        staleness_s=_MAX_TRACKING_IMAGE_AGE_S,
        name=object_name,
    )
    tracker.start()
    try:
        if not tracker.wait_first(cfg.first_target_timeout_s):
            cached = cache.get(object_name)
            if cached is not None:
                logger.info("[runner] track_detect %r: live miss → home pre-scan", object_name)
                return dict(cached)
            return None
        first = tracker.latest_target()
        if first is None:
            raise RuntimeError("track_detect first detection was already stale")
        obj0x, obj0y = float(first["x"]), float(first["y"])

        def target_is_live() -> bool:
            return tracker.target_is_live(
                no_update_grace_s=cfg.servo.lost_target_grace_s,
                max_image_age_s=_MAX_TRACKING_IMAGE_AGE_S,
            )

        def track_target() -> Pose | None:
            latest = tracker.latest_target()
            if latest is None:
                return None
            return {
                "x": obs_x + (float(latest["x"]) - obj0x),
                "y": obs_y + (float(latest["y"]) - obj0y),
                "z": obs_z,
                "r": r0,
            }

        res = _run_servo_phase(
            binding,
            track_target,
            config=cfg.servo,
            phase="track_detect",
            target_is_live=target_is_live,
        )
        logger.info(
            "[runner] track_detect %r: %s in %d ticks / %.2fs%s",
            object_name,
            res.reason,
            res.ticks,
            res.elapsed_s,
            f" error={res.error}" if res.error else "",
        )
        if not res.ok:
            raise RuntimeError(
                f"track_detect failed: {res.reason}: {res.error}" if res.error else f"track_detect failed: {res.reason}"
            )
        latest = tracker.latest_target()
        return dict(latest) if latest is not None else None
    finally:
        tracker.stop()


def _track_grasp(
    session: Any,
    object_name: str,
    approach_mm: float,
    cfg: SkillExecConfig,
) -> dict[str, Any] | None:
    """Eye-to-hand absolute approach + descend servo for a visual pick.

    A single live tracker feeds two sequential controllers.  Unlike the legacy
    ``track_detect`` operation, each target is an absolute base-frame pose from
    the latest ``get_grasp_info_simple`` result; no observation-pose-relative
    displacement is used.
    """
    binding = ServoBinding(session)
    pose0 = binding.read_pose()
    # Lock the entry yaw across approach, descend, and re-align. A future
    # non-top grasp must supply an explicit desired orientation instead of
    # adopting any actual-pose drift between phases.
    rz0 = float(pose0.get("rz", pose0.get("r", 0.0)))
    api = session.api
    tracking_provider = getattr(api, "get_grasp_tracking_sample", None)
    mask_filter: MaskTargetFilter | None = None
    if callable(tracking_provider) and cfg.mask_tracking.enabled:
        mask_filter = MaskTargetFilter(cfg.mask_tracking, name=object_name)

        def detect_fn() -> dict[str, Any] | None:
            assert mask_filter is not None
            return _detect_mask_tracking_once(tracking_provider, object_name, mask_filter)

        logger.info("[runner] track_grasp %r: fixed-camera mask filtering enabled", object_name)
    else:

        def detect_fn() -> dict[str, Any] | None:
            return _detect_once(api, object_name, require_grasp=True)

    tracker = BackgroundTracker(
        detect_fn,
        max_hz=cfg.detect_hz,
        staleness_s=_MAX_TRACKING_IMAGE_AGE_S,
        name=f"grasp-{object_name}",
    )
    tracker.start()
    try:
        if not tracker.wait_first(cfg.first_target_timeout_s):
            return None

        def target_is_live() -> bool:
            return tracker.target_is_live(
                no_update_grace_s=cfg.servo.lost_target_grace_s,
                max_image_age_s=_MAX_TRACKING_IMAGE_AGE_S,
            )

        def approach_target() -> Pose | None:
            latest = tracker.latest_target()
            if latest is None:
                return None
            return {
                "x": float(latest["x"]),
                "y": float(latest["y"]),
                "z": float(latest["grasp_z"]) + float(approach_mm),
                "rz": rz0,
            }

        def descend_target() -> Pose | None:
            latest = tracker.latest_target()
            if latest is None:
                return None
            return {
                "x": float(latest["x"]),
                "y": float(latest["y"]),
                "z": float(latest["grasp_z"]),
                "rz": rz0,
            }

        approach = _run_servo_phase(
            binding,
            approach_target,
            config=cfg.servo,
            phase="track_grasp.approach",
            target_is_live=target_is_live,
        )
        logger.info(
            "[runner] track_grasp %r approach: %s in %d ticks / %.2fs (detections=%d)%s",
            object_name,
            approach.reason,
            approach.ticks,
            approach.elapsed_s,
            tracker.detections,
            f" error={approach.error}" if approach.error else "",
        )
        if not approach.ok:
            raise RuntimeError(
                f"track_grasp approach failed: {approach.reason}: {approach.error}"
                if approach.error
                else f"track_grasp approach failed: {approach.reason}"
            )
        descend = _run_servo_phase(
            binding,
            descend_target,
            config=cfg.servo,
            phase="track_grasp.descend",
            target_is_live=target_is_live,
        )
        logger.info(
            "[runner] track_grasp %r descend: %s in %d ticks / %.2fs (detections=%d)%s",
            object_name,
            descend.reason,
            descend.ticks,
            descend.elapsed_s,
            tracker.detections,
            f" error={descend.error}" if descend.error else "",
        )
        if not descend.ok:
            raise RuntimeError(
                f"track_grasp descend failed: {descend.reason}: {descend.error}"
                if descend.error
                else f"track_grasp descend failed: {descend.reason}"
            )

        # Require a detection whose IMAGE was grabbed after descend finished:
        # capture time (not inference completion) so a frame grabbed mid-descend
        # whose inference finishes late is not mistaken for a post-descend frame.
        descend_finished_t = time.monotonic()
        final = _wait_post_descend_target(
            tracker,
            descend_finished_t,
            timeout_s=cfg.first_target_timeout_s,
        )
        if final is None:
            raise RuntimeError("track_grasp descend reached but no fresh post-descend detection arrived")
        latest, _capture_t = final
        _validate_grasp_detection(latest)

        # Re-align if the post-descend target jumped beyond reach tolerance.
        for _ in range(max(0, cfg.max_re_align_iters)):
            cur = binding.read_pose()
            err = _grasp_alignment_error_mm(cur, latest)
            if err <= cfg.servo.pos_tol_mm * 1.5:
                break
            logger.info(
                "[runner] track_grasp %r post-descend target moved %.1f mm; re-aligning",
                object_name,
                err,
            )
            re_descend = _run_servo_phase(
                binding,
                descend_target,
                config=cfg.servo,
                phase="track_grasp.re_descend",
                target_is_live=target_is_live,
            )
            if not re_descend.ok:
                raise RuntimeError(
                    f"track_grasp post-descend re-align failed: {re_descend.reason}: {re_descend.error}"
                    if re_descend.error
                    else f"track_grasp post-descend re-align failed: {re_descend.reason}"
                )
            descend_finished_t = time.monotonic()
            final = _wait_post_descend_target(
                tracker,
                descend_finished_t,
                timeout_s=cfg.first_target_timeout_s,
            )
            if final is None:
                raise RuntimeError("track_grasp re-align reached but no fresh detection arrived")
            latest, _capture_t = final
            _validate_grasp_detection(latest)
        else:
            # Loop exhausted without break: tip still off the final target.
            cur = binding.read_pose()
            err = _grasp_alignment_error_mm(cur, latest)
            if err > cfg.servo.pos_tol_mm * 1.5:
                raise RuntimeError(
                    f"track_grasp tip {err:.1f} mm off final target after re-align; aborting before close"
                )

        logger.info(
            "[runner] track_grasp %r final target: position=%s grasp_z=%s state=%s detections=%d",
            object_name,
            latest.get("position"),
            latest.get("grasp_z"),
            latest.get("_tracking_state", "unfiltered"),
            tracker.detections,
        )
        return dict(latest)
    finally:
        tracker.stop()


def _wait_post_descend_target(
    tracker: BackgroundTracker,
    descend_finished_t: float,
    *,
    timeout_s: float,
) -> tuple[dict[str, Any], float] | None:
    """Wait for a post-descend detection whose image capture time is ``>= descend_finished_t``."""
    return tracker.wait_for_capture_after(descend_finished_t, timeout_s=timeout_s)


# An executor runs ONE primitive op through whatever dispatch path the caller
# chose, returning a structured ``{ok, result?, reason?}``. The fast path passes
# an ability-manager-backed executor so every op goes through the SAME rails the
# agent uses (Safety/VisualFeedback/Recovery); tests pass a direct one.
Executor = Callable[[str, dict[str, Any]], dict[str, Any]]


class _StepExecutionError(RuntimeError):
    """Primitive-op failure with recovery ownership metadata."""

    def __init__(self, message: str, *, recovery_managed: bool = False) -> None:
        super().__init__(message)
        self.recovery_managed = recovery_managed


def _raise_executor_failure(
    response: Mapping[str, Any],
    fallback: str,
    *,
    context: str | None = None,
) -> None:
    """Raise a step failure while preserving whether rails already recovered."""
    message = str(response.get("reason") or fallback)
    if context is not None:
        message = f"{context}: {message}"
    raise _StepExecutionError(
        message,
        recovery_managed=response.get("recovery_managed") is True,
    )


def direct_executor(api_or_index: Any) -> Executor:
    """A no-rails executor that calls api methods directly (mock / tests).

    Accepts an api object or a prebuilt ``{op: method}`` index.
    """
    idx = api_or_index if isinstance(api_or_index, Mapping) else _build_action_index(api_or_index)

    def run(op: str, params: dict[str, Any]) -> dict[str, Any]:
        fn = idx.get(op)
        if fn is None:
            return {"ok": False, "reason": f"op {op!r} not available on this robot"}
        try:
            result = fn(**params)
        except Exception as exc:  # noqa: BLE001 - convert op failure to structured result
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        ok = result.get("ok", True) if isinstance(result, dict) else True
        return {"ok": ok, "result": result}

    return run


@dataclass(frozen=True)
class _TrackedGraspContext:
    object_name: str
    approach_mm: float
    bind: str | None


def _grasp_confirmation(api: Any, result: Any) -> bool | None:
    """Return adapter-specific grasp confirmation, or ``None`` if unsupported."""
    confirm = getattr(api, "is_grasp_confirmed", None)
    if not callable(confirm):
        return None
    try:
        return bool(confirm(result))
    except Exception as exc:  # noqa: BLE001 - an unsafe/unknown grasp must fail closed
        raise RuntimeError(f"grasp confirmation failed: {type(exc).__name__}: {exc}") from exc


def _retry_unconfirmed_grasp(
    session: Any,
    context: _TrackedGraspContext,
    cfg: SkillExecConfig,
    run_op: Executor,
) -> tuple[dict[str, Any], Any, int]:
    """Home, re-detect, and repeat a grasp after an adapter reports no contact."""
    for attempt in range(1, cfg.max_grasp_retries + 1):
        logger.warning(
            "[runner] grasp of %r was not confirmed; returning home for retry %d/%d",
            context.object_name,
            attempt,
            cfg.max_grasp_retries,
        )
        for op in ("open_gripper", "home"):
            res = run_op(op, {})
            if not res.get("ok"):
                _raise_executor_failure(
                    res,
                    "unknown failure",
                    context=f"grasp retry {attempt} {op} failed",
                )
            if op == "open_gripper":
                time.sleep(max(0.0, cfg.settle_grip_s))

        detection = _track_grasp(session, context.object_name, context.approach_mm, cfg)
        if detection is None:
            raise RuntimeError(f"grasp retry {attempt}: target {context.object_name!r} not detected from home")

        close_res = run_op("close_gripper", {})
        if not close_res.get("ok"):
            _raise_executor_failure(
                close_res,
                "unknown failure",
                context=f"grasp retry {attempt} close_gripper failed",
            )
        close_result = close_res.get("result")
        time.sleep(max(0.0, cfg.settle_grip_s))
        confirmed = _grasp_confirmation(session.api, close_result)
        if confirmed is not False:
            logger.info("[runner] grasp retry %d/%d confirmed", attempt, cfg.max_grasp_retries)
            return detection, close_result, attempt

    raise RuntimeError(f"grasp_not_confirmed: no contact after initial attempt + {cfg.max_grasp_retries} retry")


def run_sequence(
    session: Any,
    steps: list[ActionStep],
    *,
    config: SkillExecConfig | None = None,
    executor: Executor | None = None,
    action_index: Mapping[str, Callable[..., Any]] | None = None,
) -> dict:
    """Execute an action sequence in order, with no per-step LLM.

    Args:
        session: the live ``RobotSession``.
        steps: validated action steps (see ``sequence.parse_sequence``).
        config: servo / detection / offset tuning. Defaults applied if omitted.
        executor: dispatches one primitive op (op, params) -> {ok, result?,
            reason?}. The fast path passes an ability-manager-backed executor so
            ops run through the agent's rails. Defaults to a direct executor
            (built from ``action_index`` or ``session.api``) for mock / tests.
        action_index: legacy — used only to build the default direct executor.

    Returns:
        ``{ok, steps_done, steps:[{i, op, ok, result|reason}], env_keys}``.
        Stops at the first failing step and reports the structured reason
        (rails/RecoveryRail already handled any safe retreat for real runs).
    """
    cfg = config or SkillExecConfig()
    run_op: Executor = executor or direct_executor(action_index or session.api)
    # The env holds only detection bindings (added as tracking/detection steps run).
    # No seeded constants: working heights come from grasp_z/place_z, and any
    # other offset a skill needs is a literal number in its compiled expression.
    env: dict[str, Any] = {}

    # Eye-in-hand pre-scan before any motion (task-agnostic).
    cache = _prescan(session, steps)

    out: list[dict] = []
    holding = False
    tracked_grasp: _TrackedGraspContext | None = None
    ok_all = True
    for i, step in enumerate(steps):
        try:
            params = resolve_params(step.params, env)
            if step.op == TRACK_DETECT:
                det = _track_detect(session, params["object_name"], cfg, cache, occluded=holding)
                if det is None:
                    raise RuntimeError(f"target {params['object_name']!r} not detected")
                if step.bind:
                    env[step.bind] = det
                result: Any = {"detected": det.get("position")}
            elif step.op == TRACK_GRASP:
                det = _track_grasp(session, params["object_name"], float(params["approach_mm"]), cfg)
                if det is None:
                    raise RuntimeError(f"target {params['object_name']!r} not detected")
                if step.bind:
                    env[step.bind] = det
                tracked_grasp = _TrackedGraspContext(
                    object_name=str(params["object_name"]),
                    approach_mm=float(params["approach_mm"]),
                    bind=step.bind,
                )
                result = {"detected": det.get("position"), "grasp_z": det.get("grasp_z")}
            else:
                res = run_op(step.op, params)
                if not res.get("ok"):
                    _raise_executor_failure(res, f"{step.op} failed")
                result = res.get("result")
                if step.bind:
                    # A bind step must yield a usable detection; a detection that
                    # ran but returned ok=False (e.g. no valid depth at the target)
                    # would otherwise silently skip the bind and let a later
                    # "<bind>.field" reference reach a motion tool unresolved
                    # (a cryptic "str + float" crash). Abort here with the real cause.
                    if not (isinstance(result, dict) and result.get("ok")):
                        reason = result.get("reason", "unknown") if isinstance(result, dict) else "no result"
                        target = params.get("object_name", step.bind)
                        raise RuntimeError(
                            f"detection for {target!r} produced no usable result (reason={reason}); "
                            f"later steps read '{step.bind}.<field>' — aborting instead of crashing downstream"
                        )
                    env[step.bind] = normalize_detection(result)
                if step.op in _GRIP_CLOSE_OPS:
                    time.sleep(max(0.0, cfg.settle_grip_s))
                    confirmed = _grasp_confirmation(session.api, result)
                    if confirmed is False and tracked_grasp is not None and cfg.max_grasp_retries > 0:
                        retry_det, result, retry_count = _retry_unconfirmed_grasp(
                            session,
                            tracked_grasp,
                            cfg,
                            run_op,
                        )
                        if tracked_grasp.bind:
                            env[tracked_grasp.bind] = retry_det
                        if isinstance(result, dict):
                            result = {**result, "grasp_retry_attempts": retry_count}
                        else:
                            result = {"result": result, "grasp_retry_attempts": retry_count}
                        confirmed = _grasp_confirmation(session.api, result)
                    elif confirmed is False and tracked_grasp is not None:
                        raise RuntimeError("grasp_not_confirmed: gripper closed without object contact")
                    holding = confirmed is not False
                    tracked_grasp = None
                elif step.op in _GRIP_OPEN_OPS:
                    holding = False
                    tracked_grasp = None
                    time.sleep(max(0.0, cfg.settle_grip_s))
            out.append({"i": i, "op": step.op, "ok": True, "result": result})
            logger.info("[runner] step %d ok: %s(%s)", i, step.op, params)
        except Exception as exc:  # noqa: BLE001 - surface as structured failure
            logger.warning("[runner] step %d failed: %s(%s): %s", i, step.op, step.params, exc)
            if isinstance(exc, _StepExecutionError) and exc.recovery_managed:
                logger.info("[runner] recovery already handled by the ability rail stack")
            else:
                _safe_retreat(session)
            out.append({"i": i, "op": step.op, "ok": False, "reason": f"{type(exc).__name__}: {exc}"})
            ok_all = False
            break

    return {"ok": ok_all, "steps_done": len(out), "steps": out, "env_keys": sorted(env)}


def _safe_retreat(session: Any) -> None:
    """Best-effort safe-state recovery after a failed step (never raises).

    Compound track ops bypass the rail-aware executor, so a track failure
    must reproduce RecoveryRail's fallback here. Confirmed payloads stay
    gripped; false or unavailable payload state keeps the conservative
    release-then-home behaviour.
    """
    holding_payload = read_holding_payload(session)
    if holding_payload is True:
        logger.warning("[runner] safe retreat: preserving confirmed payload during recovery home")
    released_ok, home_ok = recover_session(
        session,
        release=holding_payload is not True,
        home=True,
        log_prefix="[runner] safe retreat",
    )
    logger.info("[runner] safe retreat complete: released_ok=%s home_ok=%s", released_ok, home_ok)
