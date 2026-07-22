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
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController, ServoResult
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker
from jiuwensymbiosis.agent.fast.sequence import (
    TRACK_DETECT,
    TRACK_GRASP,
    ActionStep,
    normalize_detection,
    resolve_params,
)
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
    servo: ServoConfig = field(default_factory=ServoConfig)  # track-loop tuning

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
        if isinstance(self.max_re_align_iters, bool) or not isinstance(self.max_re_align_iters, int):
            raise ValueError(f"SkillExecConfig.max_re_align_iters must be int, got {self.max_re_align_iters!r}.")
        if self.max_re_align_iters < 0:
            raise ValueError(f"SkillExecConfig.max_re_align_iters must be >= 0, got {self.max_re_align_iters}.")


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

        res: ServoResult = ServoController(
            binding.read_pose,
            binding.servo_to,
            track_target,
            config=cfg.servo,
            target_is_live=target_is_live,
        ).run()
        logger.info(
            "[runner] track_detect %r: %s in %d ticks / %.2fs", object_name, res.reason, res.ticks, res.elapsed_s
        )
        if not res.ok:
            raise RuntimeError(f"track_detect failed: {res.reason}")
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
    tracker = BackgroundTracker(
        lambda: _detect_once(api, object_name, require_grasp=True),
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

        approach = ServoController(
            binding.read_pose,
            binding.servo_to,
            approach_target,
            config=cfg.servo,
            target_is_live=target_is_live,
        ).run()
        logger.info(
            "[runner] track_grasp %r approach: %s in %d ticks / %.2fs (detections=%d)",
            object_name,
            approach.reason,
            approach.ticks,
            approach.elapsed_s,
            tracker.detections,
        )
        if not approach.ok:
            raise RuntimeError(f"track_grasp approach failed: {approach.reason}")
        descend = ServoController(
            binding.read_pose,
            binding.servo_to,
            descend_target,
            config=cfg.servo,
            target_is_live=target_is_live,
        ).run()
        logger.info(
            "[runner] track_grasp %r descend: %s in %d ticks / %.2fs (detections=%d)",
            object_name,
            descend.reason,
            descend.ticks,
            descend.elapsed_s,
            tracker.detections,
        )
        if not descend.ok:
            raise RuntimeError(f"track_grasp descend failed: {descend.reason}")

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
            re_descend = ServoController(
                binding.read_pose,
                binding.servo_to,
                descend_target,
                config=cfg.servo,
                target_is_live=target_is_live,
            ).run()
            if not re_descend.ok:
                raise RuntimeError(f"track_grasp post-descend re-align failed: {re_descend.reason}")
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
            "[runner] track_grasp %r final target: position=%s grasp_z=%s detections=%d",
            object_name,
            latest.get("position"),
            latest.get("grasp_z"),
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
                result = {"detected": det.get("position"), "grasp_z": det.get("grasp_z")}
            else:
                res = run_op(step.op, params)
                if not res.get("ok"):
                    raise RuntimeError(res.get("reason") or f"{step.op} failed")
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
                    holding = True
                    time.sleep(max(0.0, cfg.settle_grip_s))
                elif step.op in _GRIP_OPEN_OPS:
                    holding = False
                    time.sleep(max(0.0, cfg.settle_grip_s))
            out.append({"i": i, "op": step.op, "ok": True, "result": result})
            logger.info("[runner] step %d ok: %s(%s)", i, step.op, params)
        except Exception as exc:  # noqa: BLE001 - surface as structured failure
            logger.warning("[runner] step %d failed: %s(%s): %s", i, step.op, step.params, exc)
            _safe_retreat(session)
            out.append({"i": i, "op": step.op, "ok": False, "reason": f"{type(exc).__name__}: {exc}"})
            ok_all = False
            break

    return {"ok": ok_all, "steps_done": len(out), "steps": out, "env_keys": sorted(env)}


def _safe_retreat(session: Any) -> None:
    """Best-effort safe-state recovery after a failed step (never raises).

    Compound track ops bypass the rail-aware executor, so a track failure
    must reproduce RecoveryRail's release-then-home fallback here. Release
    first to avoid dragging a held object while homing.
    """
    released_ok = False
    for release_name in ("deactivate_suction", "open_gripper"):
        release_fn = getattr(session.api, release_name, None)
        if not callable(release_fn):
            continue
        try:
            result = release_fn()
            if isinstance(result, dict) and result.get("ok") is False:
                raise RuntimeError(str(result.get("reason") or "returned ok=False"))
            released_ok = True
            logger.info("[runner] safe retreat: %s succeeded", release_name)
            break
        except Exception as exc:  # noqa: BLE001 - every recovery action is independent
            logger.warning("[runner] safe retreat: %s failed: %s", release_name, exc)

    if not released_ok:
        env = getattr(session, "env", None)
        set_ee = getattr(env, "set_end_effector", None)
        if callable(set_ee):
            try:
                result = set_ee(False)
                if isinstance(result, dict) and result.get("ok") is False:
                    raise RuntimeError(str(result.get("reason") or "returned ok=False"))
                released_ok = True
                logger.info("[runner] safe retreat: env.set_end_effector(False) succeeded")
            except Exception as exc:  # noqa: BLE001 - home must still be attempted
                logger.warning("[runner] safe retreat: env release fallback failed: %s", exc)

    home_ok = False
    home = getattr(session.api, "home", None)
    if callable(home):
        try:
            result = home()
            if isinstance(result, dict) and result.get("ok") is False:
                raise RuntimeError(str(result.get("reason") or "returned ok=False"))
            home_ok = True
            logger.info("[runner] safe retreat: home() succeeded")
        except Exception as exc:  # noqa: BLE001 - safe retreat must never raise
            logger.warning("[runner] safe retreat: home() failed: %s", exc)
    logger.info("[runner] safe retreat complete: released_ok=%s home_ok=%s", released_ok, home_ok)
