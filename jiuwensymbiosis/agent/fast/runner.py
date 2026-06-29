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
  * one compound op, ``track_detect``: real-time-track a named target until it
    settles and bind its (task-agnostic) detection result; fall back to a home
    pre-scan when the wrist camera is occluded (gripper holding something).

The whole pick/place *meaning* lives in the SKILL.md the LLM compiled — never
here. Adding a new task = adding a SKILL.md; this runner is unchanged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from jiuwensymbiosis.agent.fast.realtime.binding import ServoBinding
from jiuwensymbiosis.agent.fast.realtime.servo import ServoConfig, ServoController, ServoResult
from jiuwensymbiosis.agent.fast.realtime.tracking import BackgroundTracker
from jiuwensymbiosis.agent.fast.sequence import (
    TRACK_DETECT,
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
    detect_staleness_s: float = 2.0  # target older than this counts as lost
    first_target_timeout_s: float = 8.0  # wait this long for the first detection
    settle_grip_s: float = 0.5  # pause after a gripper command (let it actuate)
    servo: ServoConfig = field(default_factory=ServoConfig)  # track-loop tuning


def _detect_once(api: Any, object_name: str) -> dict[str, Any] | None:
    """One detection → normalized binding dict, or ``None`` if not detected."""
    try:
        gi = api.get_grasp_info_simple(object_name)
    except Exception as exc:  # noqa: BLE001 - detection may raise; treat as miss
        logger.debug("[runner] detection raised for %r: %s", object_name, exc)
        return None
    if not isinstance(gi, dict) or not gi.get("ok"):
        return None
    return normalize_detection(gi)


def _prescan(session: Any, steps: list[ActionStep]) -> dict[str, dict[str, Any]]:
    """At the home pose, detect every ``track_detect`` target once and cache it.

    Eye-in-hand: a target grasped later occludes the wrist camera, so its
    position must be read now. Best-effort — a target not seen at home is simply
    absent (the live track will try again). Task-agnostic: just caches whatever
    objects the sequence names.
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

    tracker = BackgroundTracker(
        lambda: _detect_once(session.api, object_name),
        max_hz=cfg.detect_hz,
        staleness_s=cfg.detect_staleness_s,
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
        first = cast(Pose, tracker.latest_target())
        obj0x, obj0y = float(first["x"]), float(first["y"])

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

        res: ServoResult = ServoController(binding.read_pose, binding.servo_to, track_target, config=cfg.servo).run()
        logger.info(
            "[runner] track_detect %r: %s in %d ticks / %.2fs", object_name, res.reason, res.ticks, res.elapsed_s
        )
        latest = tracker.latest_target()
        return dict(latest) if latest is not None else None
    finally:
        tracker.stop()


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
    # The env holds only detection bindings (added as track_detect steps run).
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
            else:
                res = run_op(step.op, params)
                if not res.get("ok"):
                    raise RuntimeError(res.get("reason") or f"{step.op} failed")
                result = res.get("result")
                if step.bind and isinstance(result, dict) and result.get("ok"):
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
    """Best-effort move to a safe pose after a failed step (never raises)."""
    try:
        session.api.home()
    except Exception as exc:  # noqa: BLE001 - safe retreat must never raise
        logger.warning("[runner] safe retreat home() failed: %s", exc)
