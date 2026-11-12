# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``WorldState`` — what is true right now, in the planner's own vocabulary.

A planner can only derive an order if it knows where it is starting from. This
assembles that starting point from three sources, in order of authority:

1. **Observation** — what the env can actually measure (``holding_payload``,
   pose, joints). Authoritative: it overrides belief.
2. **Belief** — tokens established by the actions run so far
   (``ExecutionMemory.self_state``). Covers what no sensor reports, such as
   whether a carried payload has been raised to travel height.
3. **Proprioception + scene** — pose / joints / extra from the env observation,
   and sensed locations from ``ExecutionMemory``. Not tokens; context the planner
   reasons over directly.

Everything is best-effort: a body that cannot report something simply omits it,
and an omitted token means *unknown*, never *false*. That distinction matters —
``parse_sequence`` disables self-state checking when the state is unknown rather
than rejecting plans against a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from jiuwensymbiosis.api.memory import ExecutionMemory
from jiuwensymbiosis.api.state import ruled_out_by

logger = logging.getLogger(__name__)


def _observed_payload(env: Any) -> frozenset[str]:
    """``payload.held`` / ``payload.clear`` from the env, or empty when unreported."""
    try:
        value = getattr(env, "holding_payload", None)
    except Exception as exc:
        logger.debug("[world_state] holding_payload unreadable: %s", exc)
        return frozenset()
    if not isinstance(value, bool):
        return frozenset()
    return frozenset({"payload.held"} if value else {"payload.clear"})


def current_tokens(session: Any) -> frozenset[str]:
    """The self-state tokens true right now — belief corrected by observation.

    Split out of :meth:`WorldState.snapshot` because the runner re-reads this
    before every step: it touches nothing but ``env.holding_payload``, whereas a
    full snapshot grabs an observation. Never raises.
    """
    api = getattr(session, "api", None)
    memory = getattr(api, "memory", None)
    believed = memory.self_state if isinstance(memory, ExecutionMemory) else frozenset()
    env = getattr(session, "env", None)
    try:
        payloads = getattr(env, "effector_payloads", None)
    except Exception:
        payloads = {}  # Failed authoritative readback must not resurrect old beliefs.
    observed = _observed_payload(env)
    if isinstance(payloads, dict):
        believed = frozenset(t for t in believed if not t.startswith("payload."))
        observed = frozenset(
            f"payload.{'held' if v else 'clear'}:{n}" for n, v in payloads.items() if isinstance(v, bool)
        )
        if any(v is True for v in payloads.values()):
            observed |= {"payload.held"}
        elif payloads and all(v is False for v in payloads.values()):
            observed |= {"payload.clear"}
    # Observation wins, but only over what it actually contradicts: ``holding_payload`` is a
    # bool, so it can rule out "empty" or "holding" and says nothing about ``payload.stowed``.
    ruled_out = ruled_out_by(observed)
    return frozenset({t for t in believed if t not in ruled_out} | observed)


def _with_reachability(locations: list[dict[str, Any]], api: Any) -> list[dict[str, Any]]:
    """Annotate each sensed location with whether the body can reach it FROM WHERE IT IS NOW.

    This is the channel that stays fresh — it is re-read on every re-plan, unlike the pre-run
    scene — so without this the planner's up-to-date positions carried no reach information at
    all and its reach-annotated positions were stale. A verdict of ``None`` means the judge
    could not decide, and the key is then OMITTED: unknown is not "out of reach".
    """
    if "planning.reachability" not in (getattr(api, "capabilities", None) or frozenset()):
        return locations
    for loc in locations:
        position = loc.get("position_mm")
        if not position:
            continue
        try:
            verdict = api.check_reachable({"position": position})
        except Exception as exc:
            logger.debug("[world_state] check_reachable(%s) failed: %s", loc.get("referent"), exc)
            continue
        if verdict is not None:
            loc["reachable"] = bool(verdict)
    return locations


def _safe_attr(obj: Any, name: str) -> Any:
    """Read an optional env property without letting a body's getter sink the snapshot."""
    try:
        return getattr(obj, name, None)
    except Exception as exc:
        logger.debug("[world_state] %s unreadable: %s", name, exc)
        return None


def _reach_prior(api: Any) -> dict[str, Any] | None:
    """The body's URDF reach model, when it has one. ``None`` = it cannot say."""
    if "planning.reachability" not in (getattr(api, "capabilities", None) or frozenset()):
        return None
    try:
        prior = api.describe_reach()
    except Exception as exc:
        logger.debug("[world_state] describe_reach failed: %s", exc)
        return None
    return prior if isinstance(prior, dict) else None


# What each goto_xyzr orientation policy means, spelled out in the prompt: the policy NAME
# alone does not tell a planner that "preserve" will not point the tool down.
_ORIENTATION_HINTS = {
    "preserve": "沿用当前倾角，不会自动朝下；要朝下必须显式传 top_down",
    "top_down": "工具朝下",
    "grasp": "本体标定的抓取倾角",
}


@dataclass(frozen=True)
class WorldState:
    """A snapshot of the robot and its surroundings, for planning."""

    tokens: frozenset[str] = frozenset()
    locations: list[dict[str, Any]] = field(default_factory=list)
    pose: dict[str, Any] | None = None
    joints: list[float] | None = None
    # "deg" / "rad" / None. Carried beside the numbers because the numbers alone are
    # ambiguous: 1.5 is a nudge in degrees and 86 degrees in radians.
    joint_units: str | None = None
    # Which orientation_policy goto_xyzr applies when the caller omits it. The tool schema
    # can only say "there is a default" — implements() runs before any config exists.
    default_orientation_policy: str | None = None
    # The body's reach envelope, so a plan can be built inside it instead of being bounced
    # off SafetyRail one step at a time. workspace_bounds/z_min_safe are the CONFIG-stated
    # Cartesian box (a fixed arm has one; a mobile body does not); reach_prior is the
    # URDF-derived reach model, for bodies that ship a URDF. Either may be absent.
    workspace_bounds: tuple[float, float, float, float] | None = None
    z_min_safe: float | None = None
    reach_prior: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()

    @classmethod
    def snapshot(cls, session: Any) -> WorldState:
        """Read the current state off a live session. Never raises."""
        env = getattr(session, "env", None)
        api = getattr(session, "api", None)
        memory = getattr(api, "memory", None)
        memory = memory if isinstance(memory, ExecutionMemory) else ExecutionMemory()
        tokens = current_tokens(session)

        pose = joints = None
        extra: dict[str, Any] = {}
        try:
            obs = env.get_observation() if env is not None else None
        except Exception as exc:
            logger.debug("[world_state] get_observation failed: %s", exc)
            obs = None
        if obs is not None:
            pose = obs.pose
            joints = obs.joints
            # rgb / depth are large arrays and never part of a prompt; keep the rest.
            extra = {k: v for k, v in (obs.extra or {}).items() if not hasattr(v, "shape")}

        caps = tuple(sorted(getattr(env, "capabilities", None) or ()))
        return cls(
            tokens=tokens,
            locations=_with_reachability(memory.describe()["locations"], api),
            pose=pose,
            joints=joints,
            joint_units=getattr(env, "joint_units", None),
            default_orientation_policy=getattr(env, "default_orientation_policy", None),
            workspace_bounds=_safe_attr(env, "workspace_bounds"),
            z_min_safe=_safe_attr(env, "z_min_safe"),
            reach_prior=_reach_prior(api),
            extra=extra,
            capabilities=caps,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-able form for the planner prompt and the ``state`` CLI."""
        return {
            "tokens": sorted(self.tokens),
            "locations": self.locations,
            "pose": self.pose,
            "joints": self.joints,
            "joint_units": self.joint_units,
            "default_orientation_policy": self.default_orientation_policy,
            "workspace_bounds": list(self.workspace_bounds) if self.workspace_bounds else None,
            "z_min_safe": self.z_min_safe,
            "reach_prior": self.reach_prior,
            "extra": self.extra,
            "capabilities": list(self.capabilities),
        }

    def as_prompt_block(self) -> str:
        """Render the snapshot as the 【当前状态】 block of a planning prompt.

        Empty string when nothing is known, so the caller can concatenate freely
        without emitting a header over an empty body.
        """
        lines: list[str] = []
        if self.tokens:
            lines.append("状态：" + ", ".join(sorted(self.tokens)))
        if self.pose:
            pose = ", ".join(f"{k}={float(v):.0f}" for k, v in self.pose.items() if isinstance(v, (int, float)))
            if pose:
                lines.append(f"位姿：{pose}")
        envelope: list[str] = []
        if self.workspace_bounds:
            xmin, ymin, xmax, ymax = self.workspace_bounds
            envelope.append(f"XY 可达框 x∈[{xmin:.0f},{xmax:.0f}] y∈[{ymin:.0f},{ymax:.0f}]mm")
        if self.z_min_safe is not None:
            envelope.append(f"Z 下限 {self.z_min_safe:.0f}mm")
        if envelope:
            # Stated at plan time so a target outside it is never planned, instead of being
            # rejected by SafetyRail one step into the run.
            lines.append("工作范围：" + "，".join(envelope) + "（超出即被安全护栏拒绝，请在范围内规划）")
        if self.reach_prior:
            lines.append(f"可达模型：{self.reach_prior}")
        if self.default_orientation_policy:
            hint = _ORIENTATION_HINTS.get(self.default_orientation_policy, "")
            suffix = f"（{hint}）" if hint else ""
            lines.append(f"goto_xyzr 省略 orientation_policy 时用：{self.default_orientation_policy}{suffix}")
        if self.joints:
            # Name the unit, or say it is unknown — never let a bare number imply one.
            unit = self.joint_units or "单位未声明"
            lines.append(f"关节({unit})：" + ", ".join(f"{float(j):.2f}" for j in self.joints))
        for loc in self.locations:
            pos = loc.get("position_mm")
            pos_s = f"[{', '.join(f'{float(c):.0f}' for c in pos)}]mm" if isinstance(pos, (list, tuple)) else "?"
            reach = loc.get("reachable")
            # Only stated when the body could actually decide; absent means unknown.
            reach_s = "" if reach is None else ("，当前位置够得着" if reach else "，当前位置够不着，需先移动过去")
            lines.append(f"已感知 {loc['referent']}：{pos_s}（{loc['sensed_by']}，{loc['age_s']}s 前{reach_s}）")
        if not lines:
            return ""
        return "【当前状态】\n" + "\n".join(f"  {line}" for line in lines) + "\n\n"
