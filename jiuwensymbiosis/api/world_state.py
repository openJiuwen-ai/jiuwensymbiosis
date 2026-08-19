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
    observed = _observed_payload(getattr(session, "env", None))
    # Observation wins: drop believed payload tokens the env contradicts.
    if observed:
        believed = frozenset(t for t in believed if not t.startswith("payload."))
    return frozenset(believed | observed)


@dataclass(frozen=True)
class WorldState:
    """A snapshot of the robot and its surroundings, for planning."""

    tokens: frozenset[str] = frozenset()
    locations: list[dict[str, Any]] = field(default_factory=list)
    pose: dict[str, Any] | None = None
    joints: list[float] | None = None
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
            locations=memory.describe()["locations"],
            pose=pose,
            joints=joints,
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
        if self.joints:
            lines.append("关节：" + ", ".join(f"{float(j):.2f}" for j in self.joints))
        for loc in self.locations:
            pos = loc.get("position_mm")
            pos_s = f"[{', '.join(f'{float(c):.0f}' for c in pos)}]mm" if isinstance(pos, (list, tuple)) else "?"
            lines.append(f"已感知 {loc['referent']}：{pos_s}（{loc['sensed_by']}，{loc['age_s']}s 前）")
        if not lines:
            return ""
        return "【当前状态】\n" + "\n".join(f"  {line}" for line in lines) + "\n\n"
