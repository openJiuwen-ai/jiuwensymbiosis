# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""World-state vocabulary — what actions and skills declare and reason over.

An action's planning contract has three independent parts, because the things a
plan can get wrong are of three kinds.

**1. Robot self-state — a closed token vocabulary.**
Whether the end effector holds something, whether the body is home. These are
properties of the robot, not of the task, so a small closed set covers every
task: pressing a doorbell, opening a door, wiping, pouring, inspecting, or
pick-and-place. A task that does not manipulate simply never mentions them.

**2. Location freshness — scoped to the referent, tracked automatically.**
"Do I have a usable position for the thing this step is about?" is per-object,
and the object identity is *already* in the sequence: the ``bind`` name. So a
step that produces a location marks its bind **fresh**; an action that moves the
base marks every prior location **stale**; a step that consumes a location
requires a fresh one. No vocabulary is invented, and it generalises with no
role bias — one referent (``doorbell``) or two (``box`` and ``table``) work the
same way.

**3. Access — also scoped to the referent, also tracked automatically.**
"Can I reach the thing at all, or is something in the way?" A shut cabinet, a
closed drawer, a lid, a crate stacked on top, a locked door: one situation, many
nouns. It is not robot self-state (it is a fact about one object, not about the
robot) and not location freshness (the position is known perfectly well — the
hand just cannot get there). So it is its own part, and it needs exactly one
declaration: an action says ``opens_access`` / ``closes_access``, meaning it
un-blocks or re-blocks the thing it acts on.

Nothing else is annotated, because the sequence already says which steps reach
through a barrier: the task's own spatial qualifier (``reference`` +
``relation`` over :data:`CONTAINMENT_RELATIONS`) says the target is *inside*
something, and :data:`PAYLOAD_TOKENS` says which steps actually manipulate as
opposed to merely look or drive. Looking at a shut cabinet is how you find it;
reaching into one is the error. See ``agent/fast/sequence.py`` for the two rules
this supports — a plan that contradicts itself, and a plan contradicted by what
the pre-plan look saw.

Deliberately *not* modelled here: task roles such as "target" vs "destination".
Those describe pick-and-place, and would over-fit the framework to that one task
family — a doorbell has no destination, an inspection has no payload.

The contract never encodes an order. It states pre-conditions and effects, so a
planner can **derive** a legal order; any permutation whose pre-conditions hold
is accepted.

Adding a token:
  1. Append it here (and to ``EXCLUDES`` if it rules a sibling out).
  2. Annotate the actions that require/provide/invalidate it.
  3. Teach ``WorldState`` how to observe it.
"""

from __future__ import annotations

from collections.abc import Iterable

# Closed vocabulary of robot self-state. An unknown token in a @robot_tool /
# SKILL.md annotation is a hard error, not a silently ignored string — a typo
# would otherwise make a pre-condition unsatisfiable and the planner would loop
# trying to satisfy it.
KNOWN_STATE_TOKENS: frozenset[str] = frozenset(
    {
        "payload.held",  # the end effector is holding something
        "payload.clear",  # the end effector is empty
        "payload.stowed",  # a held payload is in a pose safe to travel with
        "body.home",  # the body is at its safe home posture
    }
)

# The tokens that say the end effector's contents changed. An action providing one
# MANIPULATED something; every other action only looked, drove, or posed. That is the
# line the access check needs (looking into a shut cabinet is how you discover it is
# shut; reaching into one is the error), and reading it off the vocabulary keeps the
# check from needing a hard-coded list of grasp/place action names.
PAYLOAD_TOKENS: frozenset[str] = frozenset({"payload.held", "payload.clear", "payload.stowed"})

# The relations that mean "my target is enclosed by the reference", as opposed to merely
# located by it. Subset of ``perception/scene3d.py:SPATIAL_RELATIONS`` (on / under / in /
# beside / near); ``tests/unit_tests/api/test_state.py`` pins that it stays one.
#   in    — the apple IN the drawer: the drawer must be open.
#   under — the box UNDER the crate: the crate must come off first.
# ``on`` / ``beside`` / ``near`` enclose nothing. They also appear the other way round
# ("the table UNDER the cup" identifies a surface by what sits on it) — harmless, because
# only manipulating steps are access-checked and that phrasing appears on sensing ones.
CONTAINMENT_RELATIONS: frozenset[str] = frozenset({"in", "under"})

# Producing the key token clears the listed ones. Not symmetric: a stowed payload
# is still held, but an empty end effector can be neither held nor stowed.
EXCLUDES: dict[str, frozenset[str]] = {
    "payload.held": frozenset({"payload.clear"}),
    "payload.stowed": frozenset({"payload.clear"}),
    "payload.clear": frozenset({"payload.held", "payload.stowed"}),
}


class UnknownStateToken(ValueError):
    """An annotation referenced a token outside ``KNOWN_STATE_TOKENS``."""


def validate_tokens(tokens: Iterable[str], *, field: str, owner: str) -> tuple[str, ...]:
    """Return ``tokens`` as a tuple, raising if any is not in the closed vocabulary.

    Args:
        tokens: the annotated token strings.
        field: which annotation they came from (``requires`` / ``provides`` / ``invalidates``),
            used in the error message.
        owner: the action or skill name, used in the error message.
    """
    out = tuple(str(t) for t in tokens)
    unknown = sorted(set(out) - KNOWN_STATE_TOKENS)
    if unknown:
        raise UnknownStateToken(
            f"{owner}.{field} declares unknown state token(s) {unknown}. "
            f"Add them to KNOWN_STATE_TOKENS in jiuwensymbiosis/api/state.py first. "
            f"Known: {sorted(KNOWN_STATE_TOKENS)}"
        )
    return out


def apply_effects(
    state: Iterable[str],
    *,
    provides: Iterable[str] = (),
    invalidates: Iterable[str] = (),
) -> frozenset[str]:
    """Advance a self-state set by one action's effects.

    Order matters: ``invalidates`` is applied first, then ``provides`` (plus the
    ``EXCLUDES`` clean-up), so an action that both invalidates and re-provides a
    token ends up with it held rather than dropped.
    """
    result = set(state) - set(invalidates)
    for token in provides:
        result -= EXCLUDES.get(token, frozenset())
        result.add(token)
    return frozenset(result)


def missing_requirements(state: Iterable[str], requires: Iterable[str]) -> tuple[str, ...]:
    """Required tokens not present in ``state`` (empty tuple = satisfied)."""
    held = set(state)
    return tuple(sorted(t for t in requires if t not in held))


def contradicted_requirements(state: Iterable[str], requires: Iterable[str]) -> tuple[str, ...]:
    """Required tokens the state carries positive evidence *against*.

    The runtime counterpart of :func:`missing_requirements`, and deliberately
    weaker. At plan time the state is simulated forward from a known start, so an
    absent token means "no preceding action established it" — a real gap in the
    plan. At run time the state is *measured*, and a body that cannot report a
    token simply omits it; treating that ignorance as falsehood would re-plan
    forever. So only a token ruled out by something the state actually holds
    counts as a deviation.
    """
    ruled_out = {t for token in set(state) for t in EXCLUDES.get(token, frozenset())}
    return tuple(sorted(set(requires) & ruled_out))
