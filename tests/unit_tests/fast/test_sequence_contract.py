# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``parse_sequence`` contract checks — pre-conditions and effects, never an order.

The load-bearing test here is ``test_every_equivalent_order_is_accepted``: the
validator exists so a planner can derive an order, so it must reject only what it
can prove wrong. If it ever starts accepting exactly one arrangement it has become
a written-down recipe again, which is what this whole layer removes.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence
from jiuwensymbiosis.api.decorators import robot_tool


class _Body:
    """A body exposing one action per contract shape the validator reasons about."""

    @robot_tool(desc="sense where something is", produces_location=True)
    def detect(self, object_name: str) -> dict: ...

    @robot_tool(
        desc="drive up to the sensed thing", consumes_location=True, produces_location=True, invalidates_locations=True
    )
    def approach(self) -> dict: ...

    @robot_tool(desc="drive the base", invalidates_locations=True)
    def drive(self, dx_m: float) -> dict: ...

    @robot_tool(desc="grip", requires=["payload.clear"], provides=["payload.held"])
    def grip(self) -> dict: ...

    @robot_tool(desc="release", provides=["payload.clear"])
    def release(self) -> dict: ...

    @robot_tool(desc="go to coordinates", invalidates=["body.home"])
    def goto(self, x: float, y: float, z: float) -> dict: ...

    # Takes the sensed thing by name, the way dual_arm_grasp(target) does — a body that
    # declares no param cannot legally be handed one (see TestDeclaredParamsOnly).
    @robot_tool(desc="press it", consumes_location=True)
    def press(self, target: str | None = None) -> dict: ...

    @robot_tool(
        desc="sense with a declared result shape",
        produces_location=True,
        returns={"type": "object", "properties": {"position": {"type": "array"}, "grasp_z": {"type": "number"}}},
    )
    def detect_typed(self, object_name: str) -> dict: ...

    # Reports a DIRECTION, not a position: readable data, but nothing a
    # consumes_location step may act on.
    @robot_tool(
        desc="report which way something lies",
        returns={"type": "object", "properties": {"bearing_rad": {"type": "number"}}},
    )
    def bearing(self, object_name: str) -> dict: ...

    @robot_tool(desc="turn by an angle", invalidates_locations=True)
    def turn(self, dyaw_rad: float) -> dict: ...


def _index() -> dict:
    body = _Body()
    return {
        name: getattr(body, name)
        for name in ("detect", "approach", "drive", "grip", "release", "goto", "press",
                     "detect_typed", "bearing", "turn")
    }


def _parse(steps, **kw):
    return parse_sequence(steps, allowed_ops=_index(), **kw)


CLEAR = frozenset({"payload.clear"})


# --------------------------------------------------------------------------- #
# The invariant: validation must not become prescription
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "order",
    [
        # sense, drive up, grip
        [{"op": "detect", "params": {"object_name": "x"}}, {"op": "approach"}, {"op": "grip"}],
        # release first (already clear — harmless), then the same work
        [{"op": "release"}, {"op": "detect", "params": {"object_name": "x"}}, {"op": "approach"}, {"op": "grip"}],
        # an unrelated move interleaved before sensing
        [
            {"op": "drive", "params": {"dx_m": 1}},
            {"op": "detect", "params": {"object_name": "x"}},
            {"op": "approach"},
            {"op": "grip"},
        ],
        # sense twice — wasteful, not wrong
        [
            {"op": "detect", "params": {"object_name": "x"}},
            {"op": "detect", "params": {"object_name": "x"}},
            {"op": "approach"},
            {"op": "grip"},
        ],
    ],
)
def test_every_equivalent_order_is_accepted(order):
    assert len(_parse(order, initial_state=CLEAR)) == len(order)


# --------------------------------------------------------------------------- #
# Robot self-state
# --------------------------------------------------------------------------- #
def test_unmet_requirement_names_an_action_that_would_fix_it():
    with pytest.raises(SequenceError) as exc:
        _parse([{"op": "grip"}, {"op": "grip"}], initial_state=CLEAR)
    msg = str(exc.value)
    assert "payload.clear" in msg
    assert "release" in msg  # the repair hint, not just a complaint


def test_self_state_check_is_off_when_the_state_is_unknown():
    # No initial_state: asserting pre-conditions against a guessed state would
    # reject valid plans, so the check must not run at all.
    _parse([{"op": "grip"}, {"op": "grip"}])


def test_release_makes_a_second_grip_legal():
    _parse([{"op": "grip"}, {"op": "release"}, {"op": "grip"}], initial_state=CLEAR)


# --------------------------------------------------------------------------- #
# Location freshness
# --------------------------------------------------------------------------- #
def test_coordinates_sensed_before_a_base_move_are_rejected():
    with pytest.raises(SequenceError, match="stale"):
        _parse(
            [
                {"op": "detect", "params": {"object_name": "x"}, "bind": "obj"},
                {"op": "drive", "params": {"dx_m": 1.0}},
                {"op": "goto", "params": {"x": "obj.x", "y": "obj.y", "z": "obj.z"}},
            ]
        )


def test_re_sensing_after_the_move_makes_them_usable_again():
    _parse(
        [
            {"op": "detect", "params": {"object_name": "x"}, "bind": "obj"},
            {"op": "drive", "params": {"dx_m": 1.0}},
            {"op": "detect", "params": {"object_name": "x"}, "bind": "obj"},
            {"op": "goto", "params": {"x": "obj.x", "y": "obj.y", "z": "obj.z"}},
        ]
    )


def test_cache_consuming_action_needs_something_sensed():
    with pytest.raises(SequenceError) as exc:
        _parse([{"op": "press"}])
    assert "detect" in str(exc.value)


def test_a_sensing_step_without_a_bind_still_feeds_the_cache():
    # approach re-senses into the api cache; press consumes it implicitly.
    _parse([{"op": "detect", "params": {"object_name": "x"}}, {"op": "approach"}, {"op": "press"}])


# --------------------------------------------------------------------------- #
# Result fields
# --------------------------------------------------------------------------- #
def test_reading_a_field_the_action_does_not_return_is_rejected():
    with pytest.raises(SequenceError, match="does not return"):
        _parse(
            [
                {"op": "detect_typed", "params": {"object_name": "x"}, "bind": "obj"},
                {"op": "goto", "params": {"x": "obj.nonexistent", "y": 0, "z": 0}},
            ]
        )


def test_declared_and_synthetic_fields_are_both_readable():
    _parse(
        [
            {"op": "detect_typed", "params": {"object_name": "x"}, "bind": "obj"},
            {"op": "goto", "params": {"x": "obj.position[0]", "y": "obj.y", "z": "obj.grasp_z + 40"}},
        ]
    )


def test_unknown_result_shape_skips_field_checking():
    # `detect` publishes no returns schema — unknown shape must degrade to no
    # checking rather than reject every field.
    _parse(
        [
            {"op": "detect", "params": {"object_name": "x"}, "bind": "obj"},
            {"op": "goto", "params": {"x": "obj.whatever", "y": 0, "z": 0}},
        ]
    )


# --------------------------------------------------------------------------- #
# Task-shape neutrality
# --------------------------------------------------------------------------- #
def test_a_task_with_no_payload_and_no_destination_validates():
    # "press the doorbell": one referent, nothing grasped, nowhere to put anything.
    steps = _parse(
        [
            {"op": "detect", "params": {"object_name": "doorbell"}, "bind": "doorbell"},
            {"op": "approach"},
            {"op": "press"},
        ],
        initial_state=CLEAR,
    )
    assert [s.op for s in steps] == ["detect", "approach", "press"]


def test_pressing_with_a_move_in_between_is_rejected_as_stale():
    with pytest.raises(SequenceError):
        _parse(
            [
                {"op": "detect", "params": {"object_name": "doorbell"}, "bind": "doorbell"},
                {"op": "drive", "params": {"dx_m": 2.0}},
                {"op": "press"},
            ],
            initial_state=CLEAR,
        )


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #
def test_a_bare_name_vocabulary_carries_no_contract():
    # Callers that pass names only (no bound methods) get vocabulary checking alone.
    parse_sequence([{"op": "grip"}, {"op": "grip"}], allowed_ops={"grip"}, initial_state=CLEAR)


# --------------------------------------------------------------------------- #
# Staleness is about SENSED POSITIONS, not about every value a step returns
# --------------------------------------------------------------------------- #
class TestWhatStalenessCovers:
    def test_a_non_location_reading_is_ordinary_data(self):
        """A bearing (or a pose read) can be consumed by the next step.

        Staleness exists because a coordinate measured from a standpoint the body has
        left is dangerous. A value that is not a position has no such hazard, and
        treating it as permanently stale would mean it could never be read at all.
        """
        _parse([
            {"op": "bearing", "params": {"object_name": "crate"}, "bind": "s"},
            {"op": "turn", "params": {"dyaw_rad": "s.bearing_rad"}},
        ])

    def test_a_bearing_does_not_satisfy_a_step_that_needs_a_position(self):
        """Knowing WHICH WAY something lies is not knowing WHERE it is.

        If a direction counted as a location, ``bearing`` → ``press`` would compile and
        then fail on the robot when the consuming step found an empty cache.
        """
        with pytest.raises(SequenceError, match="none is current"):
            _parse([
                {"op": "bearing", "params": {"object_name": "crate"}},
                {"op": "press", "params": {}},
            ])

    def test_a_sensed_position_is_still_staled_by_motion(self):
        with pytest.raises(SequenceError, match="stale"):
            _parse([
                {"op": "detect_typed", "params": {"object_name": "crate"}, "bind": "c"},
                {"op": "drive", "params": {"dx_m": 0.5}},
                {"op": "goto", "params": {"x": "c.position[0]", "y": 0, "z": 0}},
            ])


class TestWholeBindingPassedByName:
    """Passing the whole binding (``target="crate"``) must be freshness-checked too.

    ``referenced_binding_names`` deliberately ignores a bare name so an
    ``object_name="box"`` is never mistaken for a binding — which used to let a
    whole-binding reference slip past BOTH checks. "Drive somewhere, then act on the
    coordinate measured before the move" is the single most dangerous plan shape, and
    it compiled cleanly.
    """

    def test_stale_whole_binding_is_rejected(self):
        with pytest.raises(SequenceError, match="stale"):
            _parse([
                {"op": "detect", "params": {"object_name": "crate"}, "bind": "crate"},
                {"op": "drive", "params": {"dx_m": 0.5}},
                {"op": "press", "params": {"target": "crate"}},
            ])

    def test_fresh_whole_binding_is_accepted(self):
        _parse([
            {"op": "detect", "params": {"object_name": "crate"}, "bind": "crate"},
            {"op": "press", "params": {"target": "crate"}},
        ])

    def test_a_literal_that_merely_looks_like_a_bind_name_is_not_judged(self):
        """``object_name`` is a free string; it must not be read as a binding reference."""
        _parse([
            {"op": "detect", "params": {"object_name": "crate"}, "bind": "crate"},
            {"op": "press", "params": {}},
        ])


# --------------------------------------------------------------------------- #
# Check 5 — a param the action does not declare
# --------------------------------------------------------------------------- #
class TestDeclaredParamsOnly:
    """An invented param used to survive every check and fail at dispatch.

    That is the worst place for it: a ``TypeError`` mid-run aborts the sequence with the
    body half-way through a task, and the planner never learns what it should have written.
    Caught at compile time it is a message the correction loop can act on — the same
    argument as for checking op names rather than trusting them.
    """

    def test_an_undeclared_param_is_rejected(self):
        with pytest.raises(SequenceError, match=r"unknown param\(s\) \['beside'\]"):
            _parse([{"op": "detect", "params": {"object_name": "box", "beside": "hat"}}])

    def test_the_error_names_what_the_action_does_take(self):
        with pytest.raises(SequenceError, match=r"it takes \['object_name'\]"):
            _parse([{"op": "detect", "params": {"object_name": "box", "on": "table"}}])

    def test_an_action_declaring_nothing_accepts_nothing(self):
        with pytest.raises(SequenceError, match="takes no params"):
            _parse([{"op": "release", "params": {"width_mm": 80}}])

    def test_declared_params_pass(self):
        _parse([{"op": "drive", "params": {"dx_m": 0.5}}, {"op": "goto", "params": {"x": 1, "y": 2, "z": 3}}])

    def test_a_loop_detect_step_is_checked_too(self):
        with pytest.raises(SequenceError, match=r"unknown param\(s\) \['near'\]"):
            _parse([{
                "loop": {
                    "detect": {"op": "detect", "params": {"object_name": "box", "near": "hat"}},
                    "bind": "t",
                    "body": [{"op": "grip"}],
                }
            }])

    def test_no_contract_means_no_check(self):
        """A bare name set carries no schema, so this check must degrade to a no-op."""
        parse_sequence([{"op": "detect", "params": {"whatever": 1}}], allowed_ops={"detect"})


class TestQualifierEnforced:
    """The task's own spatial qualifier must survive into the call that uses it."""

    @staticmethod
    def _index():
        class Fake:
            @robot_tool(name="locate_for_grasp", desc="d", capability="vision.detection",
                        produces_location=True)
            def locate_for_grasp(self, object_name: str = "box", reference: str = "",
                                 relation: str = "on") -> dict:
                return {}

            @robot_tool(name="analyze_scene", desc="d", capability="vision.detection")
            def analyze_scene(self, object_name: str = "box") -> dict:
                return {}

        f = Fake()
        return {"locate_for_grasp": f.locate_for_grasp, "analyze_scene": f.analyze_scene}

    _G = {"apple": {"reference": "drawer", "relation": "in"}}

    def _step(self, params):
        return [{"op": "locate_for_grasp", "params": params, "bind": "a"}]

    def test_rejects_the_bare_call(self):
        # The failure this prevents is silent: a bare call grasps whichever apple ranks
        # first, reports ok, and nothing downstream can tell it was the wrong one.
        with pytest.raises(SequenceError, match="reference='drawer'"):
            parse_sequence(self._step({"object_name": "apple"}),
                           allowed_ops=self._index(), grounding=self._G)

    def test_rejects_a_contradicting_qualifier(self):
        with pytest.raises(SequenceError, match="relation='in'"):
            parse_sequence(self._step({"object_name": "apple", "reference": "table", "relation": "on"}),
                           allowed_ops=self._index(), grounding=self._G)

    def test_accepts_the_stated_qualifier(self):
        parse_sequence(self._step({"object_name": "apple", "reference": "drawer", "relation": "in"}),
                       allowed_ops=self._index(), grounding=self._G)

    def test_ignores_objects_the_task_did_not_qualify(self):
        parse_sequence(self._step({"object_name": "plate"}), allowed_ops=self._index(), grounding=self._G)

    def test_ignores_actions_that_cannot_carry_a_qualifier(self):
        # analyze_scene takes no reference/relation — demanding one would reject a valid plan.
        parse_sequence([{"op": "analyze_scene", "params": {"object_name": "apple"}}],
                       allowed_ops=self._index(), grounding=self._G)

    def test_no_grounding_changes_nothing(self):
        parse_sequence(self._step({"object_name": "apple"}), allowed_ops=self._index())
