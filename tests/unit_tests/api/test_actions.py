# coding: utf-8
"""动作词表（`api/actions.py`）的不变量。

这些不是覆盖率测试，每一条守住一个"改坏了就静默出事"的性质：契约只有一份、
形状可被校验器读、规划器词表只含共享动作。它们是这层重构存在的理由。
"""

from __future__ import annotations

import inspect

import pytest

from jiuwensymbiosis.api.actions import (
    ACTIONS,
    ActionSpec,
    ContractViolation,
    UnknownCapability,
    implements,
    planner_vocabulary,
)
from jiuwensymbiosis.api.decorators import schema_from_typeddict
from jiuwensymbiosis.api.state import KNOWN_STATE_TOKENS
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES


class TestVocabularyWellFormed:
    def test_every_name_matches_its_key(self):
        for key, spec in ACTIONS.items():
            assert key == spec.name

    @pytest.mark.parametrize("spec", ACTIONS.values(), ids=lambda s: s.name)
    def test_capability_is_in_the_closed_set(self, spec):
        assert spec.capability is None or spec.capability in KNOWN_CAPABILITIES

    @pytest.mark.parametrize("spec", ACTIONS.values(), ids=lambda s: s.name)
    def test_state_tokens_are_in_the_closed_set(self, spec):
        for token in (*spec.requires, *spec.provides, *spec.invalidates):
            assert token in KNOWN_STATE_TOKENS

    @pytest.mark.parametrize("spec", ACTIONS.values(), ids=lambda s: s.name)
    def test_required_params_are_declared_params(self, spec):
        assert set(spec.required_params) <= set(spec.params)

    @pytest.mark.parametrize(
        "spec", [s for s in ACTIONS.values() if s.produces_location], ids=lambda s: s.name
    )
    def test_location_producers_declare_a_result_shape(self, spec):
        """A step that senses something must say what fields it returns.

        Without it a plan cannot write ``<bind>.field`` at all, and — worse —
        ``parse_sequence``'s field check degrades to a no-op, so an invented field
        reaches the robot instead of being rejected at compile time. This is the
        regression that made every cruzr sensing action unusable for data flow.

        Checks the DERIVED schema, not just ``result is not None``: a non-TypedDict
        (``result=dict``) satisfies the weaker form while still yielding no fields.
        """
        from jiuwensymbiosis.api.decorators import schema_from_typeddict

        assert schema_from_typeddict(spec.result).get("properties")


class TestDescriptionsAreOneLanguage:
    """Descriptions are prompt data, and the vocabulary is one document — keep it in one
    language. It started out mixed (24 English + 6 Chinese) purely because the texts were
    transcribed verbatim from wherever each action used to be declared; that is exactly the
    kind of drift this layer exists to stop.
    """

    @pytest.mark.parametrize("spec", ACTIONS.values(), ids=lambda s: s.name)
    def test_description_is_english(self, spec):
        cjk = [ch for ch in spec.description if "\u4e00" <= ch <= "\u9fff"]
        assert not cjk, f"{spec.name} description mixes Chinese into the English vocabulary: {cjk[:8]}"


class TestSingleSourceOfContract:
    """The invariant this whole layer exists for: one action, one contract."""

    def test_no_action_name_is_declared_twice(self):
        # ACTIONS is built by _register, which rejects duplicates outright; assert the
        # property directly so the guarantee is visible where a reader looks for it.
        assert len(ACTIONS) == len({spec.name for spec in ACTIONS.values()})

    def test_specs_are_immutable(self):
        spec = next(iter(ACTIONS.values()))
        with pytest.raises(AttributeError):  # frozen dataclass
            spec.name = "something else"  # type: ignore[misc]


class TestSpecValidation:
    def test_unknown_capability_is_rejected(self):
        with pytest.raises(UnknownCapability):
            ActionSpec(name="x", description="d", capability="myvendor.special")

    def test_unknown_state_token_is_rejected(self):
        with pytest.raises(ValueError):
            ActionSpec(name="x", description="d", provides=("payload.levitating",))

    def test_required_param_outside_params_is_rejected(self):
        with pytest.raises(ValueError):
            ActionSpec(name="x", description="d", params=("a",), required_params=("b",))

    def test_sensing_without_a_result_shape_is_rejected(self):
        with pytest.raises(ValueError, match="produces a location"):
            ActionSpec(name="x", description="d", produces_location=True)

    def test_a_shapeless_result_type_is_rejected_too(self):
        # ``dict`` is not None but yields no fields — the loophole that let
        # pixel_to_base_xyz claim a location it could not be read from.
        with pytest.raises(ValueError, match="no readable result shape"):
            ActionSpec(name="x", description="d", produces_location=True, result=dict)


class TestImplements:
    SPEC = ActionSpec(
        name="_demo_action",
        description="demo",
        capability="motion.base",
        params=("a", "b"),
        required_params=("a",),
        tags=("motion",),
        invalidates_locations=True,
    )

    def test_contract_comes_from_the_spec_not_the_method(self):
        class Body:
            @implements(self.SPEC)
            def _demo_action(self, a: float, b: int = 0) -> dict: ...

        meta = Body._demo_action.__robot_tool__
        assert meta.name == "_demo_action"
        assert meta.capability == "motion.base"
        assert meta.description == "demo"
        assert meta.invalidates_locations is True
        assert meta.tags == ["motion"]

    def test_a_signature_that_cannot_take_a_contract_param_fails_at_import(self):
        # The whole point of checking here: a mismatch that only shows up when the
        # planner emits that param would surface as a TypeError mid-motion.
        with pytest.raises(ContractViolation, match=r"does not accept \['b'\]"):

            class Body:
                @implements(self.SPEC)
                def _demo_action(self, a: float) -> dict: ...

    def test_kwargs_satisfies_any_contract_param(self):
        class Body:
            @implements(self.SPEC)
            def _demo_action(self, **kw) -> dict: ...

        assert Body._demo_action.__robot_tool__.name == "_demo_action"

    def test_body_extras_are_not_advertised(self):
        """A param only this body has stays callable but unadvertised.

        Otherwise a plan would come to depend on something the next body lacks.
        """

        class Body:
            @implements(self.SPEC)
            def _demo_action(self, a: float, b: int = 0, vendor_knob: float = 1.0) -> dict: ...

        props = Body._demo_action.__robot_tool__.input_params["properties"]
        assert set(props) == {"a", "b"}
        assert "vendor_knob" in inspect.signature(Body._demo_action).parameters

    def test_a_body_cannot_add_prose_of_its_own(self):
        """Every word a planner reads comes from the spec — an implementation has no channel
        for saying something about itself. A fact that changes a plan and holds on only one
        robot means the action is not the same action there, which is a contract problem;
        a fact that holds everywhere belongs in the spec and every body gets it free."""
        with pytest.raises(TypeError):
            implements(self.SPEC, body_note="wheels only")  # type: ignore[call-arg]

        class Body:
            @implements(self.SPEC)
            def _demo_action(self, a: float, b: int = 0) -> dict: ...

        meta = Body._demo_action.__robot_tool__
        assert meta.description == "demo"
        assert meta.full_description() == "demo"


class TestPlannerVocabulary:
    def test_gated_by_capability(self):
        vocab = planner_vocabulary({"motion.base"})
        assert "navigate_relative" in vocab
        assert "dual_arm_grasp" not in vocab

    def test_ungated_actions_are_always_present(self):
        assert "home" in planner_vocabulary(set())

    def test_debug_actions_are_excluded(self):
        vocab = planner_vocabulary(KNOWN_CAPABILITIES)
        for name in ("drive_arc", "get_image", "pixel_to_base_xyz", "move_named_joint"):
            assert name in ACTIONS
            assert name not in vocab, f"{name} is bring-up/diagnostic; it must not compete for planner attention"

    def test_a_bearing_is_not_a_location(self):
        """search_target reports a DIRECTION, so it must not claim to produce a location.

        Claiming one would let the validator accept ``search_target`` → a step that
        consumes a location: the plan compiles, then the consuming step reads an empty
        cache and fails on the robot.
        """
        from jiuwensymbiosis.api.actions import SEARCH_TARGET

        assert SEARCH_TARGET.produces_location is False
        assert SEARCH_TARGET.invalidates_locations is False   # reading a frame moves nothing
        assert "bearing_rad" in schema_from_typeddict(SEARCH_TARGET.result)["properties"]


class TestGraspPlaceLanesAreSymmetric:
    """The vocabulary splits on "what are you about to do", not "what is that thing".

    That axis only helps if it is applied consistently: a `_for_grasp` action with no
    `_for_place` twin (or the reverse) puts the planner back to guessing, because one
    lane would silently be the only option for a task that belongs in the other.
    """

    def test_every_lane_action_has_its_twin(self):
        lanes = {n for n in ACTIONS if n.endswith(("_for_grasp", "_for_place"))}
        assert lanes, "the lane suffix vanished — did a rename drop it?"
        for name in sorted(lanes):
            stem, _, side = name.rpartition("_for_")
            twin = f"{stem}_for_{'place' if side == 'grasp' else 'grasp'}"
            assert twin in ACTIONS, f"{name} has no {twin}: one lane would be the only option"

    def test_twins_agree_on_everything_but_the_side(self):
        for name in sorted(n for n in ACTIONS if n.endswith("_for_grasp")):
            grasp, place = ACTIONS[name], ACTIONS[name.replace("_for_grasp", "_for_place")]
            assert grasp.capability == place.capability
            assert grasp.params == place.params
            assert grasp.produces_location == place.produces_location
            assert grasp.invalidates_locations == place.invalidates_locations

    def test_each_description_points_at_its_twin(self):
        """A name alone is a hint. The description must NAME the other lane, so a planner
        reading one is told the alternative exists at the moment it is choosing — the
        failure this prevents is classifying the OBJECT instead of the TASK."""
        for name in sorted(n for n in ACTIONS if n.endswith(("_for_grasp", "_for_place"))):
            stem, _, side = name.rpartition("_for_")
            twin = f"{stem}_for_{'place' if side == 'grasp' else 'grasp'}"
            assert twin in ACTIONS[name].description, f"{name} never mentions {twin}"
