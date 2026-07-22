# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the C1 fast-path action-sequence schema + expression evaluator.

These exercise the **task-agnostic** machinery only: op validation against a
dynamic action vocabulary, the whitelisted-AST expression evaluator, and the
pass-through detection binding. No test asserts any pick/place (or any other
task) semantics — the framework must carry none.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.agent.fast.sequence import (
    ActionStep,
    ExprError,
    SequenceError,
    evaluate_expr,
    normalize_detection,
    parse_sequence,
    resolve_params,
    resolve_value,
)

# A stand-in action vocabulary; in production this comes from the live api's
# @robot_tool index, so any robot/skill's actions validate the same way.
_ALLOWED = frozenset({"home", "goto_xyzr", "open_gripper", "close_gripper", "get_grasp_info_simple"})


# --------------------------------------------------------------------------- #
# parse_sequence
# --------------------------------------------------------------------------- #
def test_parse_valid_sequence():
    raw = [
        {"op": "home"},
        {"op": "track_detect", "params": {"object_name": "目标"}, "bind": "t"},
        {"op": "goto_xyzr", "params": {"x": "t.x", "y": "t.y", "z": "t.z + lift"}},
    ]
    steps = parse_sequence(raw, allowed_ops=_ALLOWED, special_ops={"track_detect"})
    assert [s.op for s in steps] == ["home", "track_detect", "goto_xyzr"]
    assert steps[1].is_detection() and steps[1].bind == "t"
    assert not steps[0].is_detection()


def test_parse_arbitrary_ops_from_vocabulary():
    # Any op present in the (dynamic) vocabulary validates — nothing is special-cased.
    vocab = frozenset({"wave", "spin", "honk"})
    steps = parse_sequence([{"op": "wave"}, {"op": "honk"}], allowed_ops=vocab, special_ops=frozenset())
    assert [s.op for s in steps] == ["wave", "honk"]


def test_parse_rejects_non_list():
    with pytest.raises(SequenceError):
        parse_sequence({"op": "home"}, allowed_ops=_ALLOWED, special_ops=frozenset())


def test_parse_rejects_unknown_op():
    with pytest.raises(SequenceError, match="unknown op"):
        parse_sequence([{"op": "fly_to_moon"}], allowed_ops=_ALLOWED, special_ops=frozenset())


def test_parse_allows_track_detect_even_if_not_in_api():
    # track_detect is the runner-implemented compound op; it must be explicitly
    # authorized via special_ops (no implicit default).
    steps = parse_sequence(
        [{"op": "track_detect", "params": {"object_name": "x"}, "bind": "p"}],
        allowed_ops=frozenset(),
        special_ops={"track_detect"},
    )
    assert steps[0].op == "track_detect"


def test_parse_track_grasp_requires_explicit_special_op_and_validates_approach():
    raw = [{"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 40}, "bind": "banana"}]
    # A known special op must be explicitly authorized via special_ops; it
    # cannot sneak in through allowed_ops (which would bypass its validator).
    with pytest.raises(SequenceError, match="not authorized in special_ops"):
        parse_sequence(raw, allowed_ops={"track_grasp"}, special_ops=frozenset())
    with pytest.raises(SequenceError, match="not authorized in special_ops"):
        parse_sequence(raw, allowed_ops=frozenset(), special_ops=frozenset())
    steps = parse_sequence(raw, allowed_ops=frozenset(), special_ops={"track_grasp"})
    assert steps[0].op == "track_grasp"
    with pytest.raises(SequenceError, match=r"\[30, 100\]"):
        parse_sequence(
            [{"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 10}, "bind": "banana"}],
            allowed_ops=frozenset(),
            special_ops={"track_grasp"},
        )
    with pytest.raises(SequenceError, match=r"\[30, 100\]"):
        parse_sequence(
            [{"op": "track_grasp", "params": {"object_name": "banana", "approach_mm": 101}, "bind": "banana"}],
            allowed_ops=frozenset(),
            special_ops={"track_grasp"},
        )


def test_parse_special_ops_default_does_not_implicitly_authorize():
    # special_ops defaults to an empty set, so omitting it stays compatible
    # with legacy callers — but a known special op is still NOT implicitly
    # authorized: it is rejected with a SequenceError (not a TypeError, and
    # never silently let through).
    with pytest.raises(SequenceError, match="not authorized in special_ops"):
        parse_sequence(
            [{"op": "track_detect", "params": {"object_name": "x"}, "bind": "p"}],
            allowed_ops=frozenset(),
        )


def test_parse_rejects_track_detect_without_object():
    with pytest.raises(SequenceError, match="object_name"):
        parse_sequence(
            [{"op": "track_detect", "params": {}, "bind": "p"}],
            allowed_ops=_ALLOWED,
            special_ops={"track_detect"},
        )


def test_parse_rejects_bad_bind_identifier():
    with pytest.raises(SequenceError, match="bind"):
        parse_sequence(
            [{"op": "track_detect", "params": {"object_name": "x"}, "bind": "1bad"}],
            allowed_ops=_ALLOWED,
            special_ops={"track_detect"},
        )


def test_parse_rejects_missing_op():
    with pytest.raises(SequenceError, match="op"):
        parse_sequence([{"params": {}}], allowed_ops=_ALLOWED, special_ops=frozenset())


def test_parse_rejects_bad_params_type():
    with pytest.raises(SequenceError, match="params"):
        parse_sequence([{"op": "home", "params": [1, 2]}], allowed_ops=_ALLOWED, special_ops=frozenset())


def test_parse_rejects_track_detect_without_bind():
    # track_detect's whole purpose is to bind a detection for later reference.
    with pytest.raises(SequenceError, match="bind"):
        parse_sequence(
            [{"op": "track_detect", "params": {"object_name": "black box"}}],
            allowed_ops=_ALLOWED,
            special_ops={"track_detect"},
        )


def test_parse_rejects_reference_to_unbound_binding():
    # The exact production failure: object_name 'black box' but a later step
    # reads 'black_box.*', which no step bound → caught at compile time instead
    # of a cryptic 'str < float' crash deep in goto_xyzr at run time.
    raw = [
        {"op": "track_detect", "params": {"object_name": "black box"}, "bind": "pick"},
        {"op": "goto_xyzr", "params": {"x": "black_box.position[0]", "z": "black_box.grasp_z + 40"}},
    ]
    with pytest.raises(SequenceError, match="unbound"):
        parse_sequence(raw, allowed_ops=_ALLOWED, special_ops={"track_detect"})


def test_parse_accepts_reference_matching_prior_bind():
    raw = [
        {"op": "track_detect", "params": {"object_name": "black box"}, "bind": "black_box"},
        {"op": "goto_xyzr", "params": {"x": "black_box.position[0]", "z": "black_box.grasp_z + 40"}},
    ]
    steps = parse_sequence(raw, allowed_ops=_ALLOWED, special_ops={"track_detect"})
    assert steps[1].op == "goto_xyzr"


def test_parse_rejects_forward_reference_before_bind():
    # Referencing a binding produced by a *later* step is still unbound at use.
    raw = [
        {"op": "goto_xyzr", "params": {"x": "pick.x"}},
        {"op": "track_detect", "params": {"object_name": "x"}, "bind": "pick"},
    ]
    with pytest.raises(SequenceError, match="unbound"):
        parse_sequence(raw, allowed_ops=_ALLOWED, special_ops={"track_detect"})


def test_referenced_binding_names_ignores_literals_and_bare_names():
    from jiuwensymbiosis.agent.fast.sequence import referenced_binding_names

    assert referenced_binding_names("black box") == set()  # syntax error → literal
    assert referenced_binding_names("黑盒子") == set()  # bare name → not a field read
    assert referenced_binding_names(40) == set()  # non-string
    assert referenced_binding_names("box.grasp_z + 30") == {"box"}
    assert referenced_binding_names("black_box.position[0]") == {"black_box"}


# --------------------------------------------------------------------------- #
# evaluate_expr
# --------------------------------------------------------------------------- #
def test_eval_arithmetic_and_constants():
    assert evaluate_expr("40", {}) == 40.0
    assert evaluate_expr("2 + 3 * 4", {}) == 14.0
    assert evaluate_expr("-5", {}) == -5.0
    assert evaluate_expr("10 / 4", {}) == 2.5


def test_eval_name_lookup():
    assert evaluate_expr("approach", {"approach": 40.0}) == 40.0
    assert evaluate_expr("a + b", {"a": 100.0, "b": 40.0}) == 140.0


def test_eval_attribute_and_subscript():
    env = {"t": {"x": 250.0, "grasp_z": 150.0, "position": [250.0, 90.0, 70.0]}}
    assert evaluate_expr("t.x", env) == 250.0
    assert evaluate_expr("t.grasp_z + 80", env) == 230.0
    assert evaluate_expr("t.position[1]", env) == 90.0


def test_eval_rejects_unknown_name():
    with pytest.raises(ExprError, match="unknown name"):
        evaluate_expr("missing + 1", {})


def test_eval_rejects_missing_field():
    with pytest.raises(ExprError):
        evaluate_expr("t.nope", {"t": {"x": 1.0}})


def test_eval_rejects_function_call():
    with pytest.raises(ExprError):
        evaluate_expr("abs(-3)", {"abs": abs})


def test_eval_rejects_non_numeric_result():
    with pytest.raises(ExprError, match="number"):
        evaluate_expr("name", {"name": "字符串"})


def test_eval_rejects_string_literal():
    with pytest.raises(ExprError):
        evaluate_expr("'hi'", {})


def test_eval_rejects_attribute_chain_injection():
    # No dunder / attribute escape to Python internals.
    with pytest.raises(ExprError):
        evaluate_expr("t.__class__", {"t": {"x": 1.0}})


# --------------------------------------------------------------------------- #
# resolve_value / resolve_params
# --------------------------------------------------------------------------- #
def test_resolve_value_keeps_literal_string():
    assert resolve_value("目标物体", {}) == "目标物体"
    assert resolve_value("white box", {}) == "white box"  # space → not an expr


def test_resolve_value_keeps_unresolved_identifier_as_literal():
    # A bare identifier not in env stays a literal string (not a hard error),
    # so object names never accidentally fail.
    assert resolve_value("white_box", {}) == "white_box"


def test_resolve_value_evaluates_expression():
    env = {"t": {"x": 250.0, "grasp_z": 150.0}, "approach": 40.0}
    assert resolve_value("t.grasp_z + approach", env) == 190.0


def test_resolve_value_passes_numbers():
    assert resolve_value(40, {}) == 40
    assert resolve_value(2.5, {}) == 2.5


def test_resolve_params_mixed():
    env = {"t": {"x": 250.0, "y": 90.0, "grasp_z": 150.0}, "approach": 40.0, "lift": 80.0}
    out = resolve_params({"x": "t.x", "y": "t.y", "z": "t.grasp_z + lift", "name": "目标"}, env)
    assert out == {"x": 250.0, "y": 90.0, "z": 230.0, "name": "目标"}


# --------------------------------------------------------------------------- #
# normalize_detection — task-agnostic pass-through + x/y/z convenience
# --------------------------------------------------------------------------- #
def test_normalize_passes_raw_fields_through():
    gi = {"position": [250.0, 90.0, 70.0], "grasp_z": 50.0, "place_z": 80.0, "score": 0.9, "depth_m": 0.3}
    b = normalize_detection(gi)
    # every raw field addressable, plus x/y/z conveniences
    assert b["grasp_z"] == 50.0 and b["place_z"] == 80.0 and b["score"] == 0.9 and b["depth_m"] == 0.3
    assert b["x"] == 250.0 and b["y"] == 90.0 and b["z"] == 70.0  # z == position[2], no task bias


def test_normalize_keeps_arbitrary_detector_fields():
    # A detector that emits non-grasp fields (e.g. orientation, radius) — all kept.
    gi = {"position": [1.0, 2.0, 3.0], "yaw_deg": 45.0, "radius_mm": 12.0}
    b = normalize_detection(gi)
    assert b["yaw_deg"] == 45.0 and b["radius_mm"] == 12.0
    assert (b["x"], b["y"], b["z"]) == (1.0, 2.0, 3.0)


def test_normalize_missing_position_defaults_zero():
    b = normalize_detection({"score": 0.5})
    assert b["x"] == 0.0 and b["y"] == 0.0 and b["z"] == 0.0 and b["score"] == 0.5


def test_normalize_binding_drives_expression():
    # End-to-end: a normalized binding feeds an expression like the runner will.
    gi = {"position": [250.0, 90.0, 70.0], "grasp_z": 50.0}
    env = {"t": normalize_detection(gi), "approach": 40.0}
    assert resolve_value("t.grasp_z + approach", env) == 90.0
    assert resolve_value("t.x", env) == 250.0


def test_step_dataclass_defaults():
    s = ActionStep(op="home")
    assert s.params == {} and s.bind is None and not s.is_detection()
