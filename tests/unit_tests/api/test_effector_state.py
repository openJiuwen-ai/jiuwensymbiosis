"""Named grippers share actions without transferring payload facts between hands."""

from types import SimpleNamespace as NS

import pytest

from jiuwensymbiosis.agent.fast.runner import ActionStep, _self_state_drift
from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence
from jiuwensymbiosis.api.actions import ACTIONS, CLOSE_GRIPPER, OPEN_GRIPPER, implements
from jiuwensymbiosis.api.memory import ExecutionMemory
from jiuwensymbiosis.api.world_state import current_tokens
from jiuwensymbiosis.tools.robot_control_tool import action_scope, record_action


class Hands:
    @implements(OPEN_GRIPPER)
    def open_gripper(self, width_mm=80.0, *, effector=None):
        return {"ok": True}

    @implements(CLOSE_GRIPPER)
    def close_gripper(self, force_n=None, *, effector=None):
        return {"ok": True, "holding_payload": None, "grasp_confirmed": False}


@pytest.mark.parametrize("left", [False, None, True])
def test_open_left_preserves_right_and_unknown_observation_overrides_belief(left):
    api = Hands()
    api.memory = ExecutionMemory(self_state=frozenset({"payload.held:right", "payload.held:left", "payload.held"}))
    with pytest.raises(RuntimeError), action_scope(api, api.open_gripper, {"effector": "left"}):
        raise RuntimeError("release interrupted")
    assert "payload.held:left" not in api.memory.self_state
    record_action(api, api.open_gripper, {"effector": "left"}, {"ok": True})
    assert "payload.held:right" in api.memory.self_state
    assert "payload.clear" not in api.memory.self_state
    session = NS(api=api, env=NS(effector_payloads={"left": left, "right": None}, holding_payload=None))
    tokens = current_tokens(session)
    assert "payload.held:right" not in tokens
    assert ("payload.held" in tokens) == (left is True)
    assert "payload.clear" not in tokens


def test_close_left_with_right_held_plans_and_checks_only_left():
    api = Hands()
    session = NS(api=api, env=NS(effector_payloads={"left": False, "right": True}, holding_payload=True))
    index = {"close_gripper": api.close_gripper, "open_gripper": api.open_gripper}
    request = {"op": "close_gripper", "params": {"effector": "left"}}
    parse_sequence([request], allowed_ops=index, initial_state=current_tokens(session))
    step = ActionStep(**request)
    assert _self_state_drift(step, api.close_gripper.__tool_meta__, session) is None
    session.env.effector_payloads["left"] = True
    assert _self_state_drift(step, api.close_gripper.__tool_meta__, session)
    with pytest.raises(SequenceError):
        parse_sequence([request], allowed_ops=index, initial_state=current_tokens(session))


def test_legacy_single_gripper_keeps_call_shape_and_rejects_unknown_selector():
    calls = []

    class Legacy:
        @implements(OPEN_GRIPPER)
        def open_gripper(self, width_mm=80.0):
            calls.append(width_mm)

    api = Legacy()
    api.open_gripper(25.0)
    with pytest.raises(ValueError):
        api.open_gripper(effector="unknown")
    assert calls == [25.0]
    assert "set_gripper" not in ACTIONS


@pytest.mark.parametrize("held", [None, False, True])
def test_selected_close_uses_measured_result_not_legacy_trust(held):
    api = Hands()
    api.trust_command_grasp = True
    api.memory = ExecutionMemory(self_state=frozenset({"payload.held:right"}))
    record_action(
        api,
        api.close_gripper,
        {"effector": "left"},
        {"ok": True, "holding_payload": held, "grasp_confirmed": held is True},
    )
    assert "payload.held:right" in api.memory.self_state
    assert ("payload.held:left" in api.memory.self_state) == (held is True)
    assert ("payload.clear:left" in api.memory.self_state) == (held is False)
