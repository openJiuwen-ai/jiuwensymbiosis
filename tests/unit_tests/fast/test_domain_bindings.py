"""The existing sequence engine consumes device-bound JSON outside robotics."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig
from jiuwensymbiosis.agent.builder import MANAGED_SESSION_PROMPT, build_session_agent
from jiuwensymbiosis.agent.fast.runner import run_bound_sequence
from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence
from jiuwensymbiosis.agent.run import run_session_task
from jiuwensymbiosis.tools.bindings import ActionBinding, build_action_tools, select_bindings


@dataclass(frozen=True)
class Contract:
    name: str
    input_schema: dict
    output_schema: dict
    description: str = "Synthetic instrument declaration"
    required_capabilities: frozenset[str] = frozenset({"instrument.observe"})
    planner_visible: bool = True


def catalog_session(*, fail: bool = False):
    calls = []
    permission = {"allowed": True}
    input_schema = {
        "type": "object",
        "properties": {"temperature_c": {"type": "number", "minimum": 0, "maximum": 50}},
        "required": ["temperature_c"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"temperature_c": {"type": "number"}, "status": {"type": "string"}},
        "required": ["temperature_c", "status"],
        "additionalProperties": False,
    }

    def read(params, key, principal):
        calls.append(("read", params, key, principal))
        if fail:
            raise RuntimeError("measurement outcome unknown")
        return {"temperature_c": 20, "status": "idle"}

    def write(params, key, principal):
        calls.append(("write", params, key, principal))
        return {"temperature_c": params["temperature_c"], "status": "idle"}

    read_binding = ActionBinding(
        Contract("device_a_read", {"type": "object", "additionalProperties": False}, output_schema),
        "physical-a",
        ("instrument:physical-a",),
        "instrument",
        read,
        lambda principal: principal == "local" and permission["allowed"],
    )
    write_binding = ActionBinding(
        Contract("device_b_write", input_schema, output_schema),
        "physical-b",
        ("instrument:physical-b",),
        "instrument",
        write,
        lambda principal: principal == "local" and permission["allowed"],
    )
    session = SimpleNamespace(
        action_bindings=lambda: [read_binding, write_binding],
        capabilities=frozenset({"instrument.observe"}),
        execution_request_prefix="durable-job",
        cancel_token=None,
        name="instruments",
    )
    return session, calls, permission


@pytest.mark.parametrize("reference", ["reading.temperature_c", {"$ref": "reading.temperature_c"}])
def test_numeric_binding_uses_same_runner_without_robot_detection(reference):
    session, calls, _ = catalog_session()
    result = run_bound_sequence(
        session,
        [
            {"op": "device_a_read", "bind": "reading", "request_key": "measurement-1"},
            {"op": "device_b_write", "params": {"temperature_c": reference}},
        ],
        request_key_prefix="submission-1",
    )
    assert result["ok"] is True
    assert result["steps"][0]["result"] == {"temperature_c": 20, "status": "idle"}
    assert calls == [
        ("read", {}, "measurement-1", "local"),
        ("write", {"temperature_c": 20}, "submission-1:1", "local"),
    ]
    assert [step["device_id"] for step in result["steps"]] == ["physical-a", "physical-b"]


def test_instrument_failure_stops_and_never_requires_robot_objects():
    session, calls, _ = catalog_session(fail=True)
    result = run_bound_sequence(
        session,
        [
            {"op": "device_a_read", "bind": "reading"},
            {"op": "device_b_write", "params": {"temperature_c": "reading.temperature_c"}},
        ],
        request_key_prefix="submission-2",
    )
    assert result["ok"] is False
    assert result["steps_done"] == 1
    assert len(calls) == 1
    assert "unknown" in result["steps"][0]["reason"]


@pytest.mark.parametrize(
    "steps",
    [
        [{"op": "device_b_write", "params": {"temperature_c": 100}}],
        [{"op": "device_b_write", "params": {"invented": 20}}],
        [{"op": "device_b_write", "params": {"temperature_c": {"$ref": "missing.temperature_c"}}}],
        [
            {"op": "device_a_read", "bind": "reading"},
            {"op": "device_b_write", "params": {"temperature_c": "reading.x"}},
        ],
    ],
)
def test_schema_and_references_rejected_before_io(steps):
    session, calls, _ = catalog_session()
    with pytest.raises(SequenceError):
        run_bound_sequence(session, steps)
    assert calls == []


async def test_actual_local_function_rechecks_permission_and_has_host_request_key():
    session, calls, permission = catalog_session()
    tools = build_action_tools(
        session.action_bindings(), session.capabilities, request_key_factory=lambda: "job:tool:0"
    )
    await tools[0].invoke({})
    assert calls == [("read", {}, "job:tool:0", "local")]
    permission["allowed"] = False
    with pytest.raises(Exception, match="authorized"):
        await tools[0].invoke({})
    assert len(calls) == 1
    assert select_bindings(session.action_bindings(), session.capabilities) == {}


def test_pure_instrument_stepagent_and_fastagent_share_catalog(monkeypatch, tmp_path):
    session, calls, _ = catalog_session()
    built = {}
    monkeypatch.setattr("jiuwensymbiosis.agent.builder.create_deep_agent", lambda **kwargs: built.update(kwargs))
    build_session_agent(
        session,
        RobotAgentConfig(
            execution={"mode": "stepagent"},
            model=object(),
            workspace=str(tmp_path),
            logging={"dir": None},
        ),
    )
    assert built["system_prompt"] == MANAGED_SESSION_PROMPT
    assert {tool.card.name for tool in built["tools"]} == {"device_a_read", "device_b_write"}
    assert built["rails"] == []
    assert built["parallel_tool_calls"] is False
    planned = {}

    def compose(query, **kwargs):
        planned.update(kwargs)
        return [
            {"op": "device_a_read", "bind": "reading"},
            {"op": "device_b_write", "params": {"temperature_c": "reading.temperature_c"}},
        ]

    monkeypatch.setattr("jiuwensymbiosis.agent.fast.planner.compose_actions", compose)
    result = run_session_task(session, "read then set", RobotAgentConfig(model_spec=ModelSpec(), logging={"dir": None}))
    assert result["ok"] is True
    assert set(planned["action_index"]) == {tool.card.name for tool in built["tools"]}
    assert MANAGED_SESSION_PROMPT in planned["system_prompt"]
    assert len(calls) == 2


def test_device_scope_is_not_flattened_when_comparing_catalogs():
    session, _, _ = catalog_session()
    bindings = select_bindings(session.action_bindings(), session.capabilities)
    assert bindings["device_a_read"].resource_keys != bindings["device_b_write"].resource_keys
    assert select_bindings(session.action_bindings(), (), planner_only=True) == {}
    with pytest.raises(SequenceError, match="unknown op"):
        parse_sequence([{"op": "device_a_read"}], allowed_ops={})


def test_existing_robot_fastagent_uses_same_gated_contract_views(monkeypatch):
    from jiuwensymbiosis.agent.run import run_fast_task
    from jiuwensymbiosis.env.mock import MockArmEnv
    from jiuwensymbiosis.tools.builder import build_robot_tools
    from tests.helpers import make_mock_session

    class MotionOnlyEnv(MockArmEnv):
        capabilities = frozenset({"motion.cartesian"})

    session = make_mock_session(env=MotionOnlyEnv())
    planned = {}

    def capture_plan(query, **kwargs):
        planned.update(kwargs)
        raise RuntimeError("stop after observing the admitted catalog")

    monkeypatch.setattr("jiuwensymbiosis.agent.fast.plan_task", capture_plan)
    monkeypatch.setattr("jiuwensymbiosis.agent.fast.planner.parse_task", lambda *args, **kwargs: {"targets": []})
    monkeypatch.setattr("jiuwensymbiosis.agent.run._perceive_scene", lambda *args: None)
    result = run_fast_task(session, "home", RobotAgentConfig(model_spec=ModelSpec(), logging={"dir": None}))
    assert result["ok"] is False
    tools = build_robot_tools(session.api, env=session.env, planner_only=True)
    assert set(planned["action_index"]) == {tool.card.name for tool in tools}
    assert "close_gripper" not in planned["action_index"]
    binding = planned["action_index"]["home"]
    assert isinstance(binding, ActionBinding)
    assert binding.contract.meta is session.api.home.__tool_meta__


def test_model_cannot_select_or_replay_persistent_invocation_keys(monkeypatch):
    session, calls, _ = catalog_session()
    monkeypatch.setattr(
        "jiuwensymbiosis.agent.fast.planner.compose_actions",
        lambda *args, **kwargs: [{"op": "device_a_read", "request_key": "an-older-measurement"}],
    )
    with pytest.raises(ValueError, match="cannot choose persistent"):
        run_session_task(session, "read a fresh value", RobotAgentConfig(model_spec=ModelSpec(), logging={"dir": None}))
    assert calls == []
    # An explicit host-submitted action still controls its durable retry key.
    result = run_bound_sequence(session, [{"op": "device_a_read", "request_key": "host-selected"}])
    assert result["ok"] is True
    assert calls[0][2] == "host-selected"
