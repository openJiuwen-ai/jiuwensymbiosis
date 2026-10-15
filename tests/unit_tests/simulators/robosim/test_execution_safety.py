"""Unknown motion results must survive both tool and direct executor boundaries."""

from unittest.mock import Mock

from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence
from jiuwensymbiosis.simulators.robosim import RobosimConfig, build_robosim_session
from jiuwensymbiosis.simulators.robosim.lowlevel import RobosimLowLevel
from jiuwensymbiosis.simulators.robosim.operations import RobosimError
from jiuwensymbiosis.tools.robot_control_tool import RobotControlTool, _build_action_index
from tests.unit_tests.fast.test_ability_exec import _run_for
from tests.unit_tests.simulators.robosim.test_adapter import observation


def session_with_unknown_motion():
    session = build_robosim_session(RobosimConfig())
    driver = Mock(spec=RobosimLowLevel)
    driver.observe.side_effect = observation
    driver.set_gripper.side_effect = RobosimError("Connection lost", outcome_unknown=True)
    session.env.low_level = driver
    return session, driver


def test_direct_runner_stops_without_retry_or_recovery():
    session, driver = session_with_unknown_motion()
    session.api.home = Mock()
    session.api.recovery_home = Mock()
    index = _build_action_index(session.api, env=session.env, planner_only=True)
    steps = parse_sequence([{"op": "open_gripper"}, {"op": "open_gripper"}], allowed_ops=index)
    result = run_sequence(session, steps, action_index=index)
    assert result["ok"] is False
    assert result["outcome_unknown"] is True
    assert result["steps_done"] == 1
    driver.set_gripper.assert_called_once_with(False)
    session.api.home.assert_not_called()
    session.api.recovery_home.assert_not_called()


async def test_control_tool_preserves_unknown_result():
    session, driver = session_with_unknown_motion()
    output = await RobotControlTool(session.api, env=session.env).invoke({"action": "open_gripper"})
    assert not output.success
    assert output.data["outcome_unknown"] is True
    assert output.data["skip_recovery"] is True
    driver.set_gripper.assert_called_once_with(False)


def test_ability_executor_preserves_unknown_result():
    from types import SimpleNamespace

    output = SimpleNamespace(
        success=False,
        data={"error_code": "robosim_transport_error", "outcome_unknown": True, "skip_recovery": True},
        error="Connection lost",
    )
    result = _run_for(output)("open_gripper", {})
    assert not result["ok"]
    assert result["outcome_unknown"] is True
    assert result["skip_recovery"] is True
