"""Admission and every Cruzr command transport use the captured ROS domain."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel, CruzrNav
from jiuwensymbiosis.runtime import prepare_binding


@pytest.fixture
def ros(monkeypatch):
    node, executor = Mock(), Mock()
    ros = SimpleNamespace(
        ok=Mock(return_value=False),
        init=Mock(),
        create_node=Mock(return_value=node),
        shutdown=Mock(),
        get_default_context=Mock(return_value=SimpleNamespace(get_domain_id=lambda: 42)),
    )
    modules = {
        "rclpy": ros,
        "rclpy.executors": SimpleNamespace(SingleThreadedExecutor=lambda: executor),
        "rclpy.qos": SimpleNamespace(
            QoSProfile=lambda **kw: SimpleNamespace(),
            ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1),
            DurabilityPolicy=SimpleNamespace(VOLATILE=1),
            HistoryPolicy=SimpleNamespace(KEEP_LAST=1),
        ),
        "mc_state_msgs.msg": SimpleNamespace(RobotState=object),
        "mc_task_msgs.msg": SimpleNamespace(JointCmd=object, RobotCommand=object),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(CruzrLowLevel, "_spin", lambda _: None)
    monkeypatch.setattr(CruzrLowLevel, "_await_first_state", lambda *_: None)
    return ros


def test_binding_domain_is_used_by_driver_after_environment_changes(tmp_path, monkeypatch, ros):
    source = tmp_path / "robot.yaml"
    source.write_text("adapter: cruzr\ncommand_topic: /robot/command\n", encoding="utf-8")
    monkeypatch.setenv("ROS_DOMAIN_ID", "027")
    binding = prepare_binding(source, include_sidecars=False)
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    assert binding.resources == ("ros:27:/robot/command",)
    driver = CruzrLowLevel(binding.build_session().env.cfg)
    try:
        ros.init.assert_called_once_with(args=None, domain_id=27)
    finally:
        driver.close()


def test_existing_context_with_different_domain_is_rejected_before_publishing(ros):
    ros.ok.return_value = True
    with pytest.raises(ValueError, match="context domain"):
        CruzrLowLevel(CruzrConfig(ros_domain_id=27))
    ros.create_node.assert_not_called()
    ros.shutdown.assert_not_called()


@pytest.mark.parametrize("operation", ["move", "arc", "spin", "drive", "resident"])
def test_command_workers_use_configured_domain(monkeypatch, operation):
    from jiuwensymbiosis.adapters.cruzr import lowlevel

    cfg = CruzrConfig(ros_domain_id=27, resident_workers=operation == "resident")
    nav = CruzrNav(cfg)
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    spawn = Mock(return_value=Mock())
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr=""))
    monkeypatch.setattr(lowlevel.subprocess, "Popen", spawn)
    monkeypatch.setattr(lowlevel.subprocess, "run", run)
    if operation == "move":
        nav.navigate_relative(0.1)
    elif operation == "arc":
        nav.navigate_arc(1.0, 0.1)
    elif operation == "spin":
        nav.start_spin()
    elif operation == "drive":
        nav.start_drive()
    else:
        monkeypatch.setattr(lowlevel.ros2_worker, "read_line", lambda *a: '{"ok":true}')
        nav.navigate_relative(0.1)
    call = run.call_args if operation in {"move", "arc"} else spawn.call_args
    assert call.kwargs["env"]["ROS_DOMAIN_ID"] == "27"
