"""Cleanup failures must remain visible until each adapter resource is closed."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError


@pytest.mark.parametrize("next_action", ["move", "arc", "spin", "drive"])
def test_resident_stop_failure_blocks_every_nav_command(monkeypatch, next_action):
    from unittest.mock import Mock

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrNav

    nav = CruzrNav(CruzrConfig(resident_workers=True))
    request = Mock(side_effect=RuntimeError("resident stop unconfirmed"))
    monkeypatch.setattr(nav._move, "request_json", request)
    spawn = Mock(side_effect=AssertionError("must not launch another motion worker"))
    monkeypatch.setattr("subprocess.Popen", spawn)
    monkeypatch.setattr("subprocess.run", spawn)
    with pytest.raises(RuntimeError, match="resident stop unconfirmed"):
        nav.navigate_relative(0.1)

    actions = {
        "move": lambda: nav.navigate_relative(0.1),
        "arc": lambda: nav.navigate_arc(1, 0.1),
        "spin": nav.start_spin,
        "drive": nav.start_drive,
    }
    with pytest.raises(HardwareCleanupError, match="resident stop unconfirmed"):
        actions[next_action]()
    request.assert_called_once()
    spawn.assert_not_called()
    with pytest.raises(HardwareCleanupError, match="resident stop unconfirmed"):
        nav.close()


@pytest.mark.parametrize("operation", ["move", "arc", "spin", "drive"])
def test_cruzr_motion_failure_caught_by_caller_still_blocks_admission(tmp_path, monkeypatch, operation):
    import subprocess
    from unittest.mock import Mock

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrNav
    from jiuwensymbiosis.agent.session import RobotSession
    from jiuwensymbiosis.runtime.admission import admitted_session
    from jiuwensymbiosis.runtime.resources import ResourceBlockedError, ResourceManager

    nav = CruzrNav(CruzrConfig())
    if operation in {"move", "arc"}:
        monkeypatch.setattr(subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("wheel", 1)))

        def action():
            return nav.navigate_relative(0.1) if operation == "move" else nav.navigate_arc(1, 0.1)
    else:
        proc = Mock()
        proc.poll.return_value = -9
        proc.returncode = -9
        proc.communicate.return_value = ("", "")
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: proc)
        handle = nav.start_spin() if operation == "spin" else nav.start_drive()

        def action():
            return nav.stop_spin(handle) if operation == "spin" else nav.stop_drive(handle)

    env = SimpleNamespace(capabilities=frozenset(), connect=lambda: None, disconnect=nav.close)
    session = RobotSession(env=env, api=SimpleNamespace(capabilities=frozenset()), name="test-nav")
    session.motion_log_dir = str(tmp_path / "motion")
    manager = ResourceManager(tmp_path / "locks")
    binding = SimpleNamespace(resources=(f"ros:0:{tmp_path}/command",))
    try:
        with pytest.raises(ResourceBlockedError):
            with admitted_session(binding, session=session, resource_manager=manager):
                with pytest.raises(HardwareCleanupError):
                    action()
        session.disconnect()  # retry must not erase physical-stop uncertainty
        assert not session.cleanup_report().released
        with pytest.raises(ResourceBlockedError):
            manager.acquire(binding.resources, operation="next-task")
    finally:
        # No hardware was opened. Release test leases to isolate later tests.
        for lease in list(manager._leases.values()):
            manager.finish(lease, CleanupReport())


def test_cruzr_close_stops_uncollected_continuous_worker(monkeypatch):
    from unittest.mock import Mock

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrNav

    nav = CruzrNav(CruzrConfig())
    proc = Mock(returncode=0)
    proc.poll.side_effect = [None, 0]
    proc.communicate.return_value = ('{"ok":true}', "")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: proc)
    nav.start_drive()
    nav.close()
    nav.close()
    proc.communicate.assert_called_once_with(input="stop\n", timeout=15.0)


class _FailOnce:
    def __init__(self, method: str, *, error: str = "close failed") -> None:
        self.method = method
        self.error = error
        self.calls = 0

    def __getattr__(self, name: str):
        if name != self.method:
            raise AttributeError(name)

        def run(*_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(self.error)

        return run


class _CallRecorder:
    def __init__(self, method: str) -> None:
        self.method = method
        self.calls = 0

    def __getattr__(self, name: str):
        if name != self.method:
            raise AttributeError(name)

        def run(*_args, **_kwargs):
            self.calls += 1

        return run


def test_piper_close_retries_only_camera_after_can_close_succeeds():
    from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

    driver = PiperLowLevel.__new__(PiperLowLevel)
    camera = _FailOnce("stop", error="camera still streaming")
    arm = _CallRecorder("DisconnectPort")
    driver._camera = camera
    driver._arm = arm
    driver._can_may_be_open = True
    driver._closed = False

    with pytest.raises(HardwareCleanupError) as failed:
        driver.close()

    assert "camera still streaming" in str(failed.value)
    assert camera.calls == 1
    assert arm.calls == 1
    assert driver._camera is camera
    assert driver._can_may_be_open is False
    assert driver._closed is False

    driver.close()
    assert camera.calls == 2
    assert arm.calls == 1
    assert driver._camera is None
    assert driver._arm is None
    assert driver._closed is True


def test_piper_close_does_not_repeat_camera_after_can_disconnect_failure():
    from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

    driver = PiperLowLevel.__new__(PiperLowLevel)
    camera = _CallRecorder("stop")
    arm = _FailOnce("DisconnectPort", error="CAN disconnect failed")
    driver._camera = camera
    driver._arm = arm
    driver._can_may_be_open = True
    driver._closed = False

    with pytest.raises(HardwareCleanupError):
        driver.close()
    assert camera.calls == 1
    assert driver._camera is None
    assert driver._can_may_be_open is True

    driver.close()
    assert camera.calls == 1
    assert arm.calls == 2
    assert driver._closed is True


def test_piper_constructor_reports_unconfirmed_can_rollback(monkeypatch):
    from jiuwensymbiosis.adapters.piper.lowlevel import PiperLowLevel

    class _Arm:
        def ConnectPort(self):
            raise RuntimeError("CAN connect interrupted")

        def DisconnectPort(self):
            raise RuntimeError("CAN disconnect unconfirmed")

    monkeypatch.setenv("JIUWEN_PIPER_CMD_LOG", "0")
    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=lambda _port: _Arm()))

    with pytest.raises(HardwareCleanupError) as failed:
        PiperLowLevel(can_port="fake-can")

    assert isinstance(failed.value.original_error, RuntimeError)
    assert "CAN connect interrupted" in str(failed.value.original_error)
    assert len(failed.value.cleanup_errors) == 1
    assert "CAN disconnect unconfirmed" in str(failed.value.cleanup_errors[0])


def test_realsense_failed_start_retains_pipeline_for_retry(monkeypatch):
    from jiuwensymbiosis.perception.camera import RealSenseCamera

    class _Config:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *_args):
            pass

    class _Pipeline:
        def __init__(self):
            self.stop_calls = 0

        def start(self, _config):
            raise RuntimeError("profile activation failed")

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("pipeline stop failed")

    pipeline = _Pipeline()
    rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_Config,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = RealSenseCamera("fake-serial")

    with pytest.raises(HardwareCleanupError) as failed:
        camera.start()

    assert "profile activation failed" in str(failed.value.original_error)
    assert camera.is_running is True
    assert camera._pipeline is pipeline
    camera.stop()
    assert camera.is_running is False
    assert pipeline.stop_calls == 2


def test_so101_disconnect_retains_camera_but_not_closed_robot():
    from jiuwensymbiosis.adapters.so101.lowlevel import So101Driver

    class _Robot:
        def __init__(self):
            self.disconnect_calls = 0

        def disconnect(self):
            self.disconnect_calls += 1

    driver = So101Driver.__new__(So101Driver)
    driver._settle_cleanup_errors = []
    driver._cfg = SimpleNamespace(disable_torque_on_disconnect=True)
    driver._camera = _FailOnce("stop", error="camera stop failed")
    robot = _Robot()
    driver._robot = robot
    driver._kin = object()
    driver._connected = True
    driver._servo_last_send_t = 0.0
    driver._servo_planned_matrix = object()
    driver._servo_planned_q = object()
    driver._servo_endpoint_state = object()

    with pytest.raises(HardwareCleanupError):
        driver.disconnect()

    assert driver._camera is not None
    assert driver._robot is None
    assert driver._connected is False
    assert robot.disconnect_calls == 1

    driver.disconnect()
    assert driver._camera is None
    assert driver._robot is None
    assert robot.disconnect_calls == 1


def test_so101_disconnect_retains_robot_but_does_not_repeat_camera_stop():
    from jiuwensymbiosis.adapters.so101.lowlevel import So101Driver

    class _Camera:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    camera = _Camera()
    robot = _FailOnce("disconnect", error="serial disconnect failed")
    driver = So101Driver.__new__(So101Driver)
    driver._settle_cleanup_errors = []
    driver._cfg = SimpleNamespace(disable_torque_on_disconnect=True)
    driver._camera = camera
    driver._robot = robot
    driver._kin = object()
    driver._connected = True
    driver._servo_last_send_t = 0.0
    driver._servo_planned_matrix = object()
    driver._servo_planned_q = object()
    driver._servo_endpoint_state = object()

    with pytest.raises(HardwareCleanupError):
        driver.disconnect()

    assert driver._camera is None
    assert camera.stop_calls == 1
    assert driver._robot is robot
    assert driver._connected is True

    driver.disconnect()
    assert camera.stop_calls == 1
    assert robot.calls == 2
    assert driver._robot is None
    assert driver._connected is False


def test_env_disconnect_keeps_driver_bound_until_close_succeeds():
    from jiuwensymbiosis.adapters.piper.config import PiperConfig
    from jiuwensymbiosis.adapters.piper.env import PiperEnv

    env = PiperEnv(PiperConfig())
    driver = _FailOnce("close")
    env._inner = driver
    env._connected = True

    with pytest.raises(RuntimeError, match="close failed"):
        env.disconnect()
    assert env._inner is driver
    assert env._connected is True

    env.disconnect()
    assert env._inner is None
    assert env._connected is False
    assert driver.calls == 2


def test_so101_env_disconnect_keeps_driver_bound_until_close_succeeds():
    from jiuwensymbiosis.adapters.so101.config import So101Config
    from jiuwensymbiosis.adapters.so101.env import So101Env

    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    cfg = So101Config(
        port="/dev/fake",
        home_joints_deg=[0.0] * 5,
        joint_limits=dict.fromkeys(joint_names, (-90.0, 90.0)),
        safety_validated=True,
    )
    env = So101Env(cfg)
    driver = _FailOnce("close")
    env._inner = driver
    env._connected = True

    with pytest.raises(RuntimeError, match="close failed"):
        env.disconnect()
    assert env._inner is driver
    assert env._connected is True

    env.disconnect()
    assert env._inner is None
    assert env._connected is False
    assert driver.calls == 2


def test_so101_env_keeps_driver_when_connect_rollback_is_uncertain(monkeypatch):
    from jiuwensymbiosis.adapters.so101 import lowlevel
    from jiuwensymbiosis.adapters.so101.config import So101Config
    from jiuwensymbiosis.adapters.so101.env import So101Env

    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    cfg = So101Config(
        port="/dev/fake",
        home_joints_deg=[0.0] * 5,
        joint_limits=dict.fromkeys(joint_names, (-90.0, 90.0)),
        safety_validated=True,
    )

    class _Driver:
        camera_available = False

        def __init__(self):
            self.close_calls = 0

        def connect(self):
            raise HardwareCleanupError(
                "So101Driver.connect",
                RuntimeError("kinematics failed"),
                (RuntimeError("serial disconnect uncertain"),),
            )

        def close(self):
            self.close_calls += 1

    driver = _Driver()
    monkeypatch.setattr(lowlevel, "So101Driver", lambda _cfg: driver)
    env = So101Env(cfg)

    with pytest.raises(HardwareCleanupError):
        env.connect()
    assert env._inner is driver
    assert env._connected is True

    env.disconnect()
    assert driver.close_calls == 1
    assert env._inner is None
    assert env._connected is False


def test_cruzr_close_retries_failed_stages_without_repeating_completed_ones():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    class _Executor:
        def __init__(self):
            self.shutdown_calls = 0
            self.remove_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            return self.shutdown_calls > 1

        def remove_node(self, _node):
            self.remove_calls += 1
            return True

    class _Thread:
        def __init__(self):
            self.alive = True
            self.join_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):  # noqa: ARG002
            self.join_calls += 1
            if self.join_calls > 1:
                self.alive = False

    class _Node:
        def __init__(self):
            self.destroy_calls = 0

        def destroy_node(self):
            self.destroy_calls += 1

    class _Ros:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    driver = CruzrLowLevel.__new__(CruzrLowLevel)
    camera = _FailOnce("close", error="camera worker stop unconfirmed")
    nav = _CallRecorder("close")
    executor = _Executor()
    thread = _Thread()
    node = _Node()
    ros = _Ros()
    driver._camera_obj = camera
    driver._nav_obj = nav
    driver._closed = False
    driver._executor = executor
    driver._spin_thread = thread
    driver._executor_shutdown = False
    driver._node_added = True
    driver._node_removed = False
    driver._node = node
    driver._publisher = object()
    driver._subscription = object()
    driver._owns_rclpy_context = True
    driver._rclpy = ros

    with pytest.raises(HardwareCleanupError) as failed:
        driver.close()

    assert any("camera worker stop unconfirmed" in str(error) for error in failed.value.cleanup_errors)
    assert any(isinstance(error, TimeoutError) for error in failed.value.cleanup_errors)
    assert driver._camera_obj is camera
    assert driver._nav_obj is None
    assert nav.calls == 1
    assert executor.remove_calls == 0
    assert node.destroy_calls == 0
    assert ros.shutdown_calls == 0

    driver.close()
    assert driver._closed is True
    assert driver._camera_obj is None
    assert camera.calls == 2
    assert nav.calls == 1
    assert executor.shutdown_calls == 2
    assert executor.remove_calls == 1
    assert node.destroy_calls == 1
    assert ros.shutdown_calls == 1


def test_cruzr_camera_retries_only_failed_resident_workers():
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrCamera

    failed_worker = _FailOnce("stop", error="camera process state unknown")
    stopped_worker = _CallRecorder("stop")
    camera = CruzrCamera.__new__(CruzrCamera)
    camera._resident = {"failed": failed_worker, "stopped": stopped_worker}

    with pytest.raises(HardwareCleanupError):
        camera.close()

    assert camera._resident == {"failed": failed_worker}
    assert failed_worker.calls == 1
    assert stopped_worker.calls == 1

    camera.close()
    assert camera._resident == {}
    assert failed_worker.calls == 2
    assert stopped_worker.calls == 1


def test_cruzr_constructor_exposes_uncertain_partial_ros_cleanup(monkeypatch):
    from types import ModuleType

    from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
    from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel

    class _Node:
        def __init__(self):
            self.destroy_calls = 0

        def create_publisher(self, *_args):
            return object()

        def create_subscription(self, *_args):
            return object()

        def destroy_node(self):
            self.destroy_calls += 1

    class _Executor:
        def __init__(self):
            self.shutdown_calls = 0

        def add_node(self, _node):
            return False

        def shutdown(self):
            self.shutdown_calls += 1
            return False

    class _QoSProfile:
        def __init__(self, depth):
            self.depth = depth

    node = _Node()
    executor = _Executor()
    ros = ModuleType("rclpy")
    ros.__path__ = []
    ros.ok = lambda: False
    ros.init = lambda args=None, domain_id=None: None
    ros.create_node = lambda _name: node
    ros.shutdown = lambda: pytest.fail("uncertain executor must retain the ROS context")

    executors = ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = lambda: executor
    qos = ModuleType("rclpy.qos")
    qos.QoSProfile = _QoSProfile
    qos.ReliabilityPolicy = SimpleNamespace(BEST_EFFORT="best_effort")
    qos.DurabilityPolicy = SimpleNamespace(VOLATILE="volatile")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last")

    state_msgs = ModuleType("mc_state_msgs")
    state_msgs.__path__ = []
    state_msg = ModuleType("mc_state_msgs.msg")
    state_msg.RobotState = type("RobotState", (), {})
    task_msgs = ModuleType("mc_task_msgs")
    task_msgs.__path__ = []
    task_msg = ModuleType("mc_task_msgs.msg")
    task_msg.JointCmd = type("JointCmd", (), {})
    task_msg.RobotCommand = type("RobotCommand", (), {})

    for name, module in (
        ("rclpy", ros),
        ("rclpy.executors", executors),
        ("rclpy.qos", qos),
        ("mc_state_msgs", state_msgs),
        ("mc_state_msgs.msg", state_msg),
        ("mc_task_msgs", task_msgs),
        ("mc_task_msgs.msg", task_msg),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(HardwareCleanupError) as failed:
        CruzrLowLevel(CruzrConfig())

    assert isinstance(failed.value.original_error, RuntimeError)
    assert "refused its node" in str(failed.value.original_error)
    assert any(isinstance(error, TimeoutError) for error in failed.value.cleanup_errors)
    assert executor.shutdown_calls == 1
    assert node.destroy_calls == 0


def test_cruzr_env_defers_driver_close_until_camera_warmup_finishes():
    from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

    class _WarmupThread:
        def __init__(self):
            self.alive = True

        def join(self, timeout=None):  # noqa: ARG002
            pass

        def is_alive(self):
            return self.alive

    class _Driver:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    env = CruzrEnv.__new__(CruzrEnv)
    env.cfg = SimpleNamespace(camera_grab_timeout_s=0.0)
    env._inner = _Driver()
    env._connected = True
    thread = _WarmupThread()
    env._warm_camera_thread = thread

    with pytest.raises(TimeoutError, match="warm-up is still running"):
        env.disconnect()
    assert env._inner.close_calls == 0
    assert env._inner is not None
    assert env._connected is True

    thread.alive = False
    env.disconnect()
    assert env._inner is None
    assert env._connected is False


def test_cruzr_env_retains_driver_after_close_failure():
    from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

    env = CruzrEnv.__new__(CruzrEnv)
    env.cfg = SimpleNamespace(camera_grab_timeout_s=0.0)
    env._inner = _FailOnce("close", error="ROS shutdown uncertain")
    env._connected = True
    env._warm_camera_thread = None

    with pytest.raises(RuntimeError, match="ROS shutdown uncertain"):
        env.disconnect()
    assert env._inner is not None
    assert env._connected is True

    env.disconnect()
    assert env._inner is None
    assert env._connected is False
