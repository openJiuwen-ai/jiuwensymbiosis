"""Wheel geometry, source preservation, bounded command and scene identity gates."""

import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.common.preview import SimulationPreview
from jiuwensymbiosis.simulators.genesis.omni_base import (
    WHEEL_NAMES,
    Moz1Base,
    OmniWheelLayout,
    UpperBodyHold,
    mobile_home_targets,
)


def test_wheel_only_body_reports_empty_upper_body_hold(urdf):
    from unittest.mock import Mock

    entity = Mock()
    reading = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(0)))
    entity.get_dofs_position.return_value = [reading]
    entity.get_dofs_velocity.return_value = [reading]
    result = UpperBodyHold(entity, urdf, {}).snapshot()
    assert result["joint_names"] == []
    assert result["max_abs_error_rad"] == result["max_abs_velocity_rad_s"] == 0


@pytest.fixture
def urdf(tmp_path):
    root = ET.Element("robot", name="fixture")
    ET.SubElement(root, "link", name="base_link")
    for i, angle in enumerate((math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4, -math.pi / 4), 1):
        name = f"wheel{i:02}_joint"
        joint = ET.SubElement(root, "joint", name=name, type="continuous")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=name + "_link")
        ET.SubElement(
            joint,
            "origin",
            xyz=f"{math.copysign(0.2423, math.cos(angle))} {math.copysign(0.2423, math.sin(angle))} 0",
            rpy=f"{math.pi} 0 {angle}",
        )
        ET.SubElement(joint, "axis", xyz="-1 0 0")
        ET.SubElement(root, "link", name=name + "_link")
        for k in range(10):
            roller = ET.SubElement(root, "joint", name=f"roller_{i}_{k}", type="continuous")
            ET.SubElement(roller, "parent", link=name + "_link")
            ET.SubElement(roller, "child", link=f"roller_link_{i}_{k}")
            ET.SubElement(roller, "origin", xyz="0 .025 0")
            link = ET.SubElement(root, "link", name=f"roller_link_{i}_{k}")
            geometry = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
            ET.SubElement(geometry, "sphere", radius=".08")
    for name, offset in (("arm", None), ("finger", ".03")):
        joint = ET.SubElement(root, "joint", name=name, type="prismatic")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "axis", xyz="0 1 0")
        if offset:
            ET.SubElement(joint, "mimic", joint="arm", offset=offset, multiplier="-1")
        ET.SubElement(root, "link", name=name)
    source = tmp_path / "source.urdf"
    ET.ElementTree(root).write(source)
    return source


def test_wheel_directions_follow_actual_diagonal_omni_axes(urdf):
    layout = OmniWheelLayout.from_urdf(urdf)
    assert layout.radius_m == pytest.approx(0.105)
    assert len(layout.roller_names) == 40
    assert np.sign(layout.wheel_rates(0.1, 0, 0)).tolist() == [-1, -1, 1, 1]
    assert np.sign(layout.wheel_rates(0, 0.1, 0)).tolist() == [1, -1, -1, 1]
    assert np.sign(layout.wheel_rates(0, 0, 0.2)).tolist() == [1, 1, 1, 1]
    # Rotating a wheel frame changes its mapping; names alone never select signs.
    tree = ET.parse(urdf)
    tree.find("joint/origin").set("rpy", f"{math.pi} 0 {-math.pi / 4}")
    tree.write(urdf)
    assert OmniWheelLayout.from_urdf(urdf).wheel_rates(0.1, 0, 0)[0] > 0


def test_home_targets_preserve_rollers_mass_and_mimic_posture(urdf, tmp_path):
    before = urdf.read_bytes()
    layout = OmniWheelLayout.from_urdf(urdf)
    target = urdf
    targets = mobile_home_targets(urdf, layout)
    assert urdf.read_bytes() == before
    root = ET.parse(target)
    moving = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    assert set(moving) == set(WHEEL_NAMES) | set(layout.roller_names) | {"arm", "finger"}
    assert targets == {"arm": 0.0, "finger": 0.03}
    finger = root.find("joint[@name='finger']")
    assert finger.get("type") == "prismatic"
    assert finger.find("mimic").attrib == {"joint": "arm", "offset": ".03", "multiplier": "-1"}


def test_unsupported_asset_fails_before_simulation(urdf):
    tree = ET.parse(urdf)
    tree.getroot().remove(tree.find("joint"))
    tree.write(urdf)
    with pytest.raises(ValueError, match="wheel01_joint"):
        OmniWheelLayout.from_urdf(urdf)


@pytest.mark.parametrize("indices", [[6, 17, 28, 39], [6, 17, 29, 40]])
def test_motor_index_optimization_preserves_discovered_wheel_dofs(indices, tmp_path, monkeypatch):
    from unittest.mock import Mock

    entity = Mock()
    entity.get_joint.side_effect = [SimpleNamespace(dofs_idx_local=[i]) for i in indices]
    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = entity
    robot.upper_body = Mock()
    robot.stop = Mock()
    robot.model_manifest, robot.path = {}, tmp_path / "robot.urdf"
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.collision_audit.audit_loaded_model", Mock())
    robot.build()
    assert list(robot.wheels) == indices
    for call in (entity.set_dofs_kp.call_args, entity.set_dofs_kv.call_args):
        assert list(call.args[1]) == indices
    if indices == [6, 17, 28, 39]:
        assert isinstance(robot.wheels, range)  # No per-step CUDA index allocation.


def test_upper_hold_keeps_home_through_motion_expiry_stop_and_reset(urdf, tmp_path, monkeypatch):
    from unittest.mock import Mock

    tree = ET.parse(urdf)
    arm = tree.find("joint[@name='arm']")
    arm.set("name", "tn__RightArm0_tD")
    arm.set("type", "revolute")
    ET.SubElement(arm, "limit", lower="-1", upper="1", effort="0", velocity="0")
    finger = tree.find("joint[@name='finger']")
    finger.set("type", "revolute")
    finger.find("mimic").set("joint", "tn__RightArm0_tD")
    tree.write(urdf)
    target = urdf
    layout = OmniWheelLayout.from_urdf(urdf)
    targets = mobile_home_targets(urdf, layout)
    entity = Mock()
    indices = dict(zip(WHEEL_NAMES, (6, 17, 28, 39), strict=True)) | {"tn__RightArm0_tD": 50, "finger": 51}
    entity.get_joint.side_effect = lambda name: SimpleNamespace(dofs_idx_local=[indices[name]])
    positions = np.zeros(52)

    def tensor(values):
        return SimpleNamespace(clone=lambda: values.copy(), cpu=lambda: SimpleNamespace(numpy=lambda: values.copy()))

    entity.get_dofs_position.side_effect = lambda dofs: [tensor(positions[list(dofs)])]
    entity.get_dofs_velocity.side_effect = lambda dofs: [tensor(np.zeros(len(dofs)))]
    hold = UpperBodyHold(entity, target, targets)
    robot = Moz1Base.__new__(Moz1Base)
    robot.entity, robot.layout, robot.upper_body = entity, layout, hold
    robot._wheel_hold = None
    robot.model_manifest, robot.path = {}, tmp_path / "robot.urdf"
    monkeypatch.setattr("jiuwensymbiosis.simulators.genesis.collision_audit.audit_loaded_model", Mock())
    robot.build()
    assert hold.dofs == [50, 51] and hold.motor_dofs == [50]
    assert ET.parse(target).find("joint[@name='tn__RightArm0_tD']/limit").get("effort") == "0"
    # A loaded joint drifts. Stop/reset must retain home, not capture that drift.
    positions[50:52] = [0.02, 0.01]
    robot.command(0.1, 0, 0, duration_s=0.01)
    robot.before_step(0.01)
    robot.before_step(0.01)
    robot.stop()
    robot.reset_controller()
    entity.set_dofs_position.assert_called_once()
    initial, initial_dofs = entity.set_dofs_position.call_args.args
    assert list(initial) == [0, 0.03] and initial_dofs == [50, 51]
    upper_commands = [c for c in entity.control_dofs_position.call_args_list if c.args[1] == [50]]
    assert len(upper_commands) == 2  # Initial hold and checkpoint recovery.
    assert all(c.args[0].tolist() == [0] for c in upper_commands)
    assert all(set(c.args[1]) <= set(indices.values()) - {51} for c in entity.control_dofs_position.call_args_list)
    sample = hold.snapshot()
    assert sample["error_rad"] == pytest.approx([0.02, -0.02])
    assert sample["mimic_followers"] == 1
    assert sample["force_limit_nm"] == [120.0]


@pytest.mark.parametrize("home", [2.0, float("nan")])
def test_upper_hold_rejects_invalid_home_before_any_motor_write(tmp_path, home):
    from unittest.mock import Mock

    path = tmp_path / "upper.urdf"
    path.write_text(
        '<robot><joint name="tn__LegWaist0_tD" type="revolute"><limit lower="-1" upper="1"/></joint></robot>'
    )
    entity = Mock()
    with pytest.raises(ValueError, match="outside joint limits"):
        UpperBodyHold(entity, path, {"tn__LegWaist0_tD": home})
    assert not entity.mock_calls


def test_collision_state_reads_solver_flags_instead_of_a_fixed_label():
    solver = SimpleNamespace(
        _enable_collision=True,
        _enable_self_collision=True,
        _enable_adjacent_collision=False,
        _enable_neutral_collision=False,
    )
    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = SimpleNamespace(solver=solver)
    assert robot.collision_settings() == {
        "enabled": True,
        "self_collision": True,
        "adjacent_collision": False,
        "neutral_collision": False,
    }
    solver._enable_self_collision = False
    assert robot.collision_settings()["self_collision"] is False


def test_repeated_velocity_preserves_expiry_and_reapplies_after_stop(urdf):
    from unittest.mock import Mock

    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = Mock()
    robot.entity.get_dofs_position.return_value = [SimpleNamespace(clone=lambda: np.zeros(4))]
    robot.layout = OmniWheelLayout.from_urdf(urdf)
    robot.wheels = range(6, 40, 11)
    robot.command(0.1, 0, 0, duration_s=0.02)
    robot.before_step(0.01)
    robot.before_step(0.01)
    assert robot.entity.control_dofs_velocity.call_count == 1
    robot.before_step(0.01)
    robot.entity.control_dofs_position.assert_called_once()
    robot.command(0.1, 0, 0, duration_s=0.02)
    robot.before_step(0.01)
    assert robot.entity.control_dofs_velocity.call_count == 2


@pytest.mark.parametrize(
    ("body_command", "expected_root_twist"),
    [
        ((0.1, 0, 0), (0, -0.1, 0)),
        ((-0.1, 0, 0), (0, 0.1, 0)),
        ((0, 0.1, 0), (0.1, 0, 0)),
        ((0, -0.1, 0), (-0.1, 0, 0)),
        ((0, 0, 0.2), (0, 0, 0.2)),
        ((0.1, -0.05, -0.2), (-0.05, -0.1, -0.2)),
    ],
)
def test_command_follows_robot_front_not_urdf_root_x(urdf, body_command, expected_root_twist):
    # Independent asset convention from RoboSim's MOZ1 profile:
    # relative_forward_offset_deg=-90; root -Y is front, root +X is left.
    class Motor:
        def control_dofs_velocity(self, values, indices):
            self.values = values

    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = Motor()
    robot.layout = OmniWheelLayout.from_urdf(urdf)
    robot.wheels = [6, 17, 28, 39]
    robot.command(*body_command)
    robot.before_step(0.01)
    root_twist = np.linalg.lstsq(robot.layout.rows, robot.entity.values, rcond=None)[0]
    assert root_twist == pytest.approx(expected_root_twist, abs=1e-12)
    assert robot.twist == body_command  # Public state remains in semantic body coordinates.


def test_command_expiry_stops_only_wheel_motors_and_invalid_input_is_atomic(urdf):
    class Motor:
        def get_dofs_position(self, indices):
            return [SimpleNamespace(clone=lambda: np.zeros(4))]

        def control_dofs_velocity(self, values, indices):
            self.values, self.indices = list(values), list(indices)

        def control_dofs_position(self, values, indices):
            self.values, self.indices = list(values), list(indices)

    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = Motor()
    robot.layout = OmniWheelLayout.from_urdf(urdf)
    robot.wheels = [6, 17, 28, 39]
    robot.command(0.1, 0, 0, duration_s=0.02)
    for args in ((float("nan"), 0, 0), (True, 0, 0), (0.3, 0.3, 0), (0, 0, 0.61)):
        with pytest.raises(ValueError):
            robot.command(*args)
    assert robot.twist == (0.1, 0, 0)
    for duration in (0, -1, 3, float("inf"), "1", True):
        with pytest.raises(ValueError):
            robot.command(0, 0, 0, duration)
    robot.before_step(0.01)
    assert any(robot.entity.values)
    robot.before_step(0.01)
    robot.before_step(0.01)
    assert robot.entity.values == [0, 0, 0, 0]
    assert robot.entity.indices == robot.wheels
    assert robot.twist == (0, 0, 0)


@pytest.mark.parametrize("trigger", ["expiry", "stop", "zero_command"])
def test_parking_holds_one_wheel_position_target_until_motion_resumes(urdf, trigger):
    class Motor:
        def __init__(self):
            self.positions = np.array([1.0, 2.0, 3.0, 4.0])
            self.position_targets = []

        def get_dofs_position(self, indices):
            assert indices == [6, 17, 28, 39]
            return [SimpleNamespace(clone=lambda: self.positions.copy())]

        def control_dofs_velocity(self, values, indices):
            self.mode = "velocity"

        def control_dofs_position(self, values, indices):
            assert indices == [6, 17, 28, 39]
            self.mode = "position"
            self.position_targets.append(np.array(values))

    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = Motor()  # No root setters: braking must use wheel motors.
    robot.layout = OmniWheelLayout.from_urdf(urdf)
    robot.wheels = [6, 17, 28, 39]
    robot._wheel_hold = None
    robot.command(0.1, 0, 0, duration_s=0.01)
    robot.before_step(0.01)
    if trigger == "expiry":
        robot.before_step(0.01)
    elif trigger == "stop":
        robot.stop()
    else:
        robot.command(0, 0, 0)
        robot.before_step(0.01)
    assert robot.entity.mode == "position"
    assert robot.entity.position_targets[-1] == pytest.approx([1, 2, 3, 4])
    robot.entity.positions += 0.02  # Contact load moves wheels; target must not follow it.
    for _ in range(3):
        robot.stop()  # Worker may repeat this on disconnect / expired request.
        robot.command(0, 0, 0)
        robot.before_step(0.01)
    assert robot.entity.position_targets[-1] == pytest.approx([1, 2, 3, 4])
    assert len(robot.entity.position_targets) == 1  # Persistent motor target needs one upload per stop.
    with pytest.raises(ValueError):
        robot.command(float("nan"), 0, 0)
    assert robot.entity.mode == "position"
    robot.command(0, 0.1, 0)
    robot.before_step(0.01)
    assert robot.entity.mode == "velocity"
    robot.stop()
    assert robot.entity.position_targets[-1] == pytest.approx([1.02, 2.02, 3.02, 4.02])


@pytest.mark.parametrize("fault", ["parent", "child", "roller_link", "radius"])
def test_incomplete_wheel_geometry_raises_explicit_error(urdf, fault):
    tree = ET.parse(urdf)
    root = tree.getroot()
    wheel = root.find("joint[@name='wheel01_joint']")
    if fault in {"parent", "child"}:
        wheel.remove(wheel.find(fault))
    elif fault == "roller_link":
        root.remove(root.find("link[@name='roller_link_1_0']"))
    else:
        root.find("link[@name='roller_link_1_0']/collision/geometry/sphere").attrib.pop("radius")
    tree.write(urdf)
    with pytest.raises(ValueError, match="requires|Missing roller|radius"):
        OmniWheelLayout.from_urdf(urdf)


def test_preview_binds_pulse_to_live_scene_and_clears_on_disconnect(tmp_path):
    preview = SimulationPreview(state_dir=tmp_path)
    scene_id = "a" * 32
    preview._state = {
        "state": "ready",
        "scene": {"instance_id": scene_id},
        "robot": {"control_mode": "wheel_velocity_physics"},
    }
    args = {"vx": 0.15, "vy": 0, "wz": 0, "scene_instance_id": scene_id, "config_revision": 0}
    with pytest.raises(RuntimeError):
        preview.base_command(**args)
    preview.set_streaming(True)
    first = preview.base_command(**args)
    with pytest.raises(ValueError, match="场景已切换"):
        preview.base_command(**{**args, "config_revision": 1})
    assert preview._base_request["id"] == first["queued_id"]
    preview.set_streaming(False)
    assert preview._base_request is None


@pytest.mark.parametrize("limits", [None, {"linear_m_s": 0.4, "angular_rad_s": 1.5}])
def test_manual_pulse_enforces_current_robot_envelope(tmp_path, limits):
    preview = SimulationPreview(state_dir=tmp_path)
    scene_id = "a" * 32
    preview._state = {
        "state": "ready",
        "scene": {"instance_id": scene_id},
        "robot": {"control_mode": "policy_locomotion", "command_limits": limits},
    }
    preview.set_streaming(True)
    linear = limits["linear_m_s"] if limits else 0.3
    angular = limits["angular_rad_s"] if limits else 0.6
    binding = {"scene_instance_id": scene_id, "config_revision": 0}
    for command in ((linear, 0, 0), (0, -linear, 0), (0, 0, angular), (0, 0, 0)):
        accepted = preview.base_command(vx=command[0], vy=command[1], wz=command[2], **binding)
        assert accepted["max_duration_s"] == 1
    last = preview._base_request.copy()
    for command in ((linear + 0.001, 0, 0), (linear, linear, 0), (0, 0, angular + 0.001), (True, 0, 0)):
        with pytest.raises(ValueError):
            preview.base_command(vx=command[0], vy=command[1], wz=command[2], **binding)
        assert preview._base_request == last
