"""Wheel geometry, source preservation, bounded command and scene identity gates."""

import math
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from jiuwensymbiosis.simulators.genesis.omni_base import (
    WHEEL_NAMES,
    Moz1Base,
    OmniWheelLayout,
    prepare_mobile_urdf,
)


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


def test_disposable_copy_retains_rollers_mass_and_mimic_posture(urdf, tmp_path):
    before = urdf.read_bytes()
    layout = OmniWheelLayout.from_urdf(urdf)
    target = tmp_path / "mobile.urdf"
    prepare_mobile_urdf(urdf, target, layout)
    assert urdf.read_bytes() == before
    root = ET.parse(target)
    moving = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    assert set(moving) == set(WHEEL_NAMES) | set(layout.roller_names)
    finger = root.find("joint[@name='finger']")
    assert np.fromstring(finger.find("origin").get("xyz"), sep=" ") == pytest.approx([0, 0.03, 0])
    assert finger.find("mimic") is None


def test_unsupported_asset_fails_before_simulation(urdf):
    tree = ET.parse(urdf)
    tree.getroot().remove(tree.find("joint"))
    tree.write(urdf)
    with pytest.raises(ValueError, match="wheel01_joint"):
        OmniWheelLayout.from_urdf(urdf)


@pytest.mark.parametrize("indices", [[6, 17, 28, 39], [6, 17, 29, 40]])
def test_motor_index_optimization_preserves_discovered_wheel_dofs(indices):
    from unittest.mock import Mock

    entity = Mock()
    entity.get_joint.side_effect = [SimpleNamespace(dofs_idx_local=[i]) for i in indices]
    robot = Moz1Base.__new__(Moz1Base)
    robot.entity = entity
    robot.stop = Mock()
    robot.build()
    assert list(robot.wheels) == indices
    for call in (entity.set_dofs_kp.call_args, entity.set_dofs_kv.call_args):
        assert list(call.args[1]) == indices
    if indices == [6, 17, 28, 39]:
        assert isinstance(robot.wheels, range)  # No per-step CUDA index allocation.


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
