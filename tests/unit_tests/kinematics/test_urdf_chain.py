# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``Chain.limits()`` is what a body's ``joint_limits`` is built from, and therefore what
decides whether SafetyRail range-checks a joint command at all.

It is a METHOD, not a property: reading ``chain.limits`` instead of calling it yields a bound
method, which is truthy, non-empty-looking, and silently useless as a limits mapping — an
adapter that made that mistake reported "limits available" while every range check was
skipped. These tests run against the SO-101 description that ships in the package, so the
contract is verified on any machine, with no robot description checked out.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from jiuwensymbiosis.kinematics.urdf_chain import parse_chain

URDF = str(
    Path(__file__).resolve().parents[3]
    / "jiuwensymbiosis"
    / "adapters"
    / "so101"
    / "description"
    / "so101_new_calib.urdf"
)
REVOLUTE = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}


@pytest.fixture
def minimal_robot() -> ET.Element:
    return ET.fromstring(
        """
        <robot name="fixture">
            <link name="base"/>
            <link name="tip"/>
            <joint name="hinge" type="revolute">
                <parent link="base"/>
                <child link="tip"/>
            </joint>
        </robot>
        """
    )


@pytest.mark.parametrize("value", [None, "", "   "], ids=["missing", "empty", "blank"])
@pytest.mark.parametrize(
    ("element", "attribute", "location"),
    [
        ("joint", "name", "joint #1"),
        ("joint", "type", "joint 'hinge'"),
        ("joint/parent", "link", "joint 'hinge' <parent>"),
        ("joint/child", "link", "joint 'hinge' <child>"),
    ],
)
def test_parse_chain_rejects_missing_or_blank_joint_attributes(
    tmp_path, minimal_robot, element, attribute, location, value
):
    target = minimal_robot.find(element)
    assert target is not None
    if value is None:
        del target.attrib[attribute]
    else:
        target.set(attribute, value)
    urdf = tmp_path / "robot.urdf"
    ET.ElementTree(minimal_robot).write(urdf)

    with pytest.raises(ValueError, match=rf"{location}.*'{attribute}'"):
        parse_chain(str(urdf), "base", "tip")


@pytest.mark.parametrize("side", ["parent", "child"])
def test_parse_chain_rejects_missing_joint_link_element(tmp_path, minimal_robot, side):
    joint = minimal_robot.find("joint")
    assert joint is not None
    link = joint.find(side)
    assert link is not None
    joint.remove(link)
    urdf = tmp_path / "robot.urdf"
    ET.ElementTree(minimal_robot).write(urdf)

    with pytest.raises(ValueError, match=rf"joint 'hinge' <{side}>.*'link'"):
        parse_chain(str(urdf), "base", "tip")


class TestChainLimits:
    def test_returns_a_mapping_not_a_bound_method(self):
        limits = parse_chain(URDF, "base_link", "gripper_frame_link").limits()
        assert isinstance(limits, dict)
        assert limits, "empty limits would disable every range check downstream"

    def test_covers_every_movable_joint(self):
        limits = parse_chain(URDF, "base_link", "gripper_frame_link").limits()
        assert set(limits) == REVOLUTE

    def test_omits_fixed_joints(self):
        """A fixed joint has no range to state, so naming one would invent a limit."""
        limits = parse_chain(URDF, "base_link", "gripper_frame_link").limits()
        assert "gripper_frame_joint" not in limits

    def test_each_entry_is_an_ordered_lo_hi_pair(self):
        limits = parse_chain(URDF, "base_link", "gripper_frame_link").limits()
        for name, bounds in limits.items():
            lo, hi = bounds
            assert lo < hi, f"{name}: lower {lo} is not below upper {hi}"
