# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from pathlib import Path
from xml.etree import ElementTree

from jiuwensymbiosis.adapters.so101 import lowlevel


def test_packaged_urdf_is_kinematics_only():
    urdf_path = Path(lowlevel.__file__).resolve().parent / "description" / "so101_new_calib.urdf"
    root = ElementTree.parse(urdf_path).getroot()

    assert {link.attrib["name"] for link in root.findall("./link")} == {
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "gripper_frame_link",
    }
    assert {joint.attrib["name"] for joint in root.findall("./joint")} == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper_frame_joint",
    }
    for tag in ("inertial", "visual", "collision", "mesh", "transmission", "material"):
        assert root.find(f".//{tag}") is None
