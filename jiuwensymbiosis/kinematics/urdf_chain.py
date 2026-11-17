# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""极简 URDF 解析：抽出 root→leaf 的关节链（origin/axis/limit）。

只读取 FK/IK 需要的字段，不依赖 ros/urdf_parser。所有 revolute 关节几何旋转
均折进 origin rpy，但解析保留真实 axis 以通用支持任意 URDF。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class Joint:
    name: str
    jtype: str  # "revolute" | "continuous" | "fixed" | "prismatic"
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class Chain:
    joints: list[Joint]  # 顺序：root→leaf
    urdf_path: str = ""  # source URDF (lets pinocchio IK reuse the same model)
    leaf_link: str = ""  # the leaf link this chain ends at (pinocchio frame name)

    def movable_names(self) -> list[str]:
        return [j.name for j in self.joints if j.jtype in ("revolute", "continuous", "prismatic")]

    def limits(self) -> dict[str, tuple[float, float]]:
        return {j.name: (j.lower, j.upper) for j in self.joints if j.jtype != "fixed"}


def _triple(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(x) for x in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {len(parts)}: {text!r}")
    return (parts[0], parts[1], parts[2])


def _required_attribute(element: ET.Element | None, attribute: str, location: str) -> str:
    value = element.get(attribute) if element is not None else None
    if value is None or not value.strip():
        raise ValueError(f"URDF {location} requires a non-blank '{attribute}' attribute")
    return value


def parse_chain(urdf_path: str, root_link: str, leaf_link: str) -> Chain:
    """Parse the kinematic chain from ``root_link`` down to ``leaf_link``."""
    root = ET.parse(urdf_path).getroot()
    # Index each child by its joint element and validated identity and parent.
    by_child: dict[str, tuple[ET.Element, str, str, str]] = {}
    for index, joint in enumerate(root.findall("joint"), start=1):
        name = _required_attribute(joint, "name", f"joint #{index}")
        location = f"joint {name!r}"
        joint_type = _required_attribute(joint, "type", location)
        parent_link = _required_attribute(joint.find("parent"), "link", f"{location} <parent>")
        child_link = _required_attribute(joint.find("child"), "link", f"{location} <child>")
        by_child[child_link] = (joint, name, joint_type, parent_link)

    # Walk from leaf up to root, collecting joints.
    rev: list[Joint] = []
    link = leaf_link
    visited: set[str] = set()
    while link != root_link:
        if link in visited:
            raise ValueError(f"cycle detected in URDF chain at link {link!r}")
        visited.add(link)
        entry = by_child.get(link)
        if entry is None:
            raise ValueError(f"no joint produces link {link!r}; chain to {root_link!r} is broken")
        joint, name, joint_type, parent_link = entry
        origin = joint.find("origin")
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")
        lo = float(limit_el.get("lower", "0.0")) if limit_el is not None else 0.0
        hi = float(limit_el.get("upper", "0.0")) if limit_el is not None else 0.0
        rev.append(
            Joint(
                name=name,
                jtype=joint_type,
                xyz=_triple(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_triple(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_triple(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
                lower=lo,
                upper=hi,
            )
        )
        link = parent_link

    rev.reverse()  # root→leaf
    return Chain(joints=rev, urdf_path=urdf_path, leaf_link=leaf_link)
