# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""极简 URDF 解析：抽出 root→leaf 的关节链（origin/axis/limit）。

只读取 FK/IK 需要的字段，不依赖 ros/urdf_parser。所有 revolute 关节几何旋转
均折进 origin rpy，但解析保留真实 axis 以通用支持任意 URDF。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional


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


def _triple(text: Optional[str], default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(x) for x in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {len(parts)}: {text!r}")
    return (parts[0], parts[1], parts[2])


def parse_chain(urdf_path: str, root_link: str, leaf_link: str) -> Chain:
    """Parse the kinematic chain from ``root_link`` down to ``leaf_link``."""
    root = ET.parse(urdf_path).getroot()
    # child_link -> joint element
    by_child: dict[str, ET.Element] = {}
    for j in root.findall("joint"):
        child = j.find("child")
        if child is not None:
            by_child[child.get("link")] = j

    # Walk from leaf up to root, collecting joints.
    rev: list[Joint] = []
    link = leaf_link
    visited: set[str] = set()
    while link != root_link:
        if link in visited:
            raise ValueError(f"cycle detected in URDF chain at link {link!r}")
        visited.add(link)
        j = by_child.get(link)
        if j is None:
            raise ValueError(f"no joint produces link {link!r}; chain to {root_link!r} is broken")
        origin = j.find("origin")
        axis_el = j.find("axis")
        limit_el = j.find("limit")
        lo = float(limit_el.get("lower", "0.0")) if limit_el is not None else 0.0
        hi = float(limit_el.get("upper", "0.0")) if limit_el is not None else 0.0
        rev.append(
            Joint(
                name=j.get("name"),
                jtype=j.get("type"),
                xyz=_triple(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_triple(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_triple(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
                lower=lo,
                upper=hi,
            )
        )
        link = j.find("parent").get("link")

    rev.reverse()  # root→leaf
    return Chain(joints=rev, urdf_path=urdf_path, leaf_link=leaf_link)
