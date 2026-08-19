# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Taking one sensing action must not mean taking its neighbours.

While ``_Scene3DBody`` was a base class, wanting ``locate_for_grasp`` meant inheriting
``locate_for_place`` and ``analyze_scene`` too — and, because capabilities are derived from
the actions a body implements, it also meant advertising a detector the body might not have.
That bundling is why piper and so101 never took the 3-D sensing at all. Holding the component
instead of inheriting it is what removes the bundle: the body declares what it offers, one
action at a time, and its api file is the whole list.
"""

from __future__ import annotations

from jiuwensymbiosis.api.actions import LOCATE_FOR_GRASP, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.components import Scene3D
from jiuwensymbiosis.env.mock import MockArmEnv


class OneActionBody(BaseRobotApi):
    """A body that wants the 3-D measurement and nothing else."""

    def __init__(self, env):
        super().__init__(env)
        self._scene = Scene3D(self)

    @implements(LOCATE_FOR_GRASP)
    def locate_for_grasp(self, object_name: str = "box", reference: str | None = None,
                         relation: str = "on") -> dict:
        return self._scene.locate_for_grasp(object_name, reference, relation)


def test_one_action_can_be_taken_alone():
    api = OneActionBody(MockArmEnv())
    assert hasattr(api, "locate_for_grasp")
    assert not hasattr(api, "locate_for_place"), "taking one must not drag in its neighbour"
    assert not hasattr(api, "analyze_scene")


def test_the_api_file_is_the_whole_list():
    """Every action a planner can see is declared in the class itself — none arrive by
    inheritance, so reading the adapter tells you exactly what the robot offers."""
    declared = {n for n in vars(OneActionBody) if hasattr(getattr(OneActionBody, n, None), "__robot_tool__")}
    inherited = {
        n
        for cls in OneActionBody.__mro__[1:]
        for n in vars(cls)
        if hasattr(getattr(cls, n, None), "__robot_tool__")
    }
    assert declared == {"locate_for_grasp"}
    assert inherited == {"home"}, f"only home is inherited (every body owes one); got {inherited}"


def test_capability_follows_the_action_not_the_component():
    """Holding a Scene3D is not evidence of a detector — implementing an action is."""
    api = OneActionBody(MockArmEnv())
    assert api.capabilities == frozenset({"vision.detection"})

    class HoldsButDeclaresNothing(BaseRobotApi):
        def __init__(self, env):
            super().__init__(env)
            self._scene = Scene3D(self)

    assert HoldsButDeclaresNothing(MockArmEnv()).capabilities == frozenset()
