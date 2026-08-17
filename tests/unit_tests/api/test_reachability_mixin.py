# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Framework Reachability: URDF-based reach judge with a clean degrade-to-None when the body
exposes no URDF, so a body can hold it without owning a URDF."""

from types import SimpleNamespace

from jiuwensymbiosis.api.components import Reachability


def _Api(env):
    """A body holds the judge; the component reads the env through it."""
    return Reachability(SimpleNamespace(env=env))


def test_degrades_to_none_without_urdf():
    api = _Api(SimpleNamespace(urdf_path=None, arm_chains=None))
    assert api.check_reachable({"center_mm": [500, 0, 600]}) is None
    assert api.describe_reach() is None


def test_degrades_when_arm_chains_missing():
    api = _Api(SimpleNamespace(urdf_path="/some.urdf", arm_chains=None))
    assert api.check_reachable({"center_mm": [500, 0, 600]}) is None
    assert api.describe_reach() is None


def test_bad_target_returns_none():
    api = _Api(SimpleNamespace(urdf_path="/some.urdf", arm_chains={"l": ("base_link", "leaf")}))
    assert api.check_reachable({"center_mm": None}) is None
    assert api.check_reachable({}) is None


def test_the_component_declares_no_capability_of_its_own():
    """Holding a judge is not the same as having a URDF prior: the body declares
    planning.reachability itself, so the claim tracks the hardware, not the class list."""
    assert not hasattr(Reachability, "capability")
