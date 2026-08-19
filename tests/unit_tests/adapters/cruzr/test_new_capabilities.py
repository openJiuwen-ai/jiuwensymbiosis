# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""P1: Cruzr declares the 5 new mobility capabilities (env + api)."""

_NEW = ("motion.base", "motion.lift", "motion.waist", "motion.goal", "grasp.dual_arm")


def test_env_declares_mobility_capabilities():
    from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv

    for cap in _NEW:
        assert cap in CruzrEnv.capabilities


def test_api_capabilities_include_new():
    from jiuwensymbiosis.adapters.cruzr.api import CruzrApi

    # ``capabilities`` reads only ``type(self).__mro__``, so no hardware is needed —
    # and going through the real property is what makes this test track the framework's
    # derivation (implemented actions + marker attrs) instead of a copy of it.
    api = object.__new__(CruzrApi)
    for cap in _NEW:
        assert cap in api.capabilities, f"{cap} not contributed by any action CruzrApi implements"
