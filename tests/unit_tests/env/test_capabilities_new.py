# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""P1: new mobility capabilities registered in KNOWN_CAPABILITIES."""

from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES


def test_new_mobility_capabilities_registered():
    for cap in ("motion.base", "motion.lift", "motion.waist", "motion.goal", "grasp.dual_arm"):
        assert cap in KNOWN_CAPABILITIES, f"{cap} missing from KNOWN_CAPABILITIES"
