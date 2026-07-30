# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hand-eye calibration scripts sub-package.

Groups the vendor-specific calibration entry points (``calibrate_hand_eye``
for piper eye-in-hand, ``so101_eye_calib`` for SO-101 eye-to-hand) with the
shared calibration engine (``handeye_core``) and board detection helpers
(``handeye_board``) so they can be imported as ``scripts.calibrate.<name>``
while still being runnable as plain scripts (``python scripts/calibrate/<x>.py``),
relying on ``sys.path[0]`` for same-directory sibling imports.
"""
