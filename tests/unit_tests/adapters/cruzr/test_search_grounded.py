# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""S2 grounded head search: ``search_target(object, reference=...)`` grabs the head RIGHT-EYE raw
image and judges "target ON reference" purely by 2-D BBOX OVERLAP — the fraction of the target's
bounding box lying inside the reference's bounding box (``|t∩r| / |t|``, boxes not masks) must clear
``head_on_overlap_min`` — before returning a bearing. Bbox (not pixel-mask) overlap so a small
object resting on top of a tall box seen from the front (mask overlap ≈0) still registers as "on".
No depth / point cloud / TF. When the reference can't be detected the frame degrades fail-open to
bearing-only (``note=head_reference_undetected_degraded``); strict (``head_grounded_strict``)
refuses the bearing instead; ``head_ground_verify=False`` forces the plain bearing path.

Fully offline: a 40×40 RGB frame, stubbed ``grab_frames(camera='head')`` and ``_seg_fn`` (colourless
"box"/"table" nouns so the shared colour check is skipped), and a stubbed ``_ensure_detector``.
No ROS / rclpy / hardware.
"""

import numpy as np

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig

TABLE_BOX = [4, 20, 35, 38]   # reference (table) bbox: x 4..35, y 20..38
BOX_X = (14, 26)              # target bbox x-span, centred in the 40-wide frame → bearing ≈ 0


class _LL:
    """Fake low-level: the head grab is a fixed RGB frame (no depth/cloud/TF)."""

    def __init__(self, rgb):
        self._rgb = rgb

    def grab_frames(self, camera="waist"):
        # search_target looks through EVERY camera now; both of cruzr's serve the same
        # rendered scene here, so which one hits is not what these tests are about.
        assert camera in ("head", "waist")
        return self._rgb, None, None, None


class _Env:
    def __init__(self, rgb, **cfg_over):
        self.low_level = _LL(rgb)
        self.cfg = CruzrConfig()
        for k, v in cfg_over.items():
            setattr(self.cfg, k, v)


def _target_bbox(kind):
    """Target (box) bbox for each relation kind — its overlap over the fixed table bbox drives the
    2-D judgement (the mask is incidental, only for the shared colour check / viz):
      'on'      box y 22..30 fully inside the table bbox → overlap 1.0
      'partial' box y 14..26 straddling the table's top edge (y=20) → overlap 0.5 (≥ default 0.15)
      'off'     box y  2..10 above the table bbox → overlap 0.0
    The x-span is always centred → bearing ≈ 0.
    """
    y = {"on": (22, 30), "partial": (14, 26), "off": (2, 10)}.get(kind, (22, 30))
    return [BOX_X[0], y[0], BOX_X[1], y[1]]


def _seg_factory(kind):
    """A ``_seg_fn`` returning the table detection for a table-ish prompt, else the box detection.
    ``kind='no_table'`` → the reference prompt returns nothing (undetected); ``'degenerate_table'``
    → the reference returns a zero-area bbox (unusable footprint → degrade)."""
    tbox = _target_bbox(kind)
    box_mask = np.zeros((40, 40), dtype=bool)
    box_mask[int(tbox[1]):int(tbox[3]), int(tbox[0]):int(tbox[2])] = True   # non-empty → not skipped
    table_mask = np.zeros((40, 40), dtype=bool)
    table_mask[TABLE_BOX[1]:TABLE_BOX[3], TABLE_BOX[0]:TABLE_BOX[2]] = True

    def _seg(img, text_prompt="box"):
        if "table" in text_prompt:
            if kind == "no_table":
                return []
            ref_box = [10, 10, 10, 10] if kind == "degenerate_table" else TABLE_BOX
            return [{"score": 0.9, "label": "table", "box": ref_box, "mask": table_mask}]
        return [{"score": 0.9, "label": "box", "box": tbox, "mask": box_mask}]

    return _seg


def _api(kind, **cfg_over):
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    api = CruzrApi(_Env(rgb, **cfg_over))
    api._ensure_detector = lambda: None                 # type: ignore[method-assign]
    api._seg_fn = _seg_factory(kind)                    # type: ignore[method-assign]
    return api


def test_search_target_grounded_on_surface_pass():
    # Target bbox fully inside the table bbox (overlap 1.0 ≥ threshold) → verified on-surface.
    api = _api("on")
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True
    assert out["verified"] is True
    assert out["reference"] == "table"
    assert out["overlap"] >= 0.99
    assert abs(out["bearing_rad"]) < 1e-6       # box bbox centred → zero bearing
    assert "note" not in out                     # a real 2-D evaluation, not a degrade


def test_search_target_grounded_partial_overlap_passes_threshold():
    # Target bbox straddles the table edge (overlap 0.5 ≥ default 0.15) → still verified on-surface.
    api = _api("partial")
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True and out["verified"] is True
    assert 0.15 <= out["overlap"] < 1.0
    assert "note" not in out


def test_search_target_grounded_off_surface_reject():
    # Target bbox does not overlap the table bbox → genuine reject (evaluated, not a degrade).
    api = _api("off")
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is False
    assert out["verified"] is True
    assert out["reason"] == "no_target_on_reference"
    assert "note" not in out


def test_search_target_grounded_high_threshold_rejects_partial():
    # A stricter overlap threshold turns the partial-overlap case into a reject (knob works).
    api = _api("partial", head_on_overlap_min=0.9)
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is False and out["verified"] is True
    assert out["reason"] == "no_target_on_reference"


def test_search_target_grounded_degrades_when_reference_undetected():
    # Reference not detected → can't judge overlap → fall-open to bearing-only with a note.
    api = _api("no_table")
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True
    assert out.get("note") == "head_reference_undetected_degraded"
    assert "verified" not in out                 # bearing-only path never claims verification


def test_search_target_grounded_degrades_when_reference_bbox_degenerate():
    # Reference detected but with a zero-area bbox → no usable footprint → degrade.
    api = _api("degenerate_table")
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True
    assert out.get("note") == "head_reference_undetected_degraded"


def test_search_target_strict_reference_undetected_refuses_bearing():
    # head_grounded_strict (fail-closed): reference undetected → can't verify → NOT found, so the
    # caller refuses to advance on bearing alone (unlike the fail-open degrade above).
    api = _api("no_table", head_grounded_strict=True)
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is False
    assert out["reason"] == "head_reference_undetected_strict"
    assert out["verified"] is False
    assert out.get("note") == "head_reference_undetected_degraded"


def test_search_target_strict_still_passes_when_verified_on_surface():
    # Strict only gates the UNVERIFIABLE case: a real overlap pass is unaffected → found.
    api = _api("on", head_grounded_strict=True)
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True and out["verified"] is True
    assert "note" not in out                     # a real evaluation, not a strict degrade


def test_search_target_strict_genuine_off_surface_reject_unchanged():
    # Strict does NOT relabel a genuine off-surface reject: still found=False, verified=True, with the
    # real reason (not the strict-degrade reason) — the 2-D relation WAS evaluated, it just failed.
    api = _api("off", head_grounded_strict=True)
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is False and out["verified"] is True
    assert out["reason"] == "no_target_on_reference"
    assert "note" not in out


def test_head_ground_verify_false_forces_bearing():
    # Global kill-switch off → grounded path skipped entirely even with on= set.
    api = _api("on", head_ground_verify=False)
    out = api.search_target("box", reference="table")
    assert out["ok"] and out["found"] is True
    assert "verified" not in out                 # plain bearing, no 2-D grounding
    assert "reference" not in out
    assert "note" not in out                      # not grounded → not a degrade either


def test_search_target_no_on_is_plain_bearing():
    # No reference → always the plain bearing path (no grounding).
    api = _api("on")
    out = api.search_target("box")
    assert out["ok"] and out["found"] is True
    assert "verified" not in out
    assert "note" not in out


def test_camera_worker_parses_pointcloud_args():
    # The worker's arg surface still accepts the (legacy) head point-cloud flags (no rclpy import).
    from jiuwensymbiosis.adapters.cruzr.ros2.camera_worker import _build_parser

    args = _build_parser().parse_args([
        "--color-topic", "/head/color",
        "--output-dir", "/tmp/x",
        "--pointcloud-topic", "/sensor/camera/stereo/pointcloud/jazzy",
        "--pointcloud-msg-type", "sensor_msgs/msg/PointCloud2",
        "--camera-optical-frame", "stereo_left_rgb_optical_link",
        "--rectify-cruzr-stereo-left",
        "--sync-tolerance-s", "0.08",
        "--ensure-rgb",
    ])
    assert args.pointcloud_topic == "/sensor/camera/stereo/pointcloud/jazzy"
    assert args.pointcloud_msg_type == "sensor_msgs/msg/PointCloud2"
    assert args.camera_optical_frame == "stereo_left_rgb_optical_link"
    assert args.rectify_cruzr_stereo_left is True
    assert args.sync_tolerance_s == 0.08
    assert args.ensure_rgb is True
