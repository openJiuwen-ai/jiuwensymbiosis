# coding: utf-8
"""Color-verification of detections — reject open-vocab false positives whose pixels
contradict the prompt color (e.g. 'white table' grounded on a brown box)."""
import numpy as np
from jiuwensymbiosis.perception.vision import extract_color_word, region_color_matches


def _region(rgb_tuple, shape=(20, 20)):
    rgb = np.zeros((*shape, 3), np.uint8); rgb[:] = rgb_tuple
    return rgb, np.ones(shape, bool)


def test_extract_color_word():
    assert extract_color_word("white table") == "white"
    assert extract_color_word("brown cardboard box") == "brown"
    assert extract_color_word("box") is None
    assert extract_color_word("grey bin") == "gray"
    assert extract_color_word("silver tray") == "gray"


def test_white_region_matches_white_not_brown():
    rgb, mask = _region((113, 120, 125))          # measured white-bin mean
    assert region_color_matches(rgb, mask, "white") is True
    assert region_color_matches(rgb, mask, "brown") is False


def test_brown_region_rejects_white():
    rgb, mask = _region((64, 62, 46))             # measured brown-box mean (the false positive)
    assert region_color_matches(rgb, mask, "white") is False
    assert region_color_matches(rgb, mask, "brown") is True


def test_unknown_color_or_tiny_region_not_rejected():
    rgb, mask = _region((64, 62, 46))
    assert region_color_matches(rgb, mask, "teal") is True     # unknown color → never reject
    tiny = np.zeros((20, 20), bool); tiny[0, 0] = True
    assert region_color_matches(rgb, tiny, "white") is True    # too few pixels → don't judge
