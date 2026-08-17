# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The closed spatial-relation vocabulary and its geometry.

A task names WHICH instance it means by relating it to something else — "the box on the
table", "the box beside the hat". These tests pin what each relation actually decides,
because that decision is the difference between picking up the right box and picking up
a box. The scene below is the one that motivated the vocabulary: a hat and a box side by
side on a table, an identical box further off, and a third box on the shelf overhead.
"""

from __future__ import annotations

import pytest

from jiuwensymbiosis.perception.scene3d import SPATIAL_RELATIONS, extent_of, relation_holds

# One table, and on it a hat with a box right next to it (mm, base frame).
TABLE = {
    "center_mm": [1600.0, 0.0, 700.0],
    "front_x_mm": 1300.0,
    "back_x_mm": 1900.0,
    "width_mm": 1200.0,
    "surface_z_mm": 750.0,
}
HAT = {
    "center_mm": [1600.0, 500.0, 800.0],
    "front_x_mm": 1500.0,
    "back_x_mm": 1700.0,
    "width_mm": 200.0,
    "top_z_mm": 850.0,
    "height_mm": 100.0,
}
BOX_BESIDE_HAT = {
    "center_mm": [1600.0, 200.0, 850.0],
    "front_x_mm": 1450.0,
    "back_x_mm": 1750.0,
    "width_mm": 300.0,
    "top_z_mm": 950.0,
    "height_mm": 200.0,
}
BOX_FAR = {
    "center_mm": [1600.0, -900.0, 850.0],
    "front_x_mm": 1450.0,
    "back_x_mm": 1750.0,
    "width_mm": 300.0,
    "top_z_mm": 950.0,
    "height_mm": 200.0,
}
# Directly above the hat, on the shelf: close in XY, nowhere near it in Z.
BOX_ON_SHELF_ABOVE = {
    "center_mm": [1600.0, 500.0, 1500.0],
    "front_x_mm": 1500.0,
    "back_x_mm": 1700.0,
    "width_mm": 200.0,
    "top_z_mm": 1600.0,
    "height_mm": 200.0,
}


class TestTheVocabularyIsClosed:
    def test_only_viewpoint_independent_relations(self):
        # "left of" / "in front of" are excluded on purpose: they are defined relative to an
        # observer, and the observer drives around, so a plan carrying one stops being true.
        assert SPATIAL_RELATIONS == ("on", "under", "in", "beside", "near")

    @pytest.mark.parametrize("bogus", ["left_of", "in_front_of", "ON", "next_to", ""])
    def test_an_unknown_relation_raises_rather_than_guessing(self, bogus):
        with pytest.raises(ValueError, match="unknown spatial relation"):
            relation_holds(BOX_BESIDE_HAT, HAT, bogus)


class TestOnAndUnderAreOneRelationBothWaysRound:
    def test_the_hat_is_on_the_table(self):
        assert relation_holds(HAT, TABLE, "on")

    def test_the_table_is_under_the_hat(self):
        assert relation_holds(TABLE, HAT, "under")

    def test_and_not_the_other_way_about(self):
        assert not relation_holds(TABLE, HAT, "on")
        assert not relation_holds(HAT, TABLE, "under")


class TestBeside:
    def test_the_box_next_to_the_hat_qualifies(self):
        assert relation_holds(BOX_BESIDE_HAT, HAT, "beside")

    def test_an_identical_box_across_the_table_does_not(self):
        assert not relation_holds(BOX_FAR, HAT, "beside")

    def test_a_box_on_the_shelf_overhead_does_not(self):
        """The whole point of the z test: it is directly above the hat, not beside it."""
        assert not relation_holds(BOX_ON_SHELF_ABOVE, HAT, "beside")

    def test_stacked_is_not_beside(self):
        # The hat sits ON the table, so it is not one of the table's neighbours.
        assert not relation_holds(HAT, TABLE, "beside")

    def test_the_gap_threshold_is_the_caller_s(self):
        assert relation_holds(BOX_FAR, HAT, "beside", beside_max_gap_mm=2000.0)


class TestNear:
    def test_near_ignores_height_where_beside_does_not(self):
        """That difference is the reason both exist — 'near' is the loose one."""
        assert relation_holds(BOX_ON_SHELF_ABOVE, HAT, "near")
        assert not relation_holds(BOX_ON_SHELF_ABOVE, HAT, "beside")

    def test_far_enough_away_is_not_near(self):
        assert not relation_holds(BOX_FAR, HAT, "near", near_max_dist_mm=500.0)


class TestExtentReadsEitherShape:
    def test_a_surface_record_is_a_plane(self):
        """It reports surface_z_mm and states no thickness, so top == bottom."""
        e = extent_of(TABLE)
        assert e.top_z_mm == e.bottom_z_mm == 750.0

    def test_an_object_record_keeps_its_height(self):
        e = extent_of(HAT)
        assert (e.top_z_mm, e.bottom_z_mm) == (850.0, 750.0)

    def test_attributes_work_as_well_as_keys(self):
        """Candidates arrive as ObjectGeometry3D, references sometimes as dicts."""
        from jiuwensymbiosis.perception.object_geometry import ObjectGeometry3D

        geo = ObjectGeometry3D(
            ok=True,
            reason="",
            center_mm=(1600.0, 500.0, 800.0),
            width_mm=200.0,
            height_mm=100.0,
            front_x_mm=1500.0,
            top_z_mm=850.0,
            n_points=500,
            back_x_mm=1700.0,
        )
        assert extent_of(geo) == extent_of(HAT)


def test_two_neighbours_a_hand_apart_are_beside_not_stacked():
    """Regression: the stacking guard must not borrow the `on` margin.

    That margin absorbs depth noise when deciding "is this resting on that". Applied to a
    lateral test it swallows an 80 mm gap, and two boxes sitting side by side on a table —
    the exact case `beside` exists for — get called a stack and rejected.
    """
    a = {
        "center_mm": [1600.0, 100.0, 850.0],
        "front_x_mm": 1500.0,
        "back_x_mm": 1700.0,
        "width_mm": 200.0,
        "top_z_mm": 950.0,
        "height_mm": 200.0,
    }
    b = {
        "center_mm": [1600.0, 260.0, 850.0],
        "front_x_mm": 1500.0,
        "back_x_mm": 1700.0,
        "width_mm": 200.0,
        "top_z_mm": 950.0,
        "height_mm": 200.0,
    }
    assert relation_holds(a, b, "beside")


# The drawer case: "get me an apple from the drawer" carries the apple's location in the
# instruction. Without containment the only way to say it was `near`, which is not what the
# task said — and `on` came out true for something sitting down inside, which would aim a
# grasp at the closed lid.
DRAWER = {"center_mm": [600.0, 0.0, 700.0], "front_x_mm": 500.0, "back_x_mm": 800.0,
          "width_mm": 400.0, "top_z_mm": 780.0, "height_mm": 160.0}
APPLE_INSIDE = {"center_mm": [620.0, 30.0, 710.0], "front_x_mm": 590.0, "back_x_mm": 650.0,
                "width_mm": 60.0, "top_z_mm": 740.0, "height_mm": 60.0}
APPLE_ON_LID = {"center_mm": [620.0, 0.0, 830.0], "front_x_mm": 590.0, "back_x_mm": 650.0,
                "width_mm": 60.0, "top_z_mm": 860.0, "height_mm": 60.0}


class TestContainment:
    def test_in_is_in_the_closed_set(self):
        assert "in" in SPATIAL_RELATIONS

    def test_an_apple_down_inside_a_drawer_is_in_it(self):
        assert relation_holds(APPLE_INSIDE, DRAWER, "in")

    def test_in_and_on_are_mutually_exclusive(self):
        """Something inside is not on top, and vice versa: a grasp planned off the wrong
        one descends onto the lid instead of into the drawer."""
        assert not relation_holds(APPLE_INSIDE, DRAWER, "on")
        assert relation_holds(APPLE_ON_LID, DRAWER, "on")
        assert not relation_holds(APPLE_ON_LID, DRAWER, "in")

    def test_being_inside_does_not_make_it_under(self):
        assert not relation_holds(APPLE_INSIDE, DRAWER, "under")

    def test_the_rim_gets_no_noise_margin(self):
        """The horizontal margin absorbs footprint noise; allowing the same slack ABOVE the
        rim is what let a thing resting on the lid read as contained."""
        just_above = dict(APPLE_ON_LID, top_z_mm=DRAWER["top_z_mm"] + 1.0,
                          center_mm=[620.0, 0.0, DRAWER["top_z_mm"] - 29.0])
        assert not relation_holds(just_above, DRAWER, "in")
