"""Camera display configuration without modifying source materials."""

import pytest


def test_home_profile_selects_neutral_and_preserves_source_exposure():
    from jiuwensymbiosis.simulators.genesis.appearance import load_appearance_profile

    profile = load_appearance_profile("library://geniesim/background/home/home_0/background")
    assert profile.color_management.effective_parameters(-8) == {
        "tone_mapper": "neutral",
        "base_exposure_ev": -8.0,
        "exposure_adjust_ev": 0.0,
        "effective_exposure_ev": -8.0,
    }


def test_scene_color_profile_preserves_defaults_and_reads_only_once(monkeypatch, tmp_path):
    from jiuwensymbiosis.simulators.genesis import appearance

    path = tmp_path / "appearance.yaml"
    path.write_text(
        "scenes:\n  indoor:\n    color_management:\n      tone_mapper: neutral\n      exposure_adjust_ev: 0.25\n"
    )
    monkeypatch.setattr(appearance, "PROFILE_PATH", path)
    default = appearance.load_appearance_profile("other")
    assert default.color_management.effective_parameters(-8) == {
        "tone_mapper": "reinhard",
        "base_exposure_ev": -8.0,
        "exposure_adjust_ev": 0.0,
        "effective_exposure_ev": -8.0,
    }
    profile = appearance.load_appearance_profile("indoor")
    # The parsed profile is the render build's snapshot, even if the file changes.
    path.write_text("invalid: document")
    assert profile.color_management.effective_parameters(-8) == {
        "tone_mapper": "neutral",
        "base_exposure_ev": -8.0,
        "exposure_adjust_ev": 0.25,
        "effective_exposure_ev": -7.75,
    }


@pytest.mark.parametrize(
    "document",
    [
        None,
        {"scenes": [], "extra": True},
        {"scenes": {"indoor": None}},
        {"scenes": {"indoor": {"typo": True}}},
        {"scenes": {"indoor": {"color_management": None}}},
        {"scenes": {"indoor": {"color_management": {"tonemapper": "neutral"}}}},
        {"scenes": {"indoor": {"color_management": {"tone_mapper": "DLSS"}}}},
        {"scenes": {"indoor": {"color_management": {"tone_mapper": []}}}},
        {"scenes": {"indoor": {"color_management": {"exposure_adjust_ev": "1"}}}},
        {"scenes": {"indoor": {"color_management": {"exposure_adjust_ev": True}}}},
        {"scenes": {"indoor": {"color_management": {"exposure_adjust_ev": float("inf")}}}},
        {"scenes": {"indoor": {"color_management": {"exposure_adjust_ev": float("nan")}}}},
    ],
)
def test_invalid_appearance_profile_is_rejected(monkeypatch, tmp_path, document):
    import yaml

    from jiuwensymbiosis.simulators.genesis import appearance

    path = tmp_path / "appearance.yaml"
    path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(appearance, "PROFILE_PATH", path)
    with pytest.raises(ValueError):
        appearance.load_appearance_profile("indoor")
