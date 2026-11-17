"""Shared display configuration must not select policies by asset identity."""

import pytest


def test_shared_profile_selects_neutral_and_preserves_source_exposure():
    from jiuwensymbiosis.simulators.genesis.appearance import load_appearance_profile

    profile = load_appearance_profile()
    assert profile.color_management.effective_parameters(-8) == {
        "tone_mapper": "neutral",
        "base_exposure_ev": -8.0,
        "exposure_adjust_ev": 0.0,
        "effective_exposure_ev": -8.0,
    }


def test_profile_is_a_snapshot(monkeypatch, tmp_path):
    from jiuwensymbiosis.simulators.genesis import appearance

    path = tmp_path / "appearance.yaml"
    path.write_text("color_management:\n  tone_mapper: neutral\n  exposure_adjust_ev: 0.25\n")
    monkeypatch.setattr(appearance, "PROFILE_PATH", path)
    profile = appearance.load_appearance_profile()
    path.write_text("invalid: document")
    assert profile.color_management.effective_parameters(-8)["effective_exposure_ev"] == -7.75


@pytest.mark.parametrize(
    "document",
    [
        None,
        {"scenes": {}},
        {"color_management": None},
        {"color_management": {"tonemapper": "neutral"}},
        {"color_management": {"tone_mapper": "DLSS"}},
        {"color_management": {"tone_mapper": []}},
        *[{"color_management": {"exposure_adjust_ev": value}} for value in ["1", True, float("inf"), float("nan")]],
    ],
)
def test_invalid_or_scene_specific_profile_rejected(monkeypatch, tmp_path, document):
    import yaml

    from jiuwensymbiosis.simulators.genesis import appearance

    path = tmp_path / "appearance.yaml"
    path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(appearance, "PROFILE_PATH", path)
    with pytest.raises(ValueError):
        appearance.load_appearance_profile()
