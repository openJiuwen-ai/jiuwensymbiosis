"""Local YAML defaults and literal asset paths do not depend on shell configuration."""

from pathlib import Path

import pytest
import yaml

from jiuwensymbiosis.simulators.assets import AssetError, load_config
from jiuwensymbiosis.simulators.assets import config as asset_config


def test_default_config_is_fixed_and_missing_config_explains_setup(tmp_path, monkeypatch):
    source = tmp_path / "configs/simulators/assets.yaml"
    monkeypatch.setattr(asset_config, "DEFAULT_CONFIG_PATH", source)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(AssetError, match=r"init-config"):
        load_config()
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump({"index": "index.sqlite", "roots": [{"id": "demo", "scanner": "geniesim", "path": "."}]})
    )
    config = load_config()
    assert config.source == source
    assert config.roots == {"demo": source.parent}
    assert config.index == source.parent / "index.sqlite"


def test_asset_and_index_paths_do_not_expand_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSET_PATH_TEST", str(tmp_path / "expanded"))
    source = tmp_path / "custom.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "index": "${ASSET_PATH_TEST}/index.sqlite",
                "roots": [{"id": "demo", "scanner": "geniesim", "path": "$ASSET_PATH_TEST"}],
            }
        )
    )
    config = load_config(source)
    assert config.roots["demo"] == tmp_path / "$ASSET_PATH_TEST"
    assert config.index == tmp_path / "${ASSET_PATH_TEST}/index.sqlite"


def test_example_is_valid_and_only_registers_geniesim():
    repository = Path(__file__).resolve().parents[4]
    config = load_config(repository / "jiuwensymbiosis/simulators/assets/data/assets.example.yaml")
    assert [(root.id, root.scanner) for root in config.spec.roots] == [("geniesim", "geniesim")]
    assert set(config.spec.model_dump()) == {"version", "index", "roots"}


def test_initialize_config_never_overwrites_existing_configuration(tmp_path):
    from jiuwensymbiosis.simulators.assets import initialize_config

    source = tmp_path / "config/assets.yaml"
    assert initialize_config(source) == source
    assert load_config(source).spec.roots[0].id == "geniesim"
    source.write_text("private existing config")
    with pytest.raises(AssetError, match="未覆盖"):
        initialize_config(source)
    assert source.read_text() == "private existing config"
