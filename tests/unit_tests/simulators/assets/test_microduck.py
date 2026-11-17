"""Official MJCF assets remain discoverable without claiming Agent support."""

from pathlib import Path

import pytest
import yaml

from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary, build_index, load_config
from jiuwensymbiosis.simulators.assets.config import RootSpec
from jiuwensymbiosis.simulators.assets.robot_assets import register_robot
from jiuwensymbiosis.simulators.assets.scanners import scan_root

MODEL_DIR = "src/mjlab_microduck/robot/microduck"


def checkout(root):
    for name in (f"{MODEL_DIR}/robot_walk.xml", f"{MODEL_DIR}/scene_walk.xml", "LICENSE", "LICENSE-MODELS.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    return root


def test_microduck_is_indexed_without_enabling_control_or_preview(tmp_path):
    root = checkout(tmp_path / "upstream")
    config = tmp_path / "assets.yaml"
    index = tmp_path / "assets.sqlite"
    config.write_text(
        yaml.safe_dump({"index": str(index), "roots": [{"id": "duck", "path": str(root), "scanner": "microduck"}]})
    )
    build_index(load_config(config))
    library = AssetLibrary(index)
    page = library.search(query="Microduck", kind="robot")
    assert page.total == 1
    asset = page.items[0]
    assert asset.id == f"library://duck/{MODEL_DIR}/robot_walk"
    assert asset.preview_mode is None
    resolved = library.resolve(asset.id)
    assert resolved.paths["source"] == root / MODEL_DIR / "robot_walk.xml"
    assert resolved.paths["model_license"] == root / "LICENSE-MODELS.md"
    # Stale source files must fail at resolution rather than reaching a loader.
    resolved.paths["source"].write_text("changed")
    with pytest.raises(AssetError, match="已变化"):
        library.resolve(asset.id)


def test_microduck_refuses_incomplete_or_escaping_checkout(tmp_path):
    root = checkout(tmp_path / "upstream")
    spec = RootSpec(id="duck", path=str(root), scanner="microduck")
    license_file = root / "LICENSE-MODELS.md"
    license_file.unlink()
    with pytest.raises(AssetError, match="缺失"):
        scan_root(spec, root)
    license_file.symlink_to(Path(__file__).resolve())
    with pytest.raises(AssetError, match="离开根目录"):
        scan_root(spec, root)


def test_microduck_policy_registration_admits_only_complete_bundle(tmp_path):
    root = checkout(tmp_path / "upstream")
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="下载官方"):
        register_robot("microduck", root, home)
    weight = root / "policies/velstand.onnx"
    weight.parent.mkdir()
    weight.write_text("fixture")
    with pytest.raises(AssetError, match="配套文件缺失"):
        register_robot("microduck", root, home)
    for name in ("policies/manifest.json", "scripts/infer_policy.py", f"{MODEL_DIR}/scene.xml"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture")
    result = register_robot("microduck", root, home)
    assert register_robot("microduck", root, home) == result
    config = load_config(result["config"])
    library = AssetLibrary(config.index)
    resolved = library.resolve(result["scene_id"])
    assert resolved.asset.preview_mode == "microduck"
    assert resolved.paths["walking_policy"] == weight
    assert result["robot_mode"] == "microduck"
    assert len(config.spec.roots) == 1
