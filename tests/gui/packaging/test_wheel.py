"""Installed CLI and workbench resources must work outside the source checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement


def test_installed_wheel_entries_and_workbench_resources(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "Open_Source_Software_Notice.txt"):
        shutil.copy2(repo / name, source / name)
    for name in ("jiuwensymbiosis", "jiuwensymbiosis_gui", "examples", "scripts"):
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("JIUWENSYMBIOSIS_SIMULATOR_HOME", None)

    def run(args, *, cwd=tmp_path):
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(tmp_path / "dist"),
            str(source),
        ]
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with ZipFile(wheel) as distribution:
        metadata_name = next(name for name in distribution.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(distribution.read(metadata_name))
        extras = metadata.get_all("Provides-Extra", [])
        assert "genesis" in extras and "genesis-macos" not in extras
        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        for platform, system, machine in (("darwin", "Darwin", "arm64"), ("linux", "Linux", "x86_64")):
            environment = {
                **default_environment(),
                "sys_platform": platform,
                "platform_system": system,
                "platform_machine": machine,
                "extra": "genesis",
            }
            selected = [req for req in requirements if req.marker is None or req.marker.evaluate(environment)]
            (torch,) = [req for req in selected if req.name == "torch"]
            (sdk,) = [req for req in selected if req.name == "genesis-world"]
            if platform == "linux":
                assert "2.11.0+cu130" in torch.specifier
                assert "2.11.0+cpu" not in torch.specifier and "2.11.0" not in torch.specifier
                assert sdk.extras == {"render", "usd"}
                assert {"gs-nyx", "gs-nyx-plugin"} <= {req.name for req in selected}
            else:
                assert "2.11.0" in torch.specifier and not sdk.extras
                assert {"gs-nyx", "gs-nyx-plugin"}.isdisjoint(req.name for req in selected)
        assert "jiuwensymbiosis/simulators/assets/robot_assets.py" in distribution.namelist()
        for removed in ("jumper_sim.py", "microduck_sim.py", "_robot_assets.py"):
            assert "examples/" + removed not in distribution.namelist()
        for name in distribution.namelist():
            assert not name.endswith((".onnx", ".rknn", ".so", ".pyd", ".stl", ".STL")), name
            assert "/outputs/vendor/" not in name, name
    installed = tmp_path / "installed"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--ignore-installed",
            "--target",
            str(installed),
            str(wheel),
        ]
    )
    # Remove the build tree too, so package-data fallback cannot hide omissions.
    shutil.rmtree(source)
    probe = """
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[1])
user_dir = root.parent / "user"
Path.home = classmethod(lambda cls: user_dir)
import examples
import jiuwensymbiosis
import jiuwensymbiosis_gui
from jiuwensymbiosis_gui.workbench import registry
for module in (examples, jiuwensymbiosis, jiuwensymbiosis_gui):
    assert Path(module.__file__).is_relative_to(root), module.__file__
gui = root / "jiuwensymbiosis_gui" / "workbench"
for filename in ("bodies.yaml", "tasks.yaml", "calibration_profiles.yaml"):
    assert (gui / "data" / filename).is_file(), filename
for filename in ("favicon.svg", "app_icon.png"):
    assert (gui / "assets" / filename).stat().st_size > 0, filename
for body in registry.list_bodies():
    config = body.config_path()
    assert config.is_relative_to(user_dir), config
    packaged = gui / "data" / "configs" / body.key / config.name
    assert config.read_bytes() == packaged.read_bytes()
dist = importlib.metadata.distribution("jiuwensymbiosis")
entry = next(e for e in dist.entry_points if e.name == sys.argv[2])
sys.argv = [entry.name, "--help"]
entry.load()()
"""
    for entry in ("jiuwensymbiosis-run", "jiuwensymbiosis-gui"):
        assert "--config" in run([sys.executable, "-I", "-c", probe, str(installed), entry])
    env["PYTHONPATH"] = str(installed)
    assert "--gui" in run([sys.executable, "-m", "jiuwensymbiosis.gui", "--help"])

    assets_probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[1])
user_dir = root.parent / "user"
Path.home = classmethod(lambda cls: user_dir)
import yaml
from jiuwensymbiosis.simulators.assets import AssetLibrary, build_index, initialize_config, load_config
from jiuwensymbiosis.simulators.assets import config as asset_config
assert Path(asset_config.__file__).is_relative_to(root)
config_path = asset_config.default_config_path()
assert config_path == user_dir / ".jiuwensymbiosis/simulators/configs/simulators/assets.yaml"
assert not config_path.exists()
initialize_config()
assert load_config().index == AssetLibrary().index
source = user_dir / "assets/background/room/demo/background.usda"
source.parent.mkdir(parents=True)
source.write_text('#usda 1.0\\n')
data = yaml.safe_load(config_path.read_text())
data["roots"][0]["path"] = str(user_dir / "assets")
config_path.write_text(yaml.safe_dump(data))
assert build_index(load_config())["total"] == 1
library = AssetLibrary()
assert library.index.is_relative_to(user_dir)
assert library.resolve(library.search().items[0].id).paths["source"] == source
assert not (root / "outputs").exists()
assert not (root / "configs").exists()
"""
    run([sys.executable, "-I", "-c", assets_probe, str(installed)])
    env["JIUWENSYMBIOSIS_SIMULATOR_HOME"] = str(tmp_path / "custom-data")
    assets_cli = installed / "jiuwensymbiosis/simulators/assets"
    run([sys.executable, "-I", str(assets_cli), "init-config"])
    assert (tmp_path / "custom-data/configs/simulators/assets.yaml").is_file()
    isolated_probe = """
import sys
sys.path.insert(0, sys.argv[1])
import assets
assert not {"jiuwensymbiosis", "openjiuwen", "torch", "genesis", "pxr"} & sys.modules.keys()
"""
    run([sys.executable, "-I", "-c", isolated_probe, str(assets_cli.parent)])

    motion_probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from jiuwensymbiosis.simulators.genesis import OrbitCamera, PreviewSettings
from jiuwensymbiosis.simulators.genesis import omni_base
from jiuwensymbiosis.simulators.common import base_motion
root = Path(sys.argv[1])
assert Path(base_motion.__file__).is_relative_to(root)
assert Path(omni_base.__file__).is_relative_to(root)
base_motion.validate_move(0.1, 0.2, 0.3)
omni_base.validate_twist(0.1, 0.1, 0.1)
assert OrbitCamera().snapshot()["up"] == [0.0, 0.0, 1.0]
assert PreviewSettings().dt > 0
assert "genesis" not in sys.modules
"""
    run([sys.executable, "-I", "-c", motion_probe, str(installed)])

    scene_probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from jiuwensymbiosis.simulators.genesis import appearance, material_baking, material_repair, material_surface, source_lighting, usd_probe
from jiuwensymbiosis.simulators.genesis.material_rules import interioragent
root = Path(sys.argv[1])
for module in (appearance, material_baking, material_repair, material_surface, interioragent, source_lighting, usd_probe):
    assert Path(module.__file__).is_relative_to(root)
for path in (appearance.PROFILE_PATH, source_lighting.PROFILE_PATH):
    assert path.is_relative_to(root) and path.is_file()
assert appearance.load_appearance_profile("unlisted").color_management.tone_mapper == "reinhard"
assert source_lighting.load_source_lighting(root / "not-needed.usda", "unlisted") is None
assert "genesis" not in sys.modules
"""
    run([sys.executable, "-I", "-c", scene_probe, str(installed)])
    # A standalone entry must also work with no source checkout or core import.
    standalone_probe = """
import sys
sys.path.insert(0, sys.argv[1])
import usd_probe
from scene_catalog import material_rules_for_asset
from types import SimpleNamespace
assert material_rules_for_asset(SimpleNamespace(asset=SimpleNamespace(material_rules="interioragent")))
assert not material_rules_for_asset(SimpleNamespace(asset=SimpleNamespace(root_id="interioragent")))
assert not {"jiuwensymbiosis", "openjiuwen", "torch", "genesis", "pxr"} & sys.modules.keys()
sys.argv = ["usd_probe", "--help"]
usd_probe.main()
"""
    assert "--inventory" in run(
        [sys.executable, "-I", "-c", standalone_probe, str(installed / "jiuwensymbiosis/simulators/genesis")]
    )

    media_probe = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[1])
from jiuwensymbiosis.simulators.common import preview
from jiuwensymbiosis.simulators.common import stream
from jiuwensymbiosis.simulators.genesis import robot_cameras, robot_materials
from jiuwensymbiosis.simulators.common import startup_diagnostics
for module in (preview, stream, robot_cameras, robot_materials, startup_diagnostics):
    assert Path(module.__file__).is_relative_to(root)
assert preview._WORKER.is_relative_to(root) and preview._WORKER.is_file()
assert (preview._WORKER.parent / "web/assets/metrics.mjs").is_file()
assert not {"genesis", "torch", "gs_nyx", "aiortc", "av"} & sys.modules.keys()
"""
    run([sys.executable, "-I", "-c", media_probe, str(installed)])
    gui_simulator_probe = """
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ.pop("JIUWENSYMBIOSIS_SIMULATOR_HOME", None)
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[1])
user_dir = root.parent / "simulator-user"
Path.home = classmethod(lambda cls: user_dir)
from jiuwensymbiosis.simulators.common import agent
from jiuwensymbiosis.simulators.common import vision
from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.simulators.common.web.routes import install_simulation_routes
from fastapi import FastAPI
from fastapi.testclient import TestClient
assets = root / "jiuwensymbiosis/simulators/common/web/assets"
for filename in ("viewer.html", "viewer.css", "viewer.js", "navigation.mjs", "selection.mjs", "workbench.css"):
    assert (assets / filename).stat().st_size > 0, filename
result = vision.save_result({"jpeg": b"filesystem-probe"}, "probe", model="stub", duration_s=0)
assert vision.OUTPUT_DIR.is_relative_to(user_dir)
assert vision.latest_result()["id"] == result["id"]
preview = SimpleNamespace(status=lambda: {"state": "ready", "cameras": []})
with patch.object(agent, "build_model", return_value=object()), patch.object(agent, "run_robot_task", return_value={"output": "probe"}):
    result = agent.run_agent_query(preview, ModelSpec(), "probe", actions_enabled=False)
output = user_dir / ".jiuwensymbiosis/simulators/outputs/genesis-agent"
assert (output / (result["id"] + ".json")).is_file()
app = FastAPI()
install_simulation_routes(app)
with TestClient(app) as client:
    assert client.get("/genesis/viewer").status_code == 200
    assert client.get("/genesis/assets/viewer.js").status_code == 200
assert not (root / "outputs").exists()
assert not {"genesis", "torch", "gs_nyx", "aiortc", "av"} & sys.modules.keys()
"""
    run([sys.executable, "-I", "-c", gui_simulator_probe, str(installed)])
    worker_probe = """
import sys
sys.path.insert(0, sys.argv[1])
import worker
assert not {"jiuwensymbiosis", "openjiuwen", "torch", "genesis", "gs_nyx", "av"} & sys.modules.keys()
"""
    run([sys.executable, "-I", "-c", worker_probe, str(installed / "jiuwensymbiosis/simulators/common")])
