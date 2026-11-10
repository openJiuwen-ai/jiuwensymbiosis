"""Export scripts must coexist with SDKs without core import hooks or path edits."""

import subprocess
import sys
from pathlib import Path


def test_export_loader_isolated_from_core_and_same_named_modules(tmp_path):
    root = Path(__file__).resolve().parents[4]
    helper = root / "scripts/simulator/genesis/_bootstrap.py"
    probe = """
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

entry = Path(sys.argv[1])
original_path = list(sys.path)
for name in ('camera', 'indoor', 'occupancy', 'scene_catalog', 'genesis'):
    sys.modules[name] = ModuleType(name)
spec = importlib.util.spec_from_file_location('export_bootstrap', entry)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
for name, attribute in (
    ('camera', 'OrbitCamera'), ('indoor', 'read_room_meshes'),
    ('occupancy', 'build_occupancy'), ('scene_catalog', 'AssetLibrary'),
    ('rendering', 'SceneRenderer'),
):
    module = bootstrap.load_genesis_module(name)
    assert hasattr(module, attribute), (name, attribute)
    assert module is bootstrap.load_genesis_module(name)
assert sys.path == original_path
assert 'openjiuwen' not in sys.modules
assert 'jiuwensymbiosis' not in sys.modules
assert not hasattr(sys.modules['genesis'], 'OrbitCamera')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(helper)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
